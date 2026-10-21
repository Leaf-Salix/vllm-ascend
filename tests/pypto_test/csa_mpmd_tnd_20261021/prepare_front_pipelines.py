# SPDX-License-Identifier: Apache-2.0
"""把独立Indexer链与主压缩/Q_B链各自组织为局部MIX，保留外层多task。"""

import ast
import copy
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from prepare_output_group import make_function
from prepare_priority_fusion import scope_name


def parse(text):
    return ast.parse(textwrap.dedent(text)).body


class LocalWorkers(ast.NodeTransformer):
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id in ("GROUPS", "PREFIX_WORKERS"):
                return ast.Name(id="FRONT_GROUPS", ctx=node.ctx)
            if node.id == "VECTORS":
                return ast.parse("2 * FRONT_GROUPS", mode="eval").body
        return node

    def visit_Call(self, node):
        if ast.unparse(node.func) == "pl.tile.get_block_idx":
            return ast.Name(id="worker", ctx=ast.Load())
        return self.generic_visit(node)


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    entry_path = target / "entry.py"
    source = entry_path.read_text()
    module = ast.parse(source)
    root = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_tnd")
    gate = next(n for n in root.body if isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_DAG")
    scopes = {scope_name(n): n for n in gate.body if scope_name(n)}
    arguments = {a.arg: copy.deepcopy(a) for a in root.args.args}
    query = ast.parse((target / "query_group.py").read_text())
    query_entry = next(n for n in query.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_query_chain")
    # 两个head视图由外层编排创建，属于局部链参数而非顶层Tensor形参。
    for arg in query_entry.args.args:
        arguments.setdefault(arg.arg, copy.deepcopy(arg))
    for label in ("idx", "main"):
        text = f"def f({label}_front_signal: pl.InOut[pl.Tensor[[{label.upper()}_FRONT_WORDS], pl.INT32]]): pass"
        arguments[label + "_front_signal"] = ast.parse(text).body[0].args.args[0]
    arguments["FRONT_GROUPS"] = ast.parse("def f(FRONT_GROUPS: pl.constexpr): pass").body[0].args.args[0]

    def body(scope):
        return [LocalWorkers().visit(copy.deepcopy(n)) for n in scopes[scope].body]

    def initialize(signal, vector):
        text = "worker = pl.tile.get_block_idx()\n"
        if vector:
            text = (
                "pl.func_attr({'split_aiv': True})\nworker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()\n"
            )
        text += "core = " + ("FRONT_GROUPS + worker" if vector else "worker") + "\n"
        text += "tokens = pl.tensor.dim(positions, 0)\n"
        text += f"({signal}, generation) = begin_invocation({signal}, core)\n"
        return parse(text)

    def publish(signal, phase):
        return parse(f"{signal} = publish_phase({signal}, {phase}, core, 3 * FRONT_GROUPS, generation)")

    def wait(signal, phase, units, vector=False):
        workers, first = ("2 * FRONT_GROUPS", "FRONT_GROUPS") if vector else ("FRONT_GROUPS", "0")
        return parse(
            f"{signal} = wait_row_owners({signal}, {phase}, 3 * FRONT_GROUPS, generation, "
            f"0, {units}, 1, {workers}, {first}, 1)"
        )

    # Indexer AIC：先发压缩和query投影，再复用等RoPE期间计算weights与compact Hadamard。
    index_cube = initialize("idx_front_signal", False)
    index_cube += body("indexer_compressor_kv_score") + publish("idx_front_signal", "I_PROJECT")
    index_cube += parse(
        "idx_query_acc = indexer_query_project_stage(qr, idx_wq_b, idx_query_acc, worker, FRONT_GROUPS)"
    )
    index_cube += publish("idx_front_signal", "I_QUERY_PROJECT")
    index_cube += parse(
        "weights_partial = indexer_weights_project_stage(normed, weights_proj, weights_partial, worker, FRONT_GROUPS)"
    )
    index_cube += publish("idx_front_signal", "I_WEIGHTS")
    index_cube += wait("idx_front_signal", "I_NORM", "pl.tensor.dim(idx_lengths, 0)", True)
    index_cube += body("indexer_compressor_hadamard") + publish("idx_front_signal", "I_HAD")
    index_cube += wait(
        "idx_front_signal",
        "I_QUERY_ROPE",
        "((tokens + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE) * (IDX_N_HEADS // DQ_ROPE_H_TILE)",
        True,
    )
    index_cube += parse(
        "idx_query_had = _query_hadamard_four_tokens("
        "idx_query_rope_heads, hadamard, idx_query_had, tokens, worker, FRONT_GROUPS)"
    )
    index_cube += publish("idx_front_signal", "I_QUERY_HAD")

    # Indexer AIV：POOL发布在任何state写之前；仅消费同owner pooled行做norm。
    index_vector = initialize("idx_front_signal", True)
    index_vector += wait(
        "idx_front_signal",
        "I_PROJECT",
        "(INDEXER_OUT_DIM // INDEXER_N_TILE) * ((tokens + INDEXER_M_TILE - 1) // INDEXER_M_TILE)",
    )
    pool = body("indexer_pool_norm_rope")
    assert len(pool) == 4 and ast.unparse(pool[2]) == "pl.system.bar_all()"
    index_vector += pool[:2] + publish("idx_front_signal", "I_POOL") + pool[3:]
    index_vector += publish("idx_front_signal", "I_NORM")
    index_vector += wait("idx_front_signal", "I_POOL", "pl.tensor.dim(idx_lengths, 0)", True)
    index_vector += body("indexer_state_commit")
    index_vector += wait("idx_front_signal", "I_QUERY_PROJECT", "IDX_N_HEADS * IDX_HEAD_DIM // QR_MM_N_TILE")
    index_vector += parse(
        "idx_query_rope = indexer_query_dequant_rope_stage("
        "qr_scale, idx_wq_b_scale, rope_cos, rope_signed, idx_query_acc, idx_query_rope, worker, 2 * FRONT_GROUPS)"
    )
    index_vector += publish("idx_front_signal", "I_QUERY_ROPE")
    index_vector += wait("idx_front_signal", "I_WEIGHTS", "WEIGHTS_OK * ((tokens + MM_ROW_TILE - 1) // MM_ROW_TILE)")
    index_vector += parse(
        "head_weights = indexer_weights_reduce_stage(weights_partial, head_weights, tokens, worker, 2 * FRONT_GROUPS)"
    )
    index_vector += wait(
        "idx_front_signal",
        "I_HAD",
        "(COMPACT_HEAD_DIM // COMPACT_HAD_COL_TILE) * "
        "((pl.cast(pl.read(idx_count, [0]), pl.INDEX) + COMPACT_HAD_ROW_TILE - 1) // COMPACT_HAD_ROW_TILE)",
    )
    index_vector += body("indexer_compressor_cache_write") + publish("idx_front_signal", "I_QUANT")
    index_vector += wait("idx_front_signal", "I_QUANT", "pl.tensor.dim(idx_lengths, 0)", True)
    # 页计划保持原48 logical writer布局；本组AIV轮转接管，不能改成按request做64B RMW。
    index_vector += parse("""
        for scale_owner in pl.range(worker, SCALE_WRITERS, 2 * FRONT_GROUPS):
            idx_cache = indexer_scale_page_commit_stage(
                idx_scales, group_offsets, idx_cache, tokens, pl.tensor.dim(idx_lengths, 0), scale_owner, SCALE_WRITERS
            )
    """)
    index_vector += wait("idx_front_signal", "I_QUERY_HAD", "(tokens + 3) // 4")
    index_vector += parse(
        "(idx_query_i8, idx_query_scale) = indexer_query_quant_stage("
        "idx_query_had, idx_query_i8, idx_query_scale, tokens, worker, 2 * FRONT_GROUPS)"
    )
    index_vector += publish("idx_front_signal", "I_QUERY_QUANT")
    index_vector += wait("idx_front_signal", "I_QUERY_QUANT", "tokens", True)
    index_vector += parse(
        "coefficients = indexer_head_coefficients_stage("
        "idx_query_scale_heads, head_weights, positions, idx_bounds, idx_lengths, group_offsets, "
        "coefficients, worker, 2 * FRONT_GROUPS, QUERY_GROUP)"
    )

    # 主链只融合主Compressor与Q_B；KV保留独立task，提前进入可用核队列。
    main_cube = initialize("main_front_signal", False)
    main_cube += body("compressor_kv_score") + publish("main_front_signal", "M_PROJECT")
    main_cube += body("qproj_b") + publish("main_front_signal", "M_QUERY")
    main_vector = initialize("main_front_signal", True)
    main_vector += wait(
        "main_front_signal", "M_PROJECT", "(CMP_OUT_DIM // CMP_N_TILE) * ((tokens + CMP_M_TILE - 1) // CMP_M_TILE)"
    )
    pool = body("compressor_pool_norm_cache")
    assert len(pool) == 4 and ast.unparse(pool[2]) == "pl.system.bar_all()"
    main_vector += pool[:2] + publish("main_front_signal", "M_POOL") + pool[3:]
    main_vector += wait("main_front_signal", "M_POOL", "pl.tensor.dim(main_lengths, 0)", True)
    main_vector += body("compressor_state_commit")
    main_vector += wait("main_front_signal", "M_QUERY", "QPROJ_N_BLOCKS")
    main_vector += body("qproj_dequant_rms_nope_rope")

    outputs = {
        "index_front_cube": [
            "idx_value",
            "idx_score",
            "idx_query_acc",
            "weights_partial",
            "idx_had",
            "idx_query_had",
            "idx_front_signal",
        ],
        "index_front_vector": [
            "idx_pooled",
            "idx_normed",
            "idx_state",
            "idx_query_rope",
            "head_weights",
            "idx_scales",
            "idx_debug",
            "idx_cache",
            "idx_query_i8",
            "idx_query_scale",
            "coefficients",
            "idx_front_signal",
        ],
        "main_front_cube": ["main_value", "main_score", "q_proj", "main_front_signal"],
        "main_front_vector": [
            "main_pooled",
            "main_normed",
            "main_debug",
            "main_cache",
            "main_state",
            "q_flat",
            "main_front_signal",
        ],
    }
    functions, parameters = {}, {}

    def register(name, statements, out, decorator):
        used = {n.id for n in ast.walk(ast.Module(body=statements, type_ignores=[])) if isinstance(n, ast.Name)}
        names = [k for k in arguments if k in used and k not in out and k != "FRONT_GROUPS"] + out + ["FRONT_GROUPS"]
        functions[name] = make_function(name, names, arguments, out, statements, decorator)
        parameters[name] = names

    for label, statements in (
        ("index_front_cube", index_cube),
        ("index_front_vector", index_vector),
        ("main_front_cube", main_cube),
        ("main_front_vector", main_vector),
    ):
        register(label, statements, outputs[label], "pl.jit.incore")
    for prefix in ("index", "main"):
        c, v = prefix + "_front_cube", prefix + "_front_vector"
        out = list(dict.fromkeys(outputs[c] + outputs[v]))
        label = prefix + "_front_chain"
        group = parse(
            f"with pl.cluster(name_hint='{label}'):\n"
            f"    ({', '.join(outputs[c])}) = {c}({', '.join(parameters[c])})\n"
            f"    ({', '.join(outputs[v])}) = {v}({', '.join(parameters[v])})\n"
        )
        register(label, group, out, "pl.jit.inline(auto_scope=False)")
        outputs[label] = out

    imports = [copy.deepcopy(n) for n in module.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    header = '''# SPDX-License-Identifier: Apache-2.0
"""两条独立前段链的局部流水，不合并整个CSA。

Indexer默认8 AIC/16 AIV，主压缩/Q_B默认16 AIC/32 AIV，共同填满硬件。
POOL阶段必须先发布再写state，避免覆盖其它角色尚未读完的历史；Norm只读本owner pooled。
Indexer scale保留48个logical page writer，当前AIV按条带接管，维持每页唯一64B RMW。
信号首次清零、各组顺序独占，每角色每调用推进一次代际；数值/图/性能以设备结果为准。
"""
'''
    constants = """
from .query_phase_sync import begin_invocation, publish_phase, wait_row_owners
from .query_group import _query_hadamard_four_tokens
from .indexer_query_stages import IDX_HEAD_DIM, QR_MM_N_TILE
from .query_projection_stages import QPROJ_N_BLOCKS
IDX_FRONT_WORDS = pl.dynamic("CSA_IDX_FRONT_WORDS")
MAIN_FRONT_WORDS = pl.dynamic("CSA_MAIN_FRONT_WORDS")
IDX_FRONT_PHASES = 11
MAIN_FRONT_PHASES = 4
SCALE_WRITERS = 48
I_PROJECT = 1
I_POOL = 2
I_NORM = 3
I_QUERY_PROJECT = 4
I_WEIGHTS = 5
I_HAD = 6
I_QUERY_ROPE = 7
I_QUERY_HAD = 8
I_QUANT = 9
I_QUERY_QUANT = 10
M_PROJECT = 1
M_POOL = 2
M_QUERY = 3
"""
    (target / "front_pipelines.py").write_text(
        header
        + "\n".join(ast.unparse(n) for n in imports)
        + constants
        + "\n\n"
        + "\n\n".join(ast.unparse(ast.fix_missing_locations(f)) for f in functions.values())
        + "\n"
    )

    removed = {
        "indexer_compressor_kv_score",
        "indexer_pool_norm_rope",
        "indexer_state_commit",
        "indexer_compressor_hadamard",
        "indexer_query_chain",
        "indexer_compressor_cache_write",
        "indexer_scale_page_commit",
        "compressor_kv_score",
        "compressor_pool_norm_cache",
        "compressor_state_commit",
        "qproj_b",
        "qproj_dequant_rms_nope_rope",
        "kv_proj",
        "kv_norm_rope",
        "csa_cache_writeback",
    }
    rebuilt = []
    for node in gate.body:
        label = scope_name(node)
        if label in removed:
            continue
        node = copy.deepcopy(node)
        if label == "index_score_topk":
            call = node.items[0].context_expr
            deps = next(k for k in call.keywords if k.arg == "deps")
            deps.value = ast.parse("[idx_cache_tid, main_front_tid, kv_cache_tid]", mode="eval").body
        rebuilt.append(node)
        if label == "qproj_a_rms_quant":
            for kv_label in ("kv_proj", "kv_norm_rope", "csa_cache_writeback"):
                kv_scope = copy.deepcopy(scopes[kv_label])
                if kv_label == "csa_cache_writeback":
                    kv_scope.items[0].optional_vars = ast.Name(id="kv_cache_tid", ctx=ast.Store())
                rebuilt.append(kv_scope)
            for prefix, bound, tid in (
                ("index", "IDX_FRONT_GROUPS", "idx_cache_tid"),
                ("main", "MAIN_FRONT_GROUPS", "main_front_tid"),
            ):
                name = prefix + "_front_chain"
                args = [bound if a == "FRONT_GROUPS" else a for a in parameters[name]]
                rebuilt += parse(
                    f"with pl.spmd({bound}, name_hint='{name}', allow_early_resolve=True, "
                    f"sync_start=True, deps=[metadata_tid]) as {tid}:\n"
                    f"    ({', '.join(outputs[name])}) = {name}({', '.join(args)})\n"
                )
    gate.body = rebuilt
    position = next(i for i, a in enumerate(root.args.args) if a.arg == "output_signal") + 1
    root.args.args[position:position] = [copy.deepcopy(arguments[k]) for k in ("idx_front_signal", "main_front_signal")]
    for key, value in (("IDX_FRONT_GROUPS", 8), ("MAIN_FRONT_GROUPS", 16)):
        arg = ast.arg(arg=key, annotation=ast.parse("pl.constexpr", mode="eval").body)
        root.args.args.append(arg)
        root.args.defaults.append(ast.Constant(value))
    for key in ("idx_front_signal", "main_front_signal"):
        root.body[-1].value.elts.append(ast.Name(id=key, ctx=ast.Load()))
    start = min([root.lineno] + [n.lineno for n in root.decorator_list]) - 1
    lines = source.splitlines(keepends=True)
    lines[start : root.end_lineno] = [ast.unparse(ast.fix_missing_locations(root)) + "\n"]
    source = "".join(lines).replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\n"
        "from .front_pipelines import IDX_FRONT_WORDS, MAIN_FRONT_WORDS, index_front_chain, main_front_chain\n",
        1,
    )
    entry_path.write_text(source)

    policy_path = target / "policy.py"
    policy = policy_path.read_text()
    needle = "    fused_dag: bool = True\n"
    assert policy.count(needle) == 1
    policy = policy.replace(needle, needle + "    idx_front_groups: int = 8\n    main_front_groups: int = 16\n")
    needle = '            "FUSED_DAG": self.fused_dag,\n'
    policy = policy.replace(
        needle,
        needle
        + '            "IDX_FRONT_GROUPS": self.idx_front_groups,\n'
        + '            "MAIN_FRONT_GROUPS": self.main_front_groups,\n',
    )
    policy_path.write_text(policy)
    binding_path = target / "binding.py"
    binding = binding_path.read_text()
    needle = '            "query_signal": ((QUERY_SIGNAL_COUNT if self.policy.fused_dag else 1,), torch.int32),\n'
    assert binding.count(needle) == 1
    binding = binding.replace(
        needle,
        '            "query_signal": ((1,), torch.int32),\n'
        '            "idx_front_signal": ((11 * 3 * self.policy.idx_front_groups * 128 '
        "if self.policy.fused_dag else 1,), torch.int32),\n"
        '            "main_front_signal": ((4 * 3 * self.policy.main_front_groups * 128 '
        "if self.policy.fused_dag else 1,), torch.int32),\n",
    )
    binding = binding.replace(
        'if name in ("query_signal", "output_signal"):',
        'if name in ("query_signal", "output_signal", "idx_front_signal", "main_front_signal"):',
    )
    binding_path.write_text(binding)
    print(target)


if __name__ == "__main__":
    main()
