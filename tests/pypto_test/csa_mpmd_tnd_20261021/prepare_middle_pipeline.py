# SPDX-License-Identifier: Apache-2.0
"""Score/Top-K/plan/Attention局部流水；按真实生产者等待，保留CSA其余task。"""

import ast
import copy
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_output_group import make_function, root_function
from prepare_priority_fusion import scope_name


def parse(source):
    return ast.parse(source).body


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    balanced_only = "--balanced-only" in sys.argv[3:]
    if any(value != "--balanced-only" for value in sys.argv[3:]):
        raise ValueError("未知参数；仅支持--balanced-only")
    frozen = here / "sources/84751e34/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    entry_path = target / "entry.py"
    source = entry_path.read_text()
    tree = ast.parse(source)
    root = root_function(source, "decode_csa_tnd")
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    gate = next(n for n in root.body if isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_DAG")
    scopes = {scope_name(n): n for n in gate.body if scope_name(n)}
    args = {a.arg: a for a in root.args.args}
    # idx_key0/idx_key64等只读别名在编排内部创建，类型来自现有incore签名。
    for label in ("index_score_topk_cube", "index_score_topk_vector", "sparse_attn_cube", "sparse_attn_vector"):
        for argument in functions[label].args.args:
            args.setdefault(argument.arg, argument)
    args["middle_signal"] = parse("def f(middle_signal: pl.InOut[pl.Tensor[[MIDDLE_SIGNAL_WORDS], pl.INT32]]): pass")[
        0
    ].args.args[0]

    phase = '''# SPDX-License-Identifier: Apache-2.0
"""局部Score到Attention的独立代际信号；不复用前段或输出链的工作区。"""
import pypto.language as pl
from .runtime_abi import CORES, GROUPS, VECTORS
from .query_phase_sync import SIGNAL_SLOT_WORDS, begin_invocation, publish_phase, wait_row_owners

SIGNAL_WORDS = pl.dynamic("CSA_MIDDLE_SIGNAL_WORDS")
SIGNAL_COUNT = 4 * CORES * SIGNAL_SLOT_WORDS
PHASE_SCORE_VECTOR = 1
PHASE_TOPK = 2
PHASE_ATTN_PLAN = 3
'''
    (target / "middle_phase.py").write_text(phase)
    for original, output in (
        ("attention_dependencies.py", "middle_dependencies.py"),
        ("indexer_publish_stages.py", "middle_publish_stages.py"),
        ("attention_plan_stages.py", "middle_plan_stages.py"),
    ):
        text = (frozen / original).read_text()
        text = text.replace("from .phase_sync import", "from .middle_phase import")
        text = text.replace("from .attention_dependencies import", "from .middle_dependencies import")
        text = text.replace("from .signals import", "from .query_signals import")
        if original in ("attention_dependencies.py", "attention_plan_stages.py"):
            text = text.replace("from .runtime_abi import", "from .middle_phase import")
        if original == "attention_dependencies.py":
            old = "item = leaf * groups + group"
            assert text.count(old) == 1
            text = text.replace(old, "item = leaf * groups + (group + groups - leaf % groups) % groups")
            text = text.replace(
                "Score item=leaf*groups+group；否则为group*leaves+leaf。",
                "Score按叶轮换group，反向item=leaf*groups+(group+groups-leaf%groups)%groups；\n"
                "    多组时仍为group*leaves+leaf。反向表达式不产生负数取余。",
            )
        (target / output).write_text(text)

    # 与已有Score的叶轮换枚举交叉核对等待owner；没有改本轮device计划。
    # 这项检查覆盖每个(group,leaf)，不依赖某次泳道上的实际执行顺序。
    for groups in range(1, 49):
        for leaves in range(1, 33):
            for item in range(groups * leaves):
                if groups < 24:
                    group, leaf = (item % groups + item // groups) % groups, item // groups
                    inverse = leaf * groups + (group + groups - leaf % groups) % groups
                else:
                    group, leaf = item // leaves, item % leaves
                    inverse = group * leaves + leaf
                assert inverse == item

    cube_parts = (functions["index_score_topk_cube"], functions["sparse_attn_cube"])
    vector_parts = (functions["index_score_topk_vector"], functions["sparse_attn_vector"])
    cube_parameters = list(dict.fromkeys(a.arg for f in cube_parts for a in f.args.args))
    cube_parameters += ["middle_signal"]
    cube_outputs = ["indexer_transfer", "score_ring", "pv_ring", "middle_signal"]
    cube_body = parse("""
worker = pl.tile.get_block_idx()
tokens = pl.tensor.dim(positions, 0)
(middle_signal, generation) = begin_invocation(middle_signal, worker)
pl.system.set_ffts(ffts)
""")
    cube_body += copy.deepcopy(cube_parts[0].body[-2:-1])
    cube_body += parse("middle_signal = wait_attention_plan_rows(middle_signal, tokens, worker, generation)")
    cube_body += copy.deepcopy(cube_parts[1].body[-2:-1])

    vector_parameters = list(dict.fromkeys(a.arg for f in vector_parts for a in f.args.args))
    for label in ("indexer_topk_query", "csa_attention_plan"):
        call = scopes[label].body[0].value
        for arg in call.args:
            if isinstance(arg, ast.Name) and arg.id in args and arg.id not in vector_parameters:
                vector_parameters.append(arg.id)
    vector_parameters += ["middle_signal"]
    vector_outputs = [
        "score_arena",
        "pair_arena",
        "topk_scores",
        "topk_indices",
        "attention_indices",
        "attention_bias",
        "attention_mask",
        "kv_ring",
        "probability_ring",
        "mi_ring",
        "li_ring",
        "packed",
        "middle_signal",
    ]
    # 当前JIT的_dep_out_metas按Out/InOut参数声明顺序向调用者传播metadata。
    # 显式return及接收目标也保持该顺序，避免把Top-K INT32类型传给BF16 packed。
    vector_outputs = [p for p in vector_parameters if p in vector_outputs]
    vector_body = parse("""
pl.func_attr({'split_aiv': True})
worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
core = GROUPS + worker
tokens = pl.tensor.dim(positions, 0)
(middle_signal, generation) = begin_invocation(middle_signal, core)
pl.system.set_ffts(ffts)
""")
    vector_body += copy.deepcopy(vector_parts[0].body[-2:-1])
    vector_body += parse("""
middle_signal = publish_phase(middle_signal, PHASE_SCORE_VECTOR, core, CORES, generation)
(topk_scores, topk_indices, middle_signal) = middle_pair_publish_stage(
    idx_bounds, idx_lengths, positions, group_plan, group_offsets, attention_indices, middle_signal,
    pair_arena, topk_scores, topk_indices, worker, VECTORS, ROOTS, BALANCED, generation,
    QUERY_GROUP, EARLY_ATTENTION_GEOMETRY)
middle_signal = publish_phase(middle_signal, PHASE_TOPK, core, CORES, generation)
(attention_indices, attention_bias, attention_mask, middle_signal) = middle_slot_plan_stage(
    topk_indices, positions, idx_bounds, idx_lengths, original_table, compressed_table,
    attention_indices, attention_bias, attention_mask, middle_signal, generation,
    pl.tensor.dim(cache_rows, 0) // 32, pl.tensor.dim(main_cache, 0) // 32, worker, VECTORS,
    ATTENTION_QUERY_METADATA, EARLY_ATTENTION_GEOMETRY, PLAN_PHYSICAL_ROWS)
middle_signal = publish_phase(middle_signal, PHASE_ATTN_PLAN, core, CORES, generation)
middle_signal = wait_attention_plan_rows(middle_signal, tokens, pl.tile.get_block_idx(), generation)
""")
    vector_body += copy.deepcopy(vector_parts[1].body[-2:-1])
    common_parameters = list(dict.fromkeys(cube_parameters + vector_parameters))
    common_outputs = [p for p in common_parameters if p in cube_outputs or p in vector_outputs]
    body = parse(
        "with pl.cluster(name_hint='score_attention_chain'):\n"
        f"    ({', '.join(cube_outputs)}) = middle_cube({', '.join(cube_parameters)})\n"
        f"    ({', '.join(vector_outputs)}) = middle_vector({', '.join(vector_parameters)})\n"
    )
    new_functions = [
        make_function("middle_cube", cube_parameters, args, cube_outputs, cube_body, "pl.jit.incore"),
        make_function("middle_vector", vector_parameters, args, vector_outputs, vector_body, "pl.jit.incore"),
        make_function(
            "score_attention_chain", common_parameters, args, common_outputs, body, "pl.jit.inline(auto_scope=False)"
        ),
    ]
    imports = [copy.deepcopy(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    # entry.py还未导入middle模块，复用其已验证的ABI和数学helper导入不形成循环。
    imports += parse("""
from .middle_phase import SIGNAL_WORDS as MIDDLE_SIGNAL_WORDS
from .middle_phase import CORES, PHASE_SCORE_VECTOR, PHASE_TOPK, PHASE_ATTN_PLAN
from .middle_phase import begin_invocation, publish_phase
from .middle_dependencies import wait_attention_plan_rows
from .middle_publish_stages import indexer_pair_publish_stage as middle_pair_publish_stage
from .middle_plan_stages import attention_slot_plan_stage as middle_slot_plan_stage
""")
    module = ast.Module(body=imports + new_functions, type_ignores=[])
    (target / "middle_pipeline.py").write_text(
        '# SPDX-License-Identifier: Apache-2.0\n"""Score到Attention局部流水，所有外部输入由TMR保证就绪。"""\n'
        + ast.unparse(ast.fix_missing_locations(module))
        + "\n"
    )

    replacement = copy.deepcopy(scopes["index_score_topk"])
    call = replacement.items[0].context_expr
    next(k for k in call.keywords if k.arg == "name_hint").value = ast.Constant("score_attention_chain")
    replacement.items[0].optional_vars = ast.Name(id="attn_tid", ctx=ast.Store())
    replacement.body = parse(f"({', '.join(common_outputs)}) = score_attention_chain({', '.join(common_parameters)})")
    rebuilt = []
    middle_labels = ("index_score_topk", "indexer_topk_query", "csa_attention_plan", "sparse_attn_csa")
    original_middle = [copy.deepcopy(n) for n in gate.body if scope_name(n) in middle_labels]
    # TaskId不跨constexpr分支逃逸；消费它的输出task随生产者留在同一分支。
    output_scope = scopes["output_group_chain"]
    for node in gate.body:
        label = scope_name(node)
        if label == "index_score_topk":
            rebuilt.append(
                ast.If(
                    test=ast.Name(id="BALANCED", ctx=ast.Load()),
                    body=[replacement, copy.deepcopy(output_scope)],
                    orelse=original_middle + [copy.deepcopy(output_scope)],
                )
                if balanced_only
                else replacement
            )
        elif balanced_only and label == "output_group_chain":
            continue
        elif label not in ("indexer_topk_query", "csa_attention_plan", "sparse_attn_csa"):
            rebuilt.append(node)
    gate.body = rebuilt
    required_count = len(root.args.args) - len(root.args.defaults)
    root.args.args.insert(required_count, args["middle_signal"])
    root.body[-1].value.elts.append(ast.Name(id="middle_signal", ctx=ast.Load()))
    start = min([root.lineno] + [n.lineno for n in root.decorator_list]) - 1
    lines = source.splitlines(keepends=True)
    lines[start : root.end_lineno] = [ast.unparse(ast.fix_missing_locations(root)) + "\n"]
    source = "".join(lines).replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\nfrom .middle_pipeline import MIDDLE_SIGNAL_WORDS, score_attention_chain\n",
        1,
    )
    entry_path.write_text(source)
    binding_path = target / "binding.py"
    binding = binding_path.read_text()
    binding = binding.replace(
        "from .output_group import OUTPUT_SIGNAL_COUNT",
        "from .middle_phase import SIGNAL_COUNT as MIDDLE_SIGNAL_COUNT\nfrom .output_group import OUTPUT_SIGNAL_COUNT",
        1,
    )
    needle = '            "query_signal": ((1,), torch.int32),\n'
    assert binding.count(needle) == 1
    binding = binding.replace(
        needle,
        needle
        + (
            '            "middle_signal": ((MIDDLE_SIGNAL_COUNT if self.policy.fused_dag'
            + (" and self.policy.balanced" if balanced_only else "")
            + " else 1,), torch.int32),\n"
        ),
    )
    binding = binding.replace('if name in ("query_signal",', 'if name in ("middle_signal", "query_signal",', 1)
    binding_path.write_text(binding)
    for file in target.glob("middle*.py"):
        ast.parse(file.read_text())
    ast.parse(entry_path.read_text())
    ast.parse(binding_path.read_text())
    print(target)


if __name__ == "__main__":
    main()
