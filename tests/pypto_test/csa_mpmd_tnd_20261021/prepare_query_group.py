# SPDX-License-Identifier: Apache-2.0
"""冻结局部Indexer query MIX候选；其余CSA继续独立多task调度。"""

import ast
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from prepare_priority_fusion import scope_name


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    source = (target / "entry.py").read_text()
    root = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "decode_csa_tnd"
    )
    annotations = {arg.arg: ast.unparse(arg.annotation) for arg in root.args.args}
    annotations["query_signal"] = "pl.InOut[pl.Tensor[[QUERY_SIGNAL_WORDS], pl.INT32]]"

    cube_args = ["qr", "idx_wq_b", "idx_query_acc", "idx_query_rope_heads", "hadamard", "idx_query_had", "query_signal"]
    vector_args = [
        "qr_scale",
        "idx_wq_b_scale",
        "rope_cos",
        "rope_signed",
        "idx_query_acc",
        "idx_query_rope",
        "idx_query_had",
        "idx_query_i8",
        "idx_query_scale",
        "idx_query_scale_heads",
        "head_weights",
        "positions",
        "idx_bounds",
        "idx_lengths",
        "group_offsets",
        "coefficients",
        "query_signal",
    ]
    cube_out = {"idx_query_acc", "idx_query_had", "query_signal"}
    vector_out = {"idx_query_rope", "idx_query_i8", "idx_query_scale", "coefficients", "query_signal"}
    group_args = list(dict.fromkeys(cube_args + vector_args))
    outputs = [
        "idx_query_acc",
        "idx_query_rope",
        "idx_query_had",
        "idx_query_i8",
        "idx_query_scale",
        "coefficients",
        "query_signal",
    ]
    # The JIT call boundary returns InOut tensors in parameter order.
    group_args = [param for param in group_args if param not in outputs] + outputs

    def function(func_name, decorator, names, written, body, constexpr=False):
        params = []
        for param in names:
            annotation = annotations[param]
            if param not in written and annotation.startswith("pl.InOut["):
                annotation = annotation[len("pl.InOut[") : -1]
            params.append(f"    {param}: {annotation},")
        if constexpr:
            params.append("    QUERY_GROUP: pl.constexpr,")
        return (
            f"@{decorator}\ndef {func_name}(\n"
            + "\n".join(params)
            + "\n):\n"
            + textwrap.indent(textwrap.dedent(body).strip(), "    ")
            + "\n\n"
        )

    cube = function(
        "query_chain_cube",
        "pl.jit.incore",
        cube_args,
        cube_out,
        """
        worker = pl.tile.get_block_idx()
        tokens = pl.tensor.dim(qr, 0)
        query_signal, generation = begin_invocation(query_signal, worker)
        idx_query_acc = indexer_query_project_stage(qr, idx_wq_b, idx_query_acc, worker, GROUPS)
        query_signal = publish_phase(query_signal, PROJECT, worker, CORES, generation)
        rope_units = ((tokens + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE) * (IDX_N_HEADS // DQ_ROPE_H_TILE)
        query_signal = wait_row_owners(query_signal, ROPE, CORES, generation, 0, rope_units, 1, VECTORS, GROUPS, 1)
        idx_query_had = indexer_query_hadamard_stage(
            idx_query_rope_heads, hadamard, idx_query_had, tokens, worker, GROUPS
        )
        query_signal = publish_phase(query_signal, HADAMARD, worker, CORES, generation)
        return idx_query_acc, idx_query_had, query_signal
    """,
    )
    vector = function(
        "query_chain_vector",
        "pl.jit.incore",
        vector_args,
        vector_out,
        """
        pl.func_attr({"split_aiv": True})
        worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
        core = GROUPS + worker
        tokens = pl.tensor.dim(qr_scale, 0)
        query_signal, generation = begin_invocation(query_signal, core)
        query_signal = wait_row_owners(
            query_signal, PROJECT, CORES, generation,
            0, IDX_N_HEADS * IDX_HEAD_DIM // QR_MM_N_TILE, 1, GROUPS, 0, 1
        )
        idx_query_rope = indexer_query_dequant_rope_stage(
            qr_scale, idx_wq_b_scale, rope_cos, rope_signed,
            idx_query_acc, idx_query_rope, worker, VECTORS
        )
        query_signal = publish_phase(query_signal, ROPE, core, CORES, generation)
        query_signal = wait_row_owners(query_signal, HADAMARD, CORES, generation, 0, tokens, 1, GROUPS, 0, 1)
        idx_query_i8, idx_query_scale = indexer_query_quant_stage(
            idx_query_had, idx_query_i8, idx_query_scale, tokens, worker, VECTORS
        )
        query_signal = publish_phase(query_signal, QUANT, core, CORES, generation)
        query_signal = wait_row_owners(query_signal, QUANT, CORES, generation, 0, tokens, 1, VECTORS, GROUPS, 1)
        coefficients = indexer_head_coefficients_stage(
            idx_query_scale_heads, head_weights, positions, idx_bounds, idx_lengths,
            group_offsets, coefficients, worker, VECTORS, QUERY_GROUP
        )
        return idx_query_rope, idx_query_i8, idx_query_scale, coefficients, query_signal
    """,
        constexpr=True,
    )
    group = function(
        "indexer_query_chain",
        "pl.jit.inline(auto_scope=False)",
        group_args,
        set(outputs),
        f"""
        with pl.cluster(name_hint="indexer_query_chain"):
            idx_query_acc, idx_query_had, query_signal = query_chain_cube({", ".join(cube_args)})
            idx_query_rope, idx_query_i8, idx_query_scale, coefficients, query_signal = query_chain_vector(
                {", ".join(vector_args)}, QUERY_GROUP
            )
        return {", ".join(outputs)}
    """,
        constexpr=True,
    )
    header = '''# SPDX-License-Identifier: Apache-2.0
"""Indexer query局部MIX链。只在本task内使用冻结参考的完成信号。

所有24个AIC/48个AIV每次共同推进代际，即使本轮无数据也不跳过。
各阶段完成本核GM写回后发布；仅等待实际生产者。GM中间存储、规约、
量化和Top-K规则不变；外部依赖和生命周期仍由TMR处理。
query_signal由准备对象首次清零并顺序独占，不能并发复用或跨布局重用。
"""

import pypto.language as pl
from .query_phase_sync import begin_invocation, publish_phase, wait_row_owners
from .indexer_group_stages import indexer_head_coefficients_stage
from .indexer_query_stages import (
    DEQUANT_T_TILE, DQ_ROPE_H_TILE, IDX_N_HEADS, IDX_HEAD_DIM, QR_MM_N_TILE,
    indexer_query_project_stage, indexer_query_dequant_rope_stage,
    indexer_query_hadamard_stage, indexer_query_quant_stage,
)
from .runtime_abi import (
    T, B, IQ, IDX_PROJ_ROWS, IDX_ROPE_ROWS, IDX_ROPE_HEAD_ROWS,
    IDX_HEAD_ROWS, IDX_SCALE_T_ROWS, WEIGHTS_ROWS, GROUP_OFFSET_ROWS,
    COEFF_ROWS, COEFF_COLS, CORES, GROUPS, VECTORS,
)

QUERY_SIGNAL_WORDS = pl.dynamic("CSA_QUERY_SIGNAL_WORDS")
QUERY_SIGNAL_COUNT = 5 * CORES * 128
PROJECT, ROPE, HADAMARD, QUANT = 1, 2, 3, 4

'''
    (target / "query_group.py").write_text(header + cube + vector + group)
    frozen = here / "sources/84751e34/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    for old, new in (("signals.py", "query_signals.py"), ("phase_sync.py", "query_phase_sync.py")):
        text = (frozen / old).read_text().replace("from .signals import", "from .query_signals import")
        (target / new).write_text(text)

    scopes = {scope_name(node): node for node in ast.walk(root) if scope_name(node)}
    removed = {"indexer_query_proj", "indexer_query_rope", "indexer_query_hadamard", "indexer_query_quant"}
    edits = []
    for item in removed:
        node = scopes[item]
        edits.append((node.lineno - 1, node.end_lineno, ""))
    node = scopes["indexer_head_coefficients"]
    body = (
        "with pl.spmd(\n"
        '    GROUPS, name_hint="indexer_query_chain", allow_early_resolve=True,\n'
        "    sync_start=True, deps=[metadata_tid]\n"
        ") as index_query_tid:\n"
        f"    {', '.join(outputs)} = indexer_query_chain(\n"
        f"        {', '.join(group_args)}, QUERY_GROUP\n"
        "    )"
    )
    edits.append((node.lineno - 1, node.end_lineno, textwrap.indent(body, "    ") + "\n"))
    first_constexpr = next(arg for arg in root.args.args if arg.arg == "QUERY_GROUP")
    edits.append(
        (
            first_constexpr.lineno - 1,
            first_constexpr.lineno - 1,
            "    query_signal: pl.InOut[pl.Tensor[[QUERY_SIGNAL_WORDS], pl.INT32]],\n",
        )
    )
    ret = root.body[-1]
    assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple)
    edits.append((ret.end_lineno - 1, ret.end_lineno - 1, "        query_signal,\n"))
    # The merged task keeps all 24/48 lanes, so these former launch-only counts are dead.
    for node in ast.walk(root):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {"query_rope_workers", "query_hadamard_workers"} for t in node.targets
        ):
            edits.append((node.lineno - 1, node.end_lineno, ""))
    lines = source.splitlines(keepends=True)
    for begin, end, text in sorted(edits, reverse=True):
        lines[begin:end] = [text]
    source = "".join(lines).replace("deps=[index_rope_tid]", "deps=[index_query_tid]")
    source = source.replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .query_group import QUERY_SIGNAL_WORDS, indexer_query_chain\n",
        1,
    )
    (target / "entry.py").write_text(source)

    path = target / "binding.py"
    text = path.read_text().replace(
        "from .policy import CSAMPMDPolicy, select_policy",
        "from .policy import CSAMPMDPolicy, select_policy\nfrom .query_group import QUERY_SIGNAL_COUNT",
    )
    text = text.replace(
        '            "ffts": ((256,), torch.int64),',
        '            "ffts": ((256,), torch.int64),\n            "query_signal": ((QUERY_SIGNAL_COUNT,), torch.int32),',
    )
    text = text.replace(
        "            self.buffers[name] = self.allocations[name] = value",
        "            self.buffers[name] = self.allocations[name] = value\n"
        '            if name == "query_signal":\n'
        "                value.zero_()",
    )
    path.write_text(text)
    assert sum(scope_name(node) is not None for node in ast.walk(ast.parse(source))) == 36
    print(target)


if __name__ == "__main__":
    main()
