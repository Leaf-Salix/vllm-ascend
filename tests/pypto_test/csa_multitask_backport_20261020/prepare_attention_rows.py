# SPDX-License-Identifier: Apache-2.0
"""迁移已验证单SPMD的八行Plan→Attention依赖，保留多task数学及FFTS流水。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE


def span(source, nodes):
    lines = source.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[nodes[0].lineno - 1 : nodes[-1].end_lineno]))


def function(name, arguments, body, *, incore=False):
    decorator = "@pl.jit.incore" if incore else "@pl.jit.inline(auto_scope=False)"
    return (
        f"{decorator}\ndef {name}(\n"
        + "".join(f"    {argument},\n" for argument in arguments)
        + "):\n"
        + textwrap.indent(body.strip() + "\n", "    ")
        + "\n\n"
    )


def tensor_args(names, writes=()):
    contracts = {
        "q_flat": "[ATTENTION_HEAD_ROWS_DYN, HEAD_DIM], pl.BF16",
        "valid_block_mask": "[T_DYN, VALID_BLOCK_MASK_COLS], pl.INT32",
        "kv_transfer": "[ATTN_TRANSFER_KV_ROWS, HEAD_DIM], pl.BF16",
        "score_transfer": "[ATTN_TRANSFER_HEADS, ATTN_K_TILE], pl.FP32",
        "probability_transfer": "[ATTN_TRANSFER_HEADS, ATTN_K_TILE], pl.BF16",
        "pv_transfer": "[ATTN_TRANSFER_HEADS, HEAD_DIM], pl.FP32",
        "ffts_workspace": "[256], pl.INT64",
        "plan_flags": "[PLAN_ROW_WORDS], pl.INT32",
        "ori_kv_flat": "[ATTENTION_ORI_ROWS_DYN, HEAD_DIM], pl.BF16",
        "ori_block_table": "[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32",
        "cmp_kv": "[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16",
        "cmp_kv_flat": "[ATTENTION_CMP_ROWS_DYN, HEAD_DIM], pl.BF16",
        "cmp_block_table": "[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32",
        "idx_topk": "[T_DYN, IDX_TOPK], pl.INT32",
        "position_ids": "[T_DYN, 1], pl.INT64",
        "seq_lens": "[B_DYN], pl.INT32",
        "attn_sink_col": "[H, 1], pl.FP32",
        "freqs_cos": "[T_DYN, ROPE_DIM], pl.FP32",
        "rope_sin_signed": "[T_PAD, ROPE_DIM], pl.FP32",
        "o_packed_heads": "[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16",
        "sparse_bias": "[T_DYN, PADDED_TOPK], pl.FP32",
        "cmp_sparse_indices": "[T_DYN, CMP_TOPK], pl.INT32",
        "mi_transfer": "[ATTN_TRANSFER_HEADS, 1], pl.FP32",
        "li_transfer": "[ATTN_TRANSFER_HEADS, 1], pl.FP32",
    }
    tensors = {name: f"pl.Tensor[{contracts[name]}]" for name in names}
    return [f"{name}: " + (f"pl.InOut[{tensors[name]}]" if name in writes else tensors[name]) for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="qa_m16_v3")
    parser.add_argument("--candidate", default="attention_rows_v1")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    base = HERE / "sources" / args.base
    target = HERE / "sources" / args.candidate
    shutil.copytree(base, target)
    path = target / "decode_sparse_attn_csa.py"
    source = path.read_text()
    tree = ast.parse(source)
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_csa")
    plan = next(
        n
        for n in entry.body
        if isinstance(n, ast.With) and "csa_slots_build_valid_qk_plan" in ast.unparse(n.items[0].context_expr)
    )
    qk = next(
        n
        for n in entry.body
        if isinstance(n, ast.With) and 'name_hint="qk_pv"' in ast.get_source_segment(source, n.items[0].context_expr)
    )
    vector_loop = qk.body[-1]
    if not isinstance(vector_loop, ast.For) or "pl.split_aiv" not in ast.unparse(vector_loop.iter):
        raise ValueError("来源的QK/PV双AIV结构改变，须人工核对")
    plan_tensors = [
        "ori_block_table",
        "cmp_kv",
        "cmp_block_table",
        "idx_topk",
        "position_ids",
        "seq_lens",
        "sparse_bias",
        "cmp_sparse_indices",
        "valid_block_mask",
    ]
    plan_outputs = ["sparse_bias", "cmp_sparse_indices", "valid_block_mask"]
    plan_math = span(source, plan.body[1:]).replace("CSA_PLAN_WORKERS", "PLAN_ROW_VECTORS")
    helpers = function(
        "attention_plan_rows",
        tensor_args(plan_tensors, plan_outputs) + ["plan_worker: pl.Scalar[pl.INDEX]", "PHYSICAL_ROWS: pl.constexpr"],
        "t_dim = pl.tensor.dim(idx_topk, 0)\n" + plan_math + "\nreturn " + ", ".join(plan_outputs),
    )
    helpers += function(
        "attention_wait_plan_rows",
        ["plan_flags: pl.InOut[pl.Tensor]", "tokens: pl.Scalar[pl.INDEX]", "worker: pl.Scalar[pl.INDEX]"],
        """if worker < tokens:
    for query in pl.range(worker, tokens, NUM_QK_CORES):
        owner = (query // BIAS_T_TILE) % PLAN_ROW_VECTORS
        plan_flags = csa_signal_wait(plan_flags, owner * PLAN_ROW_SLOT, pl.cast(1, pl.INT32))
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
return plan_flags""",
    )
    cube_tensors = [
        "q_flat",
        "valid_block_mask",
        "kv_transfer",
        "score_transfer",
        "probability_transfer",
        "pv_transfer",
        "ffts_workspace",
        "plan_flags",
    ]
    cube_outputs = ["score_transfer", "pv_transfer", "plan_flags"]
    cube_math = span(source, qk.body[:-1])
    helpers += function(
        "attention_rows_cube",
        tensor_args(cube_tensors, cube_outputs),
        "\n".join(f"{name}_gm = {name}" for name in cube_outputs)
        + "\nt_dim = pl.tensor.dim(valid_block_mask, 0)\n"
        + "plan_flags = attention_wait_plan_rows(plan_flags, t_dim, pl.tile.get_block_idx())\n"
        + cube_math
        + "\nreturn "
        + ", ".join(f"{name}_gm" for name in cube_outputs),
        incore=True,
    )
    vector_tensors = [
        "ori_kv_flat",
        "ori_block_table",
        "cmp_kv",
        "cmp_kv_flat",
        "cmp_block_table",
        "idx_topk",
        "position_ids",
        "seq_lens",
        "attn_sink_col",
        "freqs_cos",
        "rope_sin_signed",
        "o_packed_heads",
        "sparse_bias",
        "cmp_sparse_indices",
        "valid_block_mask",
        "kv_transfer",
        "score_transfer",
        "probability_transfer",
        "pv_transfer",
        "mi_transfer",
        "li_transfer",
        "ffts_workspace",
        "plan_flags",
    ]
    vector_outputs = [
        "o_packed_heads",
        "sparse_bias",
        "cmp_sparse_indices",
        "valid_block_mask",
        "kv_transfer",
        "probability_transfer",
        "mi_transfer",
        "li_transfer",
        "plan_flags",
    ]
    vector_prefix = (
        'pl.func_attr({"split_aiv": True})\n'
        + "\n".join(f"{name}_gm = {name}" for name in vector_outputs)
        + """
t_dim = pl.tensor.dim(idx_topk, 0)
qk_core = pl.tile.get_block_idx()
qk_aiv = pl.tile.get_subblock_idx()
plan_worker = qk_core * 2 + qk_aiv
"""
        + ", ".join(plan_outputs)
        + " = attention_plan_rows(\n    "
        + ", ".join(plan_tensors)
        + ", plan_worker, PHYSICAL_ROWS,\n)\n"
        + """# 每个AIV先完成全部本人八行tile，再发布一次；之后才允许等待其它owner。
plan_flags = publish_phase(plan_flags, 0, plan_worker, PLAN_ROW_VECTORS, pl.cast(1, pl.INT32))
plan_flags = attention_wait_plan_rows(plan_flags, t_dim, qk_core)
qk_query_count = (t_dim + NUM_QK_CORES - 1 - qk_core) // NUM_QK_CORES
qk_block_count = qk_query_count * SPARSE_BLOCKS
"""
    )
    helpers += function(
        "attention_rows_vector",
        tensor_args(vector_tensors, vector_outputs)
        + ["LOCAL_ATTENTION_STATS: pl.constexpr", "PHYSICAL_ROWS: pl.constexpr"],
        vector_prefix
        + span(source, vector_loop.body)
        + "\nreturn "
        + ", ".join(f"{name}_gm" for name in vector_outputs),
        incore=True,
    )
    cluster_tensors = list(dict.fromkeys(cube_tensors + vector_tensors))
    cluster_outputs = list(dict.fromkeys(cube_outputs + vector_outputs))
    cluster_body = (
        'with pl.cluster(name_hint="attention_plan_rows_pipeline"):\n    '
        + ", ".join(cube_outputs)
        + " = attention_rows_cube(\n        "
        + ", ".join(cube_tensors)
        + ",\n    )\n    "
        + ", ".join(vector_outputs)
        + " = attention_rows_vector(\n        "
        + ", ".join(vector_tensors)
        + ", LOCAL_ATTENTION_STATS, PHYSICAL_ROWS,\n    )\nreturn o_packed_heads\n"
    )
    helpers += function(
        "attention_rows_cluster",
        tensor_args(cluster_tensors, cluster_outputs)
        + ["LOCAL_ATTENTION_STATS: pl.constexpr", "PHYSICAL_ROWS: pl.constexpr"],
        cluster_body,
    )
    replacements = [
        (
            plan.lineno,
            plan.end_lineno,
            "    # 单调用信号由既有rope_cs任务清零；没有新增清零dispatch或持久代际。\n"
            "    plan_flags = pl.create_tensor([PLAN_ROW_WORDS], dtype=pl.INT32)\n"
            "    plan_flags_matrix = pl.reshape(plan_flags, [PLAN_ROW_VECTORS, PLAN_ROW_SLOT])\n",
        ),
        (
            qk.lineno,
            qk.end_lineno,
            "    # 保留原rope和Top-K等数据依赖；仅Plan→Attention改成核内真实行owner。\n"
            "    with pl.spmd(NUM_QK_CORES, deps=[rope_tid], allow_early_resolve=True) as qk_tid:\n"
            "        _packed_result = attention_rows_cluster(\n            "
            + ", ".join(cluster_tensors)
            + ", LOCAL_ATTENTION_STATS, PHYSICAL_ROWS,\n        )\n",
        ),
    ]
    lines = source.splitlines(keepends=True)
    for first, last, replacement in sorted(replacements, reverse=True):
        lines[first - 1 : last] = [replacement]
    source = "".join(lines)
    needle = "                 allow_early_resolve=True) as rope_tid:\n"
    if source.count(needle) != 1:
        raise ValueError("rope任务入口应唯一")
    source = source.replace(
        needle,
        needle
        + "        flag_zero = pl.tile.full([1, PLAN_ROW_SLOT], dtype=pl.INT32, value=0)\n"
        + "        for slot in pl.range(pl.tile.get_block_idx(), PLAN_ROW_VECTORS,\n"
        + "                             pl.min(rope_cs_blocks, ROPE_CS_WORKERS)):\n"
        + "            pl.store(flag_zero, [slot, 0], plan_flags_matrix)\n",
        1,
    )
    marker = "@pl.jit.inline(auto_scope=False)\ndef sparse_attn_csa("
    source = source.replace(
        marker,
        "PLAN_ROW_VECTORS = 48\nPLAN_ROW_SLOT = 128\nPLAN_ROW_WORDS = PLAN_ROW_VECTORS * PLAN_ROW_SLOT\n\n\n"
        'ATTENTION_HEAD_ROWS_DYN = pl.dynamic("ATTENTION_HEAD_ROWS_DYN")\n'
        'ATTENTION_ORI_ROWS_DYN = pl.dynamic("ATTENTION_ORI_ROWS_DYN")\n'
        'ATTENTION_CMP_ROWS_DYN = pl.dynamic("ATTENTION_CMP_ROWS_DYN")\n\n\n' + helpers + marker,
        1,
    )
    source = source.replace(
        "from .physical_attention_plan import (",
        "from .attention_row_phase_sync import publish_phase\n"
        "from .signals import csa_signal_wait\n\n"
        "from .physical_attention_plan import (",
        1,
    )
    # JIT的reshape元数据扫描识别维度表达式，不追踪二次Scalar赋值t_heads。
    # 保留原式值，只在view处展开乘法，真实尺寸仍从q读取。
    source = source.replace(
        "q_flat = pl.reshape(q, [t_heads, HEAD_DIM])",
        "q_flat = pl.reshape(q, [t_dim * H, HEAD_DIM])",
        1,
    )
    ast.parse(source)
    path.write_text(source)
    phase = (HERE / "sources/spmd/phase_sync.py").read_text()
    phase = phase.replace('SIGNAL_WORDS = pl.dynamic("CSA_PHASE_SIGNAL_WORDS")', "SIGNAL_WORDS = 48 * 128", 1)
    (target / "attention_row_phase_sync.py").write_text(phase)
    shutil.copy2(HERE / "sources/spmd/signals.py", target / "signals.py")
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="5da1bf0c，冻结13259f92的wait_attention_plan_rows",
                scope="原Plan与QK/PV局部MIX，其它阶段保持多task；Top-K仍单独完整任务",
                preserved="八行plan数学、物理页与fallback、QK/PV三槽FFTS、softmax及BF16边界",
                changes="Plan由16改48个AIV按八行tile轮转；每核组只等待query=group+k*24的plan写者",
                lifetime="24KiB单调用flags由原rope_cs清零；全体AIV先发布本人计划再跨核等待",
                wait_policy={"backoff": "cached100", "unit_family": "attention"},
                status="source_prepared_no_device_evidence",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
