# SPDX-License-Identifier: Apache-2.0
"""将25de3f32前置只读几何的思路适配为多task SWA偏置/页检查前移。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE, section
from prepare_fused_topk import add_parameter
from prepare_scale_early import add_calls

PARAMETERS = """    prepared_swa_bias: pl.Tensor[[T_DYN, PADDED_TOPK], pl.FP32],
    prepared_swa_mask: pl.Tensor[[T_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
    swa_prepared: pl.Scalar[pl.INT32],
    swa_prepare_tid: pl.Scalar[pl.TASK_ID],
"""
ARGUMENTS = ("prepared_swa_bias", "prepared_swa_mask", "swa_prepared", "swa_prepare_tid")


def migrate_attention(source):
    swa = section(
        source,
        "            # ---- 滑窗有效位",
        "\n    # Native cosine rows already",
    )
    helper = '''
# 25de3f32：至少两个完整M32行块，或小M16资源域。
SWA_EARLY_MIN_ROWS = 64
SWA_EARLY_SMALL_LIMIT = 32


@pl.jit.inline(auto_scope=False)
def prepare_swa_geometry(
    position_ids: pl.Tensor[[T_DYN, 1], pl.INT64],
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    ori_block_table: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32],
    sparse_bias: pl.Tensor[[T_DYN, PADDED_TOPK], pl.FP32],
    valid_block_mask: pl.Tensor[[T_DYN, VALID_BLOCK_MASK_COLS], pl.INT32],
):
    """只读几何前移；不读Top-K或cache内容，数学复用原SWA plan。"""
    t_dim = pl.tensor.dim(position_ids, 0)
    with pl.spmd(CSA_PLAN_WORKERS, name_hint="csa_swa_geometry_early",
                 allow_early_resolve=True) as prepared_tid:
        plan_worker = pl.tile.get_block_idx()
        for bias_t0 in pl.range(plan_worker * BIAS_T_TILE, t_dim, CSA_PLAN_WORKERS * BIAS_T_TILE):
            bias_rows = pl.min(BIAS_T_TILE, t_dim - bias_t0)
'''
    helper += swa + "\n    return sparse_bias, valid_block_mask, prepared_tid\n\n\n"
    marker = "@pl.jit.inline(auto_scope=False)\ndef sparse_attn_csa("
    if source.count(marker) != 1:
        raise AssertionError("需要唯一Attention入口")
    source = source.replace(marker, helper + marker, 1)
    start = source.index("def sparse_attn_csa(")
    head, body = source[:start], source[start:]
    body = body.replace(swa, "            if swa_prepared == 0:\n" + textwrap.indent(swa, "    "), 1)
    body = body.replace(
        "    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)",
        "    sparse_bias = prepared_swa_bias",
        1,
    )
    body = body.replace(
        "    valid_block_mask = pl.create_tensor([t_dim, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)",
        "    valid_block_mask = prepared_swa_mask",
        1,
    )
    old = '    with pl.spmd(CSA_PLAN_WORKERS, name_hint="csa_slots_build_valid_qk_plan", allow_early_resolve=True)'
    new = (
        "    # 多层inline的TaskId必须有值引用，不能仅出现在deps属性。\n"
        "    swa_plan_dependencies = pl.array.create(1, pl.TASK_ID)\n"
        "    swa_plan_dependencies[0] = swa_prepare_tid\n"
        '    with pl.spmd(CSA_PLAN_WORKERS, name_hint="csa_slots_build_valid_qk_plan", '
        "deps=[swa_plan_dependencies[0]], allow_early_resolve=True)"
    )
    if body.count(old) != 1:
        raise AssertionError("需要唯一晚期plan分派")
    body = body.replace(old, new, 1)
    source = head + body
    for name in ("sparse_attn_csa", "sparse_attn_csa_tp1"):
        source = add_parameter(source, name, PARAMETERS)
    source = add_calls(source, ("sparse_attn_csa",), ARGUMENTS)
    ast.parse(source)
    return source


def migrate_entry(source):
    imports = [
        n for n in ast.parse(source).body if isinstance(n, ast.ImportFrom) and n.module == "decode_sparse_attn_csa"
    ]
    if len(imports) != 1:
        raise AssertionError("需要唯一Attention import")
    lines = source.splitlines(keepends=True)
    lines.insert(
        imports[0].end_lineno,
        "from .decode_sparse_attn_csa import (\n"
        "    prepare_swa_geometry, PADDED_TOPK, VALID_BLOCK_MASK_COLS,\n"
        "    SWA_EARLY_MIN_ROWS, SWA_EARLY_SMALL_LIMIT,\n"
        ")\n",
    )
    source = "".join(lines)
    old_view = "        position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])\n"
    if source.count(old_view) != 1:
        raise AssertionError("需要唯一Native位置view")
    source = source.replace(old_view, "", 1)
    marker = "        # 27711336前段证明的多task适配"
    start = source.index(marker)
    prepare = """        # 25de3f32：小M16或T>=64的只读Attention几何与投影并行。
        # SWA先写前128列及mask[0]；晚期plan依赖此TaskId再写压缩区，
        # 避免两个核同时写同一mask行的DCache，QK仍等完整plan。
        position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])
        prepared_swa_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)
        prepared_swa_mask = pl.create_tensor([t_dim, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
        swa_prepared = pl.cast(0, pl.INT32)
        if t_dim >= SWA_EARLY_MIN_ROWS:
            swa_prepared = pl.cast(1, pl.INT32)
        elif max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:
            if t_dim >= LONG_S6_MIN_QUERY_ROWS:
                if t_dim < SWA_EARLY_SMALL_LIMIT:
                    swa_prepared = pl.cast(1, pl.INT32)
        swa_prepare_deps = pl.array.create(1, pl.TASK_ID)
        swa_prepare_deps[0] = _widen_tid
        if swa_prepared != 0:
            prepared_swa_bias, prepared_swa_mask, swa_prepare_tid = prepare_swa_geometry(
                position_ids_t1, kv_seq_lens, ori_block_table, prepared_swa_bias, prepared_swa_mask,
            )
            swa_prepare_deps[0] = swa_prepare_tid

"""
    source = source[:start] + prepare + source[start:]
    source = add_calls(
        source,
        ("sparse_attn_csa_tp1",),
        ("prepared_swa_bias", "prepared_swa_mask", "swa_prepared", "swa_prepare_deps[0]"),
    )
    ast.parse(source)
    return source


def reuse_scale_proof(attention, entry):
    """沿已有长G6的单核metadata生命周期，不新增SWA分派。"""
    start = attention.index("def prepare_swa_geometry(")
    end = attention.index("@pl.jit.inline(auto_scope=False)\ndef sparse_attn_csa(", start)
    helper = attention[start:end]
    header = helper[: helper.index("    with pl.spmd(")]
    body_start = helper.index("            bias_rows =")
    body_end = helper.index("\n    return sparse_bias, valid_block_mask, prepared_tid")
    body = textwrap.indent(textwrap.dedent(helper[body_start:body_end]), " " * 8)
    helper = (
        header.replace("def prepare_swa_geometry(", "def prepare_swa_geometry_stage(", 1)
        + "    for bias_t0 in pl.range(0, t_dim, BIAS_T_TILE):\n"
        + body
        + "\n    return sparse_bias, valid_block_mask\n\n\n"
    )
    attention = attention[:start] + helper + attention[end:]
    entry = entry.replace("prepare_swa_geometry", "prepare_swa_geometry_stage")
    start = entry.index("        swa_prepared = pl.cast(0, pl.INT32)\n")
    end = entry.index("        swa_prepare_deps = pl.array.create", start)
    selection = (
        "        swa_prepared = pl.cast(0, pl.INT32)\n"
        "        if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:\n"
        "            if t_dim >= LONG_S6_MIN_QUERY_ROWS:\n"
        "                if t_dim >= SWA_EARLY_MIN_ROWS:\n"
        "                    swa_prepared = pl.cast(1, pl.INT32)\n"
        "                elif t_dim < SWA_EARLY_SMALL_LIMIT:\n"
        "                    swa_prepared = pl.cast(1, pl.INT32)\n"
    )
    entry = entry[:start] + selection + entry[end:]
    start = entry.index("        if swa_prepared != 0:\n")
    end = entry.index("        # 27711336", start)
    entry = entry[:start] + entry[end:]
    tree = ast.parse(entry)
    assignments = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "prove_scale_layout"
    ]
    if len(assignments) != 1:
        raise AssertionError("需要唯一早期页证明")
    call = assignments[0]
    indent = " " * call.col_offset
    lines = entry.splitlines(keepends=True)
    lines.insert(
        call.end_lineno,
        indent
        + "if swa_prepared != 0:\n"
        + indent
        + "    prepared_swa_bias, prepared_swa_mask = prepare_swa_geometry_stage(\n"
        + indent
        + "        position_ids_t1, kv_seq_lens, ori_block_table, prepared_swa_bias, prepared_swa_mask,\n"
        + indent
        + "    )\n",
    )
    entry = "".join(lines).replace(
        "                early_scale_deps[0] = early_scale_tid\n",
        "                early_scale_deps[0] = early_scale_tid\n"
        "                swa_prepare_deps[0] = early_scale_tid\n",
        1,
    )
    ast.parse(attention)
    ast.parse(entry)
    return attention, entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="qa_m16_v3")
    parser.add_argument("--candidate", default="early_swa")
    parser.add_argument("--reuse-proof", action="store_true", help="长G6复用既有页证明task，不单独派发SWA")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    for name, fn in (("decode_sparse_attn_csa.py", migrate_attention), ("decode_csa.py", migrate_entry)):
        path = target / name
        path.write_text(fn(path.read_text()))
    if args.reuse_proof:
        attention, entry = reuse_scale_proof(
            (target / "decode_sparse_attn_csa.py").read_text(), (target / "decode_csa.py").read_text()
        )
        (target / "decode_sparse_attn_csa.py").write_text(attention)
        (target / "decode_csa.py").write_text(entry)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="25de3f32",
                reuse_existing_scale_proof=args.reuse_proof,
                scope="前置只读SWA偏置/页检查，原数学保持；不是照搬SPMD动态owner扫描",
                domain="T>=64或长G6的24<=T<32；其余路径仍在原plan中计算SWA",
                dependency="晚期压缩plan显式等待SWA生产者；不并发写共享mask缓存行",
                changed_files=["decode_sparse_attn_csa.py", "decode_csa.py"],
                pending_device_validation=True,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
