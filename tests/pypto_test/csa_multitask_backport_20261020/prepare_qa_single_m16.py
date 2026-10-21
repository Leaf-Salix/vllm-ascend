# SPDX-License-Identifier: Apache-2.0
"""将来源66882634的独立输出块分工适配到长历史单M16区间。"""

import argparse
import ast
import json
import shutil

from prepare import HERE
from prepare_query_ready import replace_once
from prepare_query_score import append_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="formal_4d5b2ed9")
    parser.add_argument("--candidate", default="qa_single_m16_v2")
    parser.add_argument("--omit-legacy-constexpr", action="store_true", help="复现v1漏传既有调用constexpr的CPU错误")
    parser.add_argument("--sync-start", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not all(name.isidentifier() for name in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / name for name in (args.base, args.candidate))
    source = (base / "qkv_proj_rope.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q_proj_qa_m16_parallel")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[node.body[0].lineno - 1 : node.end_lineno])
    body = body.replace("QR_N_BLOCKS", "(Q_LORA // N_TILE)").replace("QR_N_TILE", "N_TILE")
    body = replace_once(
        body, "allow_early_resolve=True) as qa_tid:", "allow_early_resolve=True, sync_start=SYNC_START) as qa_tid:"
    )
    body = body.replace("每个M16/N128独立输出分核", "每个M16/N_TILE独立输出分核")
    lines[node.body[0].lineno - 1 : node.end_lineno] = [body]
    source = "".join(lines)
    source = append_parameters(
        source,
        "q_proj_qa_m16_parallel",
        ["N_TILE: pl.constexpr = QR_N_TILE", "SYNC_START: pl.constexpr = False"],
    )
    source = replace_once(
        source,
        "QA_M16_WORKER_LIMIT = QR_N_BLOCKS * QR_FIXED_M_GROUPS\n",
        "QA_M16_WORKER_LIMIT = QR_N_BLOCKS * QR_FIXED_M_GROUPS\nQA_SINGLE_M16_N_TILE = QR_N_TILE // 2\n",
    )
    source = replace_once(
        source,
        "    elif qa_m16_parallel != 0:\n",
        "    elif qa_m16_parallel == 2:\n"
        "        qa_tid = q_proj_qa_m16_parallel(\n"
        "            x, wq_a, qr_fp32, tile_base, tile_rows,\n"
        f"            N_TILE=QA_SINGLE_M16_N_TILE, SYNC_START={args.sync_start},\n"
        "        )\n"
        "    elif qa_m16_parallel != 0:\n",
    )
    if not args.omit_legacy_constexpr:
        source = replace_once(
            source,
            "        qa_tid = q_proj_qa_m16_parallel(x, wq_a, qr_fp32, tile_base, tile_rows)\n",
            "        qa_tid = q_proj_qa_m16_parallel(\n"
            "            x, wq_a, qr_fp32, tile_base, tile_rows, N_TILE=QR_N_TILE, SYNC_START=False,\n"
            "        )\n",
        )
    root = (base / "decode_csa.py").read_text()
    root = replace_once(
        root,
        "            # 66882634：长G6且16<T<32；按实际T选择，不枚举batch点。\n",
        "            # 来源66882634的独立输出块分工；长历史单M16用N64，原两M16区间保留N128。\n",
    )
    marker = (
        "            if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:\n"
        "                if t_dim >= LONG_S6_MIN_QUERY_ROWS:\n"
        "                    if t_dim < 2 * QR_M_TILE:\n"
        "                        qa_m16_parallel = pl.cast(1, pl.INT32)\n"
    )
    root = replace_once(
        root,
        marker,
        "            if max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:\n"
        "                if t_dim <= QR_M_TILE:\n"
        "                    qa_m16_parallel = pl.cast(2, pl.INT32)\n"
        "                elif t_dim >= LONG_S6_MIN_QUERY_ROWS:\n"
        "                    if t_dim < 2 * QR_M_TILE:\n"
        "                        qa_m16_parallel = pl.cast(1, pl.INT32)\n",
    )
    for code in (source, root):
        ast.parse(code)
    shutil.copytree(base, target)
    (target / "qkv_proj_rope.py").write_text(source)
    (target / "decode_csa.py").write_text(root)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="66882634 q_proj_qa_m16_parallel；已提交并验证的独立输出块分核",
                adaptation="原单M16只有8个N128块，调整为16个N64块；不拆K、不增加atomic、不改数学次序",
                resource_rule="压缩历史>8192且0<T<=16，按一个M16区间选择；原两M16及大T保持原路径",
                sync_start=args.sync_start,
                omit_legacy_constexpr=args.omit_legacy_constexpr,
                evidence=(
                    "B1_QA_OCCUPANCY_REVIEW.json：原QA8块，五窗平均活跃AIC约10.65..12.74；含核内等待，不是Cube busy"
                ),
                preserved="所有有效行与图重放动态长度；每列仍完整K256顺序、wq_a NZ/ND及缓存属性不变",
                legacy_interface="仅根CSA设置策略值2；原默认0及策略1继续使用原N128/同步设置",
                caveat="N64是本次允许的分块适配，不冒称来源已测过N64或单M16；须核内和完整CSA验证",
                changed_files=["qkv_proj_rope.py", "decode_csa.py"],
                status="private_candidate_no_device_evidence",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
