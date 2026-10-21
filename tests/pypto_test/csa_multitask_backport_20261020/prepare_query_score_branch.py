# SPDX-License-Identifier: Apache-2.0
"""隔离Q/Score融合的前段：私有实验保留完整原Indexer，避免无效Tensor占位。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE
from prepare_query_ready import replace_once
from prepare_query_score_dispatch import frontend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="query_score_v1")
    parser.add_argument("--candidate", default="query_score_branch_v8")
    parser.add_argument("--keep-result-names", action="store_true", help="只用于复现v7的跨分支符号维度错误")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    original = HERE / "sources/formal_4d5b2ed9"
    entry = (base / "decode_csa.py").read_text()
    start, stop, fused = frontend(entry)
    _, _, fallback = frontend((original / "decode_csa.py").read_text())
    fused = replace_once(fused, "            q, qa_tid,", "            _fused_q, qa_tid,")
    fallback = replace_once(fallback, "            q, qa_tid =", "            _standalone_q, qa_tid =")
    if not args.keep_result_names:
        fused = replace_once(fused, "_index_scores, index_indices =", "_fused_scores, _fused_indices =")
        fallback = replace_once(fallback, "_index_scores, index_indices =", "_standalone_scores, _standalone_indices =")
    fallback = replace_once(fallback, "= indexer(\n", "= indexer_standalone(\n")
    replacement = (
        "            if defer_q_to_score == 1:\n"
        + textwrap.indent(fused, "    ")
        + "            else:\n"
        + textwrap.indent(fallback, "    ")
    )
    entry = entry[:start] + replacement + entry[stop:]
    entry = replace_once(
        entry,
        "from .qkv_proj_rope import QR_M_TILE, PREFILL_DENSE_TILE, qkv_prepare_query_score\n",
        "from .qkv_proj_rope import QR_M_TILE, PREFILL_DENSE_TILE, qkv_prepare_query_score, qkv_proj_rope_policy\n"
        "from .decode_indexer_standalone import indexer as indexer_standalone\n",
    )
    entry = replace_once(entry, "                index_indices,\n", "                idx_topk,\n")
    ast.parse(entry)
    shutil.copytree(base, target)
    (target / "decode_csa.py").write_text(entry)
    shutil.copyfile(original / "decode_indexer.py", target / "decode_indexer_standalone.py")
    manifest = json.loads((base / "candidate.json").read_text())
    manifest.update(
        candidate=args.candidate,
        base=args.base,
        delta="按当次设备max_seq_len和T分派完整QKV至Indexer前段；小批/短历史保留原QKV及Indexer",
        fallback_contract="复制正式4d5b2ed9的原Indexer只用于私有因果实验；无占位Tensor或额外设备分配",
        output_contract="Q和idx_topk为外层预分配输出，Attention使用原句柄；实际依赖与同图分支切换需验证",
        caveats=[
            "两份Indexer暂仅用于私有实验；若有收益，正式采用前必须消除重复数学实现",
            "动态分支和额外编译变体可能有编排成本；数学相同不等于性能相同",
        ],
        status="private_candidate_no_device_evidence",
        repro_only_keep_result_names=args.keep_result_names,
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
