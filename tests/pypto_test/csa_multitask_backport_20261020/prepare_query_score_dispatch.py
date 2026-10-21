# SPDX-License-Identifier: Apache-2.0
"""按当次有效长度分派完整前段，非融合路径保留原QKV及scope。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE
from prepare_query_ready import replace_once
from prepare_query_score import FIELDS, append_call_arguments, append_parameters


def frontend(source):
    start = source.index("            q, qa_tid")
    stop = source.index("            # sparse_attn_csa folds", start)
    return start, stop, source[start:stop]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="query_score_v1")
    parser.add_argument("--candidate", default="query_score_dispatch_v6")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    original = (HERE / "sources/formal_4d5b2ed9/decode_csa.py").read_text()
    entry = (base / "decode_csa.py").read_text()
    indexer = (base / "decode_indexer.py").read_text()
    for name in ("indexer_score_topk_forest", "indexer_weights_score", "indexer"):
        indexer = append_parameters(indexer, name, ["allow_query_fusion: pl.constexpr"])
    indexer = append_call_arguments(
        indexer, {"indexer_weights_score", "indexer_score_topk_forest"}, ["allow_query_fusion"]
    )
    calls = [
        n
        for n in ast.walk(ast.parse(indexer))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "indexer_score_topk_native_cube"
        and isinstance(n.args[-1], ast.Constant)
        and n.args[-1].value is True
    ]
    if len(calls) != 1:
        raise ValueError("只允许一个长G6拆M的融合调用")
    final_arg = calls[0].args[-1]
    lines = indexer.splitlines(keepends=True)
    line = lines[final_arg.lineno - 1]
    lines[final_arg.lineno - 1] = line[: final_arg.col_offset] + "allow_query_fusion" + line[final_arg.end_col_offset :]
    indexer = "".join(lines)

    start, stop, fused = frontend(entry)
    _, _, fallback = frontend(original)
    # 分支内的Q返回别名不用于跨分支传递；两条路径都原地写入外层Q。
    fused = replace_once(fused, "            q, qa_tid,", "            _fused_q, qa_tid,")
    fallback = replace_once(fallback, "            q, qa_tid =", "            _standalone_q, qa_tid =")
    fused = textwrap.indent(append_call_arguments(textwrap.dedent(fused), {"indexer"}, ["True"]), " " * 12)
    fallback = textwrap.indent(
        append_call_arguments(textwrap.dedent(fallback), {"indexer"}, ["qr"] * len(FIELDS) + ["False"]),
        " " * 12,
    )
    # False是编译期常量，七个Q字段全部不可达。传真实已存在的Tensor描述符，
    # 不分配占位空间，也不虚构容量；CPU/设备代码必须确认不捕获这些字段。
    fallback = (
        "            # allow_query_fusion=False：附加Q字段编译期不可达，qr仅作合法描述符。\n"
        "            # 本分支必须保留原QKV内部scope和独立Q尾段，不扩大QB生命周期。\n" + fallback
    )
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
        "from .qkv_proj_rope import QR_M_TILE, PREFILL_DENSE_TILE, qkv_prepare_query_score, qkv_proj_rope_policy\n",
    )
    entry = replace_once(entry, "                index_indices,\n", "                idx_topk,\n")
    ast.parse(entry)
    ast.parse(indexer)
    shutil.copytree(base, target)
    (target / "decode_csa.py").write_text(entry)
    (target / "decode_indexer.py").write_text(indexer)
    manifest = json.loads((base / "candidate.json").read_text())
    manifest.update(
        candidate=args.candidate,
        base=args.base,
        delta="按当次设备长度和T分派整个前段；未融合恢复原QKV内部scope；融合仍为v1数学和生命周期",
        fallback_contract="allow_query_fusion=False在编译期删除Q字段读取；七个额外参数使用真实qr描述符，不分配占位",
        output_contract="Q和idx_topk均为外层预分配输出，后续Attention直接使用原句柄，实际依赖另查",
        caveat="新增动态分支和编译特化可能改变编排；不可仅因数学相同假定非融合路径性能不变",
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
