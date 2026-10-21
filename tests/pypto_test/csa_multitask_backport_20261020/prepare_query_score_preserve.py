# SPDX-License-Identifier: Apache-2.0
"""Q/Score融合分支保留；未融合形状恢复原QKV流程及其内部scope。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE
from prepare_query_ready import replace_once


def find_assignment(source, callee):
    matches = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == callee
    ]
    if len(matches) != 1:
        raise ValueError(f"需要唯一调用：{callee}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="query_score_v1")
    parser.add_argument("--candidate", default="query_score_preserve_v2")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    source = (base / "decode_csa.py").read_text()
    source = replace_once(
        source,
        "from .qkv_proj_rope import QR_M_TILE, PREFILL_DENSE_TILE, qkv_prepare_query_score\n",
        "from .qkv_proj_rope import (\n"
        "    QR_M_TILE, QPROJ_T_PAD, PREFILL_DENSE_TILE, qkv_prepare_query_score, qkv_proj_rope_policy,\n"
        ")\n",
    )
    normal = (HERE / "sources/formal_4d5b2ed9/decode_csa.py").read_text()
    old_node = find_assignment(normal, "qkv_proj_rope_policy")
    old_call = textwrap.dedent("".join(normal.splitlines(keepends=True)[old_node.lineno - 1 : old_node.end_lineno]))
    fused_node = find_assignment(source, "qkv_prepare_query_score")
    lines = source.splitlines(keepends=True)
    fused_call = textwrap.dedent("".join(lines[fused_node.lineno - 1 : fused_node.end_lineno]))
    body = (
        "# 未融合分支不返回原大QB暂存；这些有效小描述符只占位，不被任何核读取。\n"
        "# projected只有一行，不能作为实际Q_B输入；真实融合分支会全部替换它们。\n"
        "q_placeholder_rows = pl.min(t_dim, 1)\n"
        "fused_q_scale = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32)\n"
        "fused_q_projected = pl.create_tensor([q_placeholder_rows, H * HEAD_DIM], dtype=pl.INT32)\n"
        "fused_q_sin = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)\n"
        "fused_q_swap = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.INT32)\n"
        "if defer_q_to_score == 1:\n"
        + textwrap.indent(fused_call, "    ")
        + "else:\n"
        + textwrap.indent(old_call, "    ")
    )
    lines[fused_node.lineno - 1 : fused_node.end_lineno] = [textwrap.indent(body, " " * fused_node.col_offset)]
    updated = "".join(lines)
    ast.parse(updated)
    shutil.copytree(base, target)
    (target / "decode_csa.py").write_text(updated)
    manifest = json.loads((base / "candidate.json").read_text())
    manifest.update(
        candidate=args.candidate,
        base=args.base,
        delta="未融合分支恢复4d5b2ed9原QKV调用和内部scope；融合分支不改变数学/依赖/分工",
        placeholder_contract="小的合法Tensor占位，未融合Score不得捕获或读取；融合分支必须替换为真实QB/scale/RoPE缓冲",
        placeholder_bytes="INT32一行32768列128KiB，加512行scale和两份T×64的RoPE描述符；不伪造容量或别名权重",
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
