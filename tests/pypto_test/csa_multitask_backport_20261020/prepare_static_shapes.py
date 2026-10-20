# SPDX-License-Identifier: Apache-2.0
"""恢复新增constexpr分支后丢失的Attention固定scratch尺寸；不改变容量或数学。"""

import argparse
import ast
import json
import shutil

from prepare import HERE


def migrate(source):
    old = (
        "    transfer_slots = NUM_QK_CORES * QK_TRANSFER_SLOTS\n"
        "    transfer_heads = transfer_slots * H\n"
        "    transfer_kv_rows = transfer_slots * ATTN_K_TILE\n"
    )
    if source.count(old) != 1:
        raise AssertionError("需要原Attention固定尺寸定义")
    source = source.replace(old, "", 1)
    index = source.index("@pl.jit.inline(auto_scope=False)\ndef sparse_attn_csa(")
    constants = (
        "# 尺寸仅由硬件核数/固定tile决定；放在Python常量层，避免跨策略推导成动态维。\n"
        "ATTN_TRANSFER_HEADS = NUM_QK_CORES * QK_TRANSFER_SLOTS * H\n"
        "ATTN_TRANSFER_KV_ROWS = NUM_QK_CORES * QK_TRANSFER_SLOTS * ATTN_K_TILE\n\n\n"
    )
    source = source[:index] + constants + source[index:]
    source = source.replace("transfer_heads", "ATTN_TRANSFER_HEADS")
    source = source.replace("transfer_kv_rows", "ATTN_TRANSFER_KV_ROWS")
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="score_stats_ob")
    parser.add_argument("--candidate", default="score_stats_ob_static")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    source = target / "decode_sparse_attn_csa.py"
    source.write_text(migrate(source.read_text()))
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                changed_files=[source.name],
                reason="保持已验证多task原固定scratch维度，不改变字节数/访问范围/调度",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
