# SPDX-License-Identifier: Apache-2.0
"""以实际叶计划选择Q融合通信槽数，不枚举batch/seqlen测试点。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE
from prepare_query_ready import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="query_score_fifo4_v10")
    parser.add_argument("--candidate", default="query_score_fifo_policy_v11")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    source = (base / "decode_indexer.py").read_text()
    source = replace_once(
        source,
        "SCORE_CONSUMED_EVENT = 1\n",
        "SCORE_CONSUMED_EVENT = 1\n"
        "# 两长档完整叶受益；均衡5/6步叶保持双槽。阈值是两个四槽窗口。\n"
        "Q_SCORE_BASE_FIFO_SLOTS = 2\n"
        "Q_SCORE_EXTENDED_FIFO_SLOTS = 4\n"
        "Q_SCORE_EXTENDED_MIN_LEAF_STEPS = 2 * Q_SCORE_EXTENDED_FIFO_SLOTS\n",
    )
    calls = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "indexer_score_topk_native_cube"
        and ast.unparse(n.value.args[-1]) == "2 + 2 * allow_query_fusion"
    ]
    if len(calls) != 1 or len(calls[0].value.args) != 32:
        raise ValueError("只修改长G6拆M的一处槽数选择")
    n = calls[0]
    lines = source.splitlines(keepends=True)
    original = textwrap.dedent("".join(lines[n.lineno - 1 : n.end_lineno]))
    extended = replace_once(
        original,
        "2 + 2 * allow_query_fusion,",
        "Q_SCORE_BASE_FIFO_SLOTS + (Q_SCORE_EXTENDED_FIFO_SLOTS - Q_SCORE_BASE_FIFO_SLOTS) * allow_query_fusion,",
    )
    normal = replace_once(original, "2 + 2 * allow_query_fusion,", "Q_SCORE_BASE_FIFO_SLOTS,")
    body = (
        """
# 与Score内部使用相同的当次长度/T及同一叶计划，最后的实际短尾仍按min排空。
_fifo_leaves, fifo_leaf_steps, _fifo_extra = indexer_long_leaf_plan(
    pl.min(max_topk_cache_len, TOPK_MAX_CANDIDATES), pl.tensor.dim(position_ids, 0) // S,
)
if fifo_leaf_steps >= Q_SCORE_EXTENDED_MIN_LEAF_STEPS:
""".lstrip("\n")
        + textwrap.indent(extended, "    ")
        + "else:\n"
        + textwrap.indent(normal, "    ")
    )
    lines[n.lineno - 1 : n.end_lineno] = [textwrap.indent(body, " " * n.col_offset)]
    source = "".join(lines)
    ast.parse(source)
    shutil.copytree(base, target)
    (target / "decode_indexer.py").write_text(source)
    manifest = json.loads((base / "candidate.json").read_text())
    manifest.update(
        candidate=args.candidate,
        base=args.base,
        delta="只有Q融合且当次叶计划的名义步数至少8时使用4槽，其余2槽；不按batch白名单",
        decision_inputs="TMR当次设备最大长度与实际T，复用indexer_long_leaf_plan；不增加device回读或新task",
        cost="编排侧新增少量整数叶计划推导及分支，完整CSA仍需与正式版实测",
        caveat="8步阈值来自B24/B28受益及B16的5/6步叶回退，不宣称所有资源域最优；末尾实际不足4步已单独验证",
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
