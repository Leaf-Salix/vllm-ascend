# SPDX-License-Identifier: Apache-2.0
"""将Attention输出缓冲留在各前端分支末尾分配，避免提升到临时缓冲之前。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="query_score_fifo_policy_v11")
    parser.add_argument("--candidate", default="query_score_packed_order_v12")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    source = (base / "decode_csa.py").read_text()
    tree = ast.parse(source)
    choices = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "defer_q_to_score == 1"]
    allocations = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "o_packed_heads"
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.create_tensor"
    ]
    if len(choices) != 1 or len(allocations) != 1:
        raise ValueError("必须只有一处分流和一处公共packed分配")
    choice, allocation = choices[0], allocations[0]
    if allocation.lineno <= choice.end_lineno or allocation.col_offset != choice.col_offset:
        raise ValueError("packed必须原本位于分流后的同一scope")
    lines = source.splitlines(keepends=True)
    declaration = textwrap.dedent("".join(lines[allocation.lineno - 1 : allocation.end_lineno]))
    comment = "# 留在本分支末尾，使hoist后的packed仍排在前端暂存之后；不新增scope。\n"
    addition = textwrap.indent(comment + declaration, " " * (choice.col_offset + 4))
    edits = [
        (allocation.lineno - 1, allocation.end_lineno, []),
        (choice.body[-1].end_lineno, choice.body[-1].end_lineno, [addition]),
        (choice.orelse[-1].end_lineno, choice.orelse[-1].end_lineno, [addition]),
    ]
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = replacement
    source = "".join(lines)
    after = ast.parse(source)
    if sum(isinstance(n, ast.With) for n in ast.walk(tree)) != sum(isinstance(n, ast.With) for n in ast.walk(after)):
        raise ValueError("不得新增scope或SPMD任务")
    shutil.copytree(base, target)
    (target / "decode_csa.py").write_text(source)
    manifest = json.loads((base / "candidate.json").read_text())
    manifest.update(
        candidate=args.candidate,
        base=args.base,
        delta="仅将同一packed输出分配分别放在两个前端分支末尾，恢复其位于前端暂存之后的分配顺序",
        affected_files_this_step=["decode_csa.py"],
        unchanged=(
            "无新增scope、数学task、显式依赖、信号、数学、cache布局或槽数策略；"
            "合并分配可减少无kernel的allocation creator"
        ),
        evidence=(
            "短B16/B32 Attention执行期间无其它incore重叠，实际AIC/AIV代码相同，生成编排的packed分配从末尾移到了最前"
        ),
        caveat="编排分配顺序改变是证据，设备地址/缓存冲突尚未直接观测；必须实测，不预称根因已确认",
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
