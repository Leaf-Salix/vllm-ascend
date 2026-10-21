# SPDX-License-Identifier: Apache-2.0
"""迁移已提交的双QK＋WS列池小收益候选；保持多task拓扑与AIV不变。"""

import argparse
import ast
import json
import shutil
import subprocess
import textwrap

from prepare import HERE, REPO

DONOR_COMMIT = "fd55e29a"
DONOR_FILE = "tests/pypto_test/csa_critical_path_20261020/score_split_pool/source_candidate/indexer_score_stages.py"


def assignment_block(tree, name, count):
    """找出唯一赋值及同一语句块中的后续语句，避免替换其它策略。"""
    matches = []
    for parent in ast.walk(tree):
        for _, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue
            for index, node in enumerate(value):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == name for target in node.targets
                ):
                    matches.append(value[index : index + count])
    if len(matches) != 1 or len(matches[0]) != count:
        raise ValueError(f"赋值区间不唯一或长度不符：{name}")
    return matches[0]


def excerpt(source, nodes):
    return textwrap.dedent("".join(source.splitlines(keepends=True)[nodes[0].lineno - 1 : nodes[-1].end_lineno]))


def prepare(base, candidate):
    target = HERE / "sources" / candidate
    source = (HERE / "sources" / base / "decode_indexer.py").read_text()
    donor = subprocess.check_output(["git", "show", f"{DONOR_COMMIT}:{DONOR_FILE}"], cwd=REPO, text=True)
    original_tree = ast.parse(source)
    donor_tree = ast.parse(donor)
    original_kernel = next(
        n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_native_cube"
    )
    donor_kernel = next(
        n for n in donor_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store"
    )
    # pool有后续assemble赋值，创建语句按目标和tile.create限定。
    creation = [
        n
        for n in ast.walk(donor_kernel)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "split_ws_pool" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.tile.create"
    ]
    if len(creation) != 1:
        raise ValueError("来源列池创建不唯一")
    edits = [
        (assignment_block(original_kernel, "split_ws_acc", 1), creation),
        (
            assignment_block(original_kernel, "split_ws_first", 3),
            assignment_block(donor_kernel, "split_ws_col", 6),
        ),
        (
            assignment_block(original_kernel, "split_ws_first_last", 3),
            assignment_block(donor_kernel, "split_ws_last_col", 6),
        ),
    ]
    for old, _ in edits[1:]:
        if not isinstance(old[-1], ast.Expr) or ast.unparse(old[-1].value.func) != "pl.store":
            raise ValueError("原WS区间必须以原FIFO写回结束")
    lines = source.splitlines(keepends=True)
    for old, new in sorted(edits, key=lambda pair: pair[0][0].lineno, reverse=True):
        # 前端在折叠split constexpr之前也检查G2/N128/score_tile768分支。
        # 用整步一半宽度表示来源的8*N64；真实启用的长G6仍是16x512。
        # 未启用分支也须有合法Tile/FIFO宽度，不改变选择区间。
        replacement = excerpt(donor, new).replace("SPLIT_WS_POOL_PANELS", "(score_tile // (2 * qk_panel_cols))")
        lines[old[0].lineno - 1 : old[-1].end_lineno] = [textwrap.indent(replacement, " " * old[0].col_offset)]
    edited = "".join(lines)
    marker = "SCORE_SPLIT_MIN_QUERY_ROWS = 32\n"
    if edited.count(marker) != 1:
        raise ValueError("split资源域常量不唯一")
    edited_tree = ast.parse(edited)
    # 只修改Acc存放/写回；矩阵算术次数及全部硬件握手保持。
    for operation in ("pl.tile.matmul_acc", "pl.system.sync_set", "pl.system.sync_wait"):
        counts = [
            sum(isinstance(n, ast.Call) and ast.unparse(n.func) == operation for n in ast.walk(tree))
            for tree in (original_tree, edited_tree)
        ]
        if counts[0] != counts[1]:
            raise ValueError(f"改变了原算术或握手数目：{operation} {counts}")
    vector_marker = "        for buf_score_lane in pl.split_aiv("
    if source.count(vector_marker) != 1 or source.split(vector_marker, 1)[1] != edited.split(vector_marker, 1)[1]:
        raise ValueError("AIV或后续编排发生意外修改")
    shutil.copytree(HERE / "sources" / base, target)
    (target / "decode_indexer.py").write_text(edited)
    (target / "candidate.json").write_text(
        json.dumps(
            {
                "candidate": candidate,
                "base": base,
                "donor_commit": DONOR_COMMIT,
                "donor_file": DONOR_FILE,
                "changed_files": ["decode_indexer.py"],
                "changes": "仅既有split长G6/N64区间，WS由16次小写回改为2次512列写回",
                "preserved": "双QK早FIX、两段K192顺序、全部AIV、TaskId/READY/ACK与原动态范围",
                "adaptation": "半步列宽在长G6严格为512；让折叠前的未启用G2分支也满足FIFO容量",
                "source_evidence": "../csa_critical_path_20261020/score_split_pool/PERFORMANCE.json",
                "source_limit": "来源两长档精确通过且中位数净省5.42/12.12us；未扩七档，不预设迁移收益",
                "status": "private_candidate_no_device_evidence",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="plan_workers_v1")
    parser.add_argument("--candidate", default="ws_pool_v2")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    print(prepare(args.base, args.candidate))


if __name__ == "__main__":
    main()
