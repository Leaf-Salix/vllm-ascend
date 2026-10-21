# SPDX-License-Identifier: Apache-2.0
"""只迁移12089c06已验证的WS列池区间，保留已存在的连续scale与Top-K实现。"""

import argparse
import ast
import json
import shutil
import subprocess
import textwrap

from prepare import HERE, REPO
from prepare_query_ready import replace_once
from prepare_query_score import append_call_arguments, append_parameters
from prepare_ws_pool import assignment_block, excerpt

DONOR_COMMIT = "12089c06"
DONOR_FILE = "tests/pypto_test/csa_roofline_optimization_20261020/source_score_stream_guarded/indexer_score_stages.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="formal_4d5b2ed9")
    parser.add_argument("--candidate", default="ws_interval_v2")
    parser.add_argument("--combined-and", action="store_true", help="复现v1编排And表达式不被当前代码生成支持")
    args = parser.parse_args()
    if not all(name.isidentifier() for name in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / name for name in (args.base, args.candidate))
    source = (base / "decode_indexer.py").read_text()
    donor = subprocess.check_output(["git", "show", f"{DONOR_COMMIT}:{DONOR_FILE}"], cwd=REPO, text=True)
    old_kernel = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_native_cube"
    )
    donor_kernel = next(
        n
        for n in ast.parse(donor).body
        if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store"
    )
    creations = [
        n
        for n in ast.walk(donor_kernel)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "split_ws_pool" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.tile.create"
    ]
    if len(creations) != 1:
        raise ValueError("来源WS池声明必须唯一")
    edits = [
        (assignment_block(old_kernel, "split_ws_acc", 1), creations),
        (assignment_block(old_kernel, "split_ws_first", 3), assignment_block(donor_kernel, "split_ws_col", 6)),
        (
            assignment_block(old_kernel, "split_ws_first_last", 3),
            assignment_block(donor_kernel, "split_ws_last_col", 6),
        ),
    ]
    lines = source.splitlines(keepends=True)
    for old, new in sorted(edits, key=lambda item: item[0][0].lineno, reverse=True):
        original = excerpt(source, old)
        # 未启用的G2分支仍须满足前端类型检查；真实启用的G6/N64恰为8列窗。
        replacement = excerpt(donor, new).replace("WS_POOL_PANELS", "(score_tile // (2 * qk_panel_cols))")
        changed = (
            original + replacement
            if len(old) == 1
            else "if use_ws_pool:\n"
            + textwrap.indent(replacement, "    ")
            + "else:\n"
            + textwrap.indent(original, "    ")
        )
        lines[old[0].lineno - 1 : old[-1].end_lineno] = [textwrap.indent(changed, " " * old[0].col_offset)]
    changed = "".join(lines)
    changed = append_parameters(changed, "indexer_score_topk_native_cube", ["use_ws_pool: pl.constexpr"])
    changed = append_call_arguments(changed, {"indexer_score_topk_native_cube"}, ["use_ws_pool=False"])
    changed = replace_once(
        changed,
        "SCORE_SPLIT_MIN_QUERY_ROWS = 32\n",
        "SCORE_SPLIT_MIN_QUERY_ROWS = 32\n"
        "# 12089c06：前置几何至少两个M32行块，局部Attention统计仍驻留的资源区间。\n"
        "WS_POOL_MIN_QUERY_ROWS = 64\n"
        "WS_POOL_MAX_QUERY_ROWS = 96\n",
    )
    forest = next(
        n for n in ast.parse(changed).body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_forest"
    )
    choices = [
        n
        for n in ast.walk(forest)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "pl.tensor.dim(position_ids, 0) >= SCORE_SPLIT_MIN_QUERY_ROWS"
    ]
    if len(choices) != 1 or len(choices[0].body) != 1:
        raise ValueError("只替换既有长G6拆M路径的一处调用")
    call = choices[0].body[0]
    lines = changed.splitlines(keepends=True)
    disabled = textwrap.dedent("".join(lines[call.lineno - 1 : call.end_lineno]))
    enabled = replace_once(disabled, "use_ws_pool=False", "use_ws_pool=True")
    predicate = "if pool_query_rows >= WS_POOL_MIN_QUERY_ROWS and pool_query_rows <= WS_POOL_MAX_QUERY_ROWS:\n"
    if not args.combined_and:
        predicate = (
            "pool_enabled = pl.cast(0, pl.INT32)\n"
            "if pool_query_rows >= WS_POOL_MIN_QUERY_ROWS:\n"
            "    if pool_query_rows <= WS_POOL_MAX_QUERY_ROWS:\n"
            "        pool_enabled = pl.cast(1, pl.INT32)\n"
            "if pool_enabled != 0:\n"
        )
    branch = (
        "pool_query_rows = pl.tensor.dim(position_ids, 0)\n"
        + predicate
        + textwrap.indent(enabled, "    ")
        + "else:\n"
        + textwrap.indent(disabled, "    ")
    )
    lines[call.lineno - 1 : call.end_lineno] = [textwrap.indent(branch, " " * call.col_offset)]
    changed = "".join(lines)
    ast.parse(changed)
    # 所有AIV语句、实际页表和同分Top-K保持；只插入AIC列池及编排选择。
    marker = "        for buf_score_lane in pl.split_aiv("
    old_vector = source.split(marker, 1)[1].split("@pl.jit", 1)[0]
    new_vector = changed.split(marker, 1)[1].split("@pl.jit", 1)[0]
    if old_vector != new_vector:
        raise ValueError("意外改变AIV代码")
    shutil.copytree(base, target)
    (target / "decode_indexer.py").write_text(changed)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor_commit=DONOR_COMMIT,
                donor_file=DONOR_FILE,
                source_status="已提交正反向及S5验证的独立B16组合；尚未正式合入单SPMD，不包含未收口B28归属优化",
                changed_files=["decode_indexer.py"],
                resource_rule="仅长G6拆M且64<=实际T<=96启用8列窗，其余保持单窗；实际长度和页表仍动态",
                combined_and=args.combined_and,
                change="WS输出由16次64列FIX改成2次512列FIX，完整六query的K192+K192累加顺序不变",
                already_present="多task长G6已经默认证明并批量读取连续scale；此项不重复迁移或认领新收益",
                preserved="全部AIV/Top-K/页表/scale策略、READY/ACK/FIFO槽数、task数量与原调度属性不变",
                source_evidence="SCORE_STREAM_TASK/REVERSE_TASK/S5_TASK；FOCUS_INTERVAL_IR_AUDIT的stream_enabled/disabled为true",
                source_caveat="同一审计文件总体passed=false由独立B28 owner分支引起，不能当整个文件通过",
                context_switch_history=32765,
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
