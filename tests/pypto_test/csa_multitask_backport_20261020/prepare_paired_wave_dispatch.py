# SPDX-License-Identifier: Apache-2.0
"""将已测两波策略移到编排层，constexpr隔离两套Score前缀实现。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE
from prepare_query_ready import replace_once


def migrate(source):
    selector = (
        "        # 已测B8两波区间成对压缩回退；按实际分组数×叶数选择，不枚举batch。\n"
        "        buf_use_paired_prefix = pl.cast(1, pl.INT32)\n"
        "        if balance_leaves:\n"
        "            buf_score_items = buf_query_count // query_group_size * buf_max_leaves\n"
        "            if buf_score_items > TOPK_SCORE_WORKERS:\n"
        "                if buf_score_items <= 2 * TOPK_SCORE_WORKERS:\n"
        "                    buf_use_paired_prefix = pl.cast(0, pl.INT32)\n"
    )
    source = replace_once(source, selector, "")
    source = replace_once(source, "if buf_use_paired_prefix == 0:", "if not use_paired_prefix:")
    tree = ast.parse(source)
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_native_cube")
    lines = source.splitlines(keepends=True)
    header_end = next(i for i in range(kernel.lineno, kernel.end_lineno) if lines[i] == "):\n")
    lines.insert(header_end, "    use_paired_prefix: pl.constexpr,\n")
    source = "".join(lines)
    calls = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "indexer_score_topk_native_cube"
    ]
    if len(calls) != 6 or any(n.keywords for n in calls):
        raise ValueError("需保持六处原Score调用及位置参数约定")
    lines = source.splitlines(keepends=True)
    for call in sorted(calls, key=lambda n: n.end_lineno, reverse=True):
        indent = len(lines[call.end_lineno - 1]) - len(lines[call.end_lineno - 1].lstrip())
        lines.insert(call.end_lineno - 1, " " * (indent + 4) + "False,\n")
    source = "".join(lines)
    forest = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_forest"
    )
    groups = [n for n in ast.walk(forest) if isinstance(n, ast.If) and ast.unparse(n.test) == "long_s6_group == 1"]
    if len(groups) != 1 or len(groups[0].body) != 1:
        raise ValueError("需唯一长G6编排分支")
    old = groups[0].body[0]
    lines = source.splitlines(keepends=True)
    body = textwrap.dedent("".join(lines[old.lineno - 1 : old.end_lineno]))
    # 两个长G6调用的最后一参，不能改动原split_score_rows等布尔参数。
    old_tail = "        layout_tid,\n        False,\n"
    if body.count(old_tail) != 2:
        raise ValueError("长G6应有拆分/非拆分两处调用")
    paired = body.replace(old_tail, "        layout_tid,\n        True,\n")
    policy = (
        "# 只按入口长度上界选择等价算法；实际有效长度/页表仍在原核内读取。\n"
        "# 两波区间沿用旧短前缀，constexpr让每个核函数只包含选定实现。\n"
        "policy_groups = pl.tensor.dim(position_ids, 0) // S\n"
        "policy_leaves, policy_leaf_tiles, policy_extra_leaves = indexer_long_leaf_plan(\n"
        "    pl.min(max_topk_cache_len, TOPK_MAX_CANDIDATES), policy_groups\n"
        ")\n"
        "policy_items = policy_groups * policy_leaves\n"
        "policy_use_paired = pl.cast(1, pl.INT32)\n"
        "if policy_items > TOPK_SCORE_WORKERS:\n"
        "    if policy_items <= 2 * TOPK_SCORE_WORKERS:\n"
        "        policy_use_paired = pl.cast(0, pl.INT32)\n"
        "if policy_use_paired == 1:\n" + textwrap.indent(paired, "    ") + "else:\n" + textwrap.indent(body, "    ")
    )
    lines[old.lineno - 1 : old.end_lineno] = [textwrap.indent(policy, " " * old.col_offset)]
    result = "".join(lines)
    ast.parse(result)
    if "buf_use_paired_prefix" in result:
        raise ValueError("不能残留核内动态选择器")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="paired_wave_policy_v1")
    parser.add_argument("--candidate", default="paired_wave_dispatch_v2")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    source = migrate((base / "decode_indexer.py").read_text())
    shutil.copytree(base, target)
    (target / "decode_indexer.py").write_text(source)
    manifest = {
        "candidate": args.candidate,
        "base": args.base,
        "formal_base": "4d5b2ed9",
        "donor": "c0d27e9a已验证成对压缩，helper与paired_wave_policy_v1相同",
        "changed_files": ["decode_indexer.py", "topk_compact_stages.py"],
        "policy": "入口长度上界×实际query分组推导Score工作项，24<items<=48选旧短前缀，其余成对压缩",
        "change_from_v1": "编排层选择constexpr特化，去掉核内动态分支和未选helper",
        "preserved": "实际leaf/有效长度/页表仍按原核内metadata处理，两种排序均保持完整同分规则",
        "caveat": "长度上界只影响算法选择；同图metadata变短不代表切换策略，需边界验证",
        "context_switch_history": 32765,
        "status": "private_candidate_no_device_evidence",
    }
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
