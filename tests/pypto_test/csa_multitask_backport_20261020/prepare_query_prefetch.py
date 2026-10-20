# SPDX-License-Identifier: Apache-2.0
"""迁移9d5327e5的Query整波头分组和ef608731的短G2资源区间预取。"""

import argparse
import ast
import copy
import json
import re
import shutil
import textwrap

from prepare import HERE, section


def migrate_query(source, donor):
    old = section(source, "        # 按本档位的 token 数", "        for dq_work in pl.range(")
    new = section(donor, "    dq_token_tiles =", "    # 将“八行块、head组”")
    # 多task可按tile提交；分组依据当前tile的真实行数，保留tile_base与原消费者。
    new = re.sub(r"\btokens\b", "tile_rows", new)
    replacement = "        # 9d5327e5：完整八行按48-AIV整波选最大head组；尾行保持4。\n" + textwrap.indent(
        textwrap.dedent(new), " " * 8
    )
    result = source.replace(old, replacement, 1)
    ast.parse(result)
    return result


def migrate_prefetch(source):
    tree = ast.parse(source)
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_native_cube"
    )
    arguments = [a.arg for a in function.args.args]
    gi = arguments.index("query_group_size")
    si = arguments.index("score_tile")
    pi = arguments.index("key_prefetch_panels")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function.name
        and isinstance(node.args[gi], ast.Constant)
        and node.args[gi].value == 2
        and isinstance(node.args[si], ast.Name)
        and node.args[si].id == "BUFFERED_SCORE_TILE"
    ]
    if len(calls) != 1 or ast.literal_eval(calls[0].args[pi]) != 0:
        raise AssertionError("需要唯一的短G2、未预取调用")
    call = calls[0]
    lines = source.splitlines(keepends=True)
    original = "".join(lines[call.lineno - 1 : call.end_lineno])
    indent = " " * (len(original) - len(original.lstrip()))
    candidate = copy.deepcopy(call)
    candidate.args[pi] = ast.Constant(1)
    assignment = "score_tid = " + ast.unparse(candidate) + "\n"
    replacement = (
        indent
        + "# ef608731：半叶排序不超过2048项时预取一个Key面板。\n"
        + indent
        + "short_steps_upper = (max_topk_cache_len + BUFFERED_SCORE_TILE - 1) // BUFFERED_SCORE_TILE\n"
        + indent
        + "short_half_span_upper = short_steps_upper * (BUFFERED_SCORE_TILE // 2)\n"
        + indent
        + "if short_half_span_upper <= SHORT_PREFETCH_HALF_SORT_LIMIT:\n"
        + textwrap.indent(assignment, indent + "    ")
        + indent
        + "else:\n"
        + textwrap.indent(original, "    ")
    )
    lines[call.lineno - 1 : call.end_lineno] = [replacement]
    result = "".join(lines)
    marker = "\n@pl.jit"
    position = result.index(marker)
    result = (
        result[:position] + "\n# 单SPMD已验证的半叶直接归并资源边界。\n"
        "SHORT_PREFETCH_HALF_SORT_LIMIT = 2048\n\n" + result[position:]
    )
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="physical_static")
    parser.add_argument("--candidate", default="query_prefetch")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    query = target / "qkv_proj_rope.py"
    donor = (HERE / "sources/spmd/query_projection_stages.py").read_text()
    query.write_text(migrate_query(query.read_text(), donor))
    indexer = target / "decode_indexer.py"
    indexer.write_text(migrate_prefetch(indexer.read_text()))
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donors=["9d5327e5", "ef608731"],
                changed_files=[query.name, indexer.name],
                scope="仅Query独立head分工及短G2既有Key预取开关；不改数学、任务数量或同步合同",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
