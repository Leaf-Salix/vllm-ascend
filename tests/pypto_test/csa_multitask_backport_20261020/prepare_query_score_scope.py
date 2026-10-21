# SPDX-License-Identifier: Apache-2.0
"""为Q/Score融合的前段临时工作区建立独立scope，不增加占位Tensor。"""

import argparse
import ast
import json
import shutil
import textwrap

from prepare import HERE
from prepare_query_ready import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="query_score_v1")
    parser.add_argument("--candidate", default="query_score_scope_v5")
    parser.add_argument("--keep-inner-indexer-scope", action="store_true", help="仅复现v4的ring 3失败")
    parser.add_argument("--keep-q-alias", action="store_true", help="仅复现v3的子scope Q别名编译失败")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    source = (base / "decode_csa.py").read_text()
    start = source.index("            # Projection-chain dependency marker.\n")
    end = source.index("            # sparse_attn_csa folds", start)
    old = source[start:end]
    if old.count("qkv_prepare_query_score(") != 1 or old.count("= indexer(") != 1:
        raise ValueError("必须包含完整Q准备及Indexer前段")
    helper = next(
        n
        for n in ast.parse((base / "qkv_proj_rope.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "qkv_prepare_query_score"
    )
    returns = [n for n in ast.walk(helper) if isinstance(n, ast.Return)]
    if len(returns) != 1 or ast.unparse(returns[0].value.elts[0]) != "q":
        raise ValueError("准备函数必须直接返回原q句柄")
    if any(isinstance(n, ast.Name) and n.id == "q" and isinstance(n.ctx, ast.Store) for n in ast.walk(helper)):
        raise ValueError("准备函数不得重新绑定q")
    # q已经分配于外层；不要用子scope内的返回别名覆盖外层句柄。
    if not args.keep_q_alias:
        old = replace_once(
            old,
            "q, qa_tid, fused_q_scale, fused_q_projected, fused_q_sin, fused_q_swap =",
            "_prepared_q, qa_tid, fused_q_scale, fused_q_projected, fused_q_sin, fused_q_swap =",
        )
    # 移动原Indexer scope起点，而非增加层级。否则原135MB工作区落入32MB ring 3。
    flat = textwrap.dedent(old)
    scopes = [
        n
        for n in ast.walk(ast.parse(flat))
        if isinstance(n, ast.With) and len(n.items) == 1 and ast.unparse(n.items[0].context_expr) == "pl.scope()"
    ]
    if len(scopes) != 1 or scopes[0].col_offset != 0:
        raise ValueError("前段应只有一个顶层Indexer scope")
    scope = scopes[0]
    flat_lines = flat.splitlines(keepends=True)
    payload = textwrap.dedent("".join(flat_lines[scope.body[0].lineno - 1 : scope.body[-1].end_lineno]))
    if not args.keep_inner_indexer_scope:
        flat_lines[scope.lineno - 1 : scope.end_lineno] = [payload]
        old = textwrap.indent("".join(flat_lines), " " * 12).replace(
            "Bound indexer scratch to its own runtime scope.",
            "Indexer与Q临时缓冲共用前段scope，保持原Indexer's heap层级。",
        )
    replacement = (
        "            # Q与idx_topk是外层分配/入参；QB、scale、RoPE等临时缓冲只归前段。\n"
        "            # Scope关闭后，外层Attention仍通过真实Q/TopK/cache生产者建立依赖。\n"
        "            with pl.scope():\n" + textwrap.indent(old, "    ")
    )
    updated = source[:start] + replacement + source[end:]
    # 去掉新增scope后操作树应完全一致，不借生命周期修改叠加数学策略。
    tree = ast.parse(updated)
    matching = [n for n in ast.walk(tree) if isinstance(n, ast.With) and n.lineno == source[:start].count("\n") + 3]
    if len(matching) != 1:
        raise ValueError("新增scope定位不唯一")
    if ast.dump(ast.Module(body=matching[0].body, type_ignores=[])) != ast.dump(ast.parse(textwrap.dedent(old))):
        raise ValueError("前段操作发生意外变化")
    shutil.copytree(base, target)
    (target / "decode_csa.py").write_text(updated)
    manifest = json.loads((base / "candidate.json").read_text())
    manifest.update(
        candidate=args.candidate,
        base=args.base,
        delta="将原Indexer scope起点前移并丢弃与输入q相同的返回别名；不加嵌套层级，无占位或新信号",
        lifetime="Q/TopK/cache在外层，临时QB/scale/RoPE/Indexer工作区归前段；后续不再引用它们",
        caveat="不声称完整恢复原QKV循环；需验证scope传播的依赖和实际性能",
        status="private_candidate_no_device_evidence",
        repro_only_keep_inner_scope=args.keep_inner_indexer_scope,
        repro_only_keep_q_alias=args.keep_q_alias,
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
