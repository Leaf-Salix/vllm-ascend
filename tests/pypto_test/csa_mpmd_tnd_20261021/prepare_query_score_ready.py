# SPDX-License-Identifier: Apache-2.0
"""在局部 query MIX 候选上限制 Score 抢先占核，保留独立多 task。

Score 等主 Compressor cache 与 Q 后处理，避免局部 query 提前完成后，
长 MIX 反而把 Attention 的其他生产者压到 Score 之后。这是调度取舍，
不是新增数学依赖；最终以同进程完整 L1 验证，不预设有收益。
"""

import argparse
import ast
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from prepare_priority_fusion import scope_name, set_keyword


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("parent")
    parser.add_argument("--release-query-priority", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    name, parent = args.name, args.parent
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    path = target / "entry.py"
    source = path.read_text()
    scopes = {scope_name(node): node for node in ast.walk(ast.parse(source)) if scope_name(node)}
    assert len(scopes) == 36 and "indexer_query_chain" in scopes
    query = scopes["qproj_dequant_rms_nope_rope"]
    assert query.items[0].optional_vars is None
    query.items[0].optional_vars = ast.Name(id="query_ready_tid", ctx=ast.Store())
    score = scopes["index_score_topk"]
    deps = next(item.value for item in score.items[0].context_expr.keywords if item.arg == "deps")
    assert ast.unparse(deps) == "[idx_cache_tid]"
    set_keyword(score, "deps", ast.parse("[idx_cache_tid, main_pool_tid, query_ready_tid]", mode="eval").body)
    changed = {"qproj_dequant_rms_nope_rope": query, "index_score_topk": score}
    if args.release_query_priority:
        # v18只等query投影/RoPE；融合后TaskId变成等完整五阶段，不能照搬。
        # 删除三个纯调度边，Q_B/后处理/Compressor的真实Tensor依赖仍完整。
        for name in ("compressor_kv_score", "qproj_b", "qproj_dequant_rms_nope_rope"):
            scope = scopes[name]
            call = scope.items[0].context_expr
            deps = next(item.value for item in call.keywords if item.arg == "deps")
            assert ast.unparse(deps) == "[index_query_tid]"
            call.keywords = [item for item in call.keywords if item.arg != "deps"]
            changed[name] = scope
        scope = scopes["indexer_query_chain"]
        assert ast.unparse(scope.items[0].optional_vars) == "index_query_tid"
        scope.items[0].optional_vars = None
        changed["indexer_query_chain"] = scope
    lines = source.splitlines(keepends=True)
    for node in sorted(changed.values(), key=lambda value: value.lineno, reverse=True):
        lines[node.lineno - 1 : node.end_lineno] = [
            textwrap.indent(ast.unparse(ast.fix_missing_locations(node)), "    ") + "\n"
        ]
    path.write_text("".join(lines))
    print(path)


if __name__ == "__main__":
    main()
