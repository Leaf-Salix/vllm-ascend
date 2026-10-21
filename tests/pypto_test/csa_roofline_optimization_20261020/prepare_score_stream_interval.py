# SPDX-License-Identifier: Apache-2.0
"""从当前正式源组合Score的WS列池与连续scale读取，仅选既有T64..96区间。

保留B1原半叶修复，不复制历史旧版整个Score文件。复用已验证的WS列池
函数并补回当前显式叶边界参数；池宽改为constexpr，池宽1时仅含一个列窗。
不是B16白名单：条件来自前置几何与局部stats两个既有资源边界。
"""

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"替换上下文计数不是1：{old[:100]!r}")
    return text.replace(old, new)


def function_span(text, name):
    node = next(item for item in ast.parse(text).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    lines = text.splitlines(keepends=True)
    return sum(map(len, lines[:node.lineno - 1])), sum(map(len, lines[:node.end_lineno]))


def main():
    here = Path(__file__).resolve().parent
    original = here / "source_after_b1_half_v2"
    candidate = here / "source_score_stream_interval"
    shutil.copytree(original, candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = candidate / "policy.py"
    text = path.read_text()
    text = replace_once(text, "    split_score_rows: bool = False\n",
                        "    split_score_rows: bool = False\n    score_ws_pool_panels: int = 1\n")
    text = replace_once(text, '            "SPLIT_SCORE_ROWS": self.split_score_rows,\n',
                        '            "SPLIT_SCORE_ROWS": self.split_score_rows,\n'
                        '            "SCORE_WS_POOL_PANELS": self.score_ws_pool_panels,\n')
    text = replace_once(text,
                        "        scale_page_coalescing=(long_history and group == 6 and\n"
                        "                               (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS)),",
                        "        # 与WS列池共同覆盖前置几何已开启、局部stats仍驻留的资源区间。\n"
                        "        # 页连续性仍由本轮device完整证明，碎片页走既有回退。\n"
                        "        scale_page_coalescing=(long_history and group == 6 and\n"
                        "                               (parallel_small_qa or tokens >= EARLY_GEOMETRY_MIN_TOKENS)),")
    text = replace_once(text,
                        "        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,",
                        "        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,\n"
                        "        # 双48KiB QK Acc加32KiB WS列池恰好用满128KiB L0C。\n"
                        "        # 大T有更多前后阶段竞争，保留原4KiB WS，不能照搬小T收益。\n"
                        "        score_ws_pool_panels=8 if (long_history and group == 6 and\n"
                        "                                  EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS) else 1,")
    path.write_text(text)

    # 新增编译期资源参数；不新增device/Host实际长度参数。
    for name in ("entry.py", "cube_schedule.py"):
        path = candidate / name
        text = path.read_text()
        old = "    SPLIT_SCORE_ROWS: pl.constexpr = False,"
        expected = 2 if name == "entry.py" else 1
        assert text.count(old) == expected
        text = text.replace(old, old + "\n    SCORE_WS_POOL_PANELS: pl.constexpr = 1,")
        old = "            SPLIT_SCORE_ROWS," if name == "entry.py" else "        SPLIT_SCORE_ROWS,"
        assert text.count(old) == expected
        indent = "            " if name == "entry.py" else "        "
        text = text.replace(old, old + "\n" + indent + "SCORE_WS_POOL_PANELS,")
        path.write_text(text)

    path = candidate / "indexer_score_stages.py"
    text = path.read_text()
    old_source = (here / "source_score_pool_goal10/indexer_score_stages.py").read_text()
    start, end = function_span(old_source, "_indexer_score_cube_split_store")
    helper = old_source[start:end]
    helper = helper.replace("SPLIT_WS_POOL_PANELS", "WS_POOL_PANELS")
    helper = replace_once(helper, "    SCORE_CONSUMED_EVENT: pl.constexpr,\n)",
                          "    SCORE_CONSUMED_EVENT: pl.constexpr,\n    WS_POOL_PANELS: pl.constexpr,\n)")
    helper = replace_once(helper,
        "            buf_leaf, buf_leaf_tiles, buf_extra_leaves, buf_visible_count\n",
        "            buf_leaf, buf_leaf_tiles, buf_extra_leaves, buf_visible_count,\n"
        "            balance_leaves, query_group_size,\n")
    # 列池1和8都用相同shape的列窗别名，避免将512列父池赋给64列临时量。
    # 宽度1的slice/assemble是否消除，要看生成物；不以源码形式推断零成本。
    helper = helper.replace("八个WS列窗合并一次原布局写回", "按资源区间将1或8个WS列窗写回原布局")
    helper = helper.replace("WS32KiB列池", "WS 4KiB或32KiB列池")
    helper = helper.replace("每列窗init_cond=True初始化。", "每列窗init_cond=True初始化；仅分档改变写回批量。")
    start, end = function_span(text, "_indexer_score_cube_split_store")
    text = text[:start] + helper + text[end:]
    text = replace_once(text, "    SPLIT_SCORE_ROWS: pl.constexpr = False,\n)",
                        "    SPLIT_SCORE_ROWS: pl.constexpr = False,\n    SCORE_WS_POOL_PANELS: pl.constexpr = 1,\n)")
    old = "            SCORE_CONSUMED_EVENT,\n        )\n    else:\n        result = _indexer_score_cube_original("
    new = "            SCORE_CONSUMED_EVENT,\n            SCORE_WS_POOL_PANELS,\n        )\n    else:\n        result = _indexer_score_cube_original("
    text = replace_once(text, old, new)
    path.write_text(text)

    policy_spec = importlib.util.spec_from_file_location("score_stream_policy", candidate / "policy.py")
    module = importlib.util.module_from_spec(policy_spec)
    sys.modules[policy_spec.name] = module
    policy_spec.loader.exec_module(module)
    rows = []
    for history in (8192, 131072):
        for tokens, batch in ((6, 1), (48, 8), (63, 11), (64, 11), (80, 16), (96, 16), (97, 17), (144, 24), (168, 28)):
            control = module.select_policy(tokens=tokens, requests=batch, max_query_tokens=6, max_seq_len=history + 6)
            expected = 8 if history == 131072 and 64 <= tokens <= 96 else 1
            assert control.score_ws_pool_panels == expected, (history, tokens, control)
            rows.append(dict(history=history, tokens=tokens, requests=batch,
                             score_ws_pool_panels=control.score_ws_pool_panels,
                             scale_page_coalescing=control.scale_page_coalescing))
    controls = module.select_policy(tokens=96, requests=16, max_query_tokens=6, max_seq_len=131078).constexprs()
    (here / "controls_score_stream_b16.json").write_text(json.dumps(controls, indent=2) + "\n")
    changed = [p.name for p in original.glob("*.py") if p.read_bytes() != (candidate / p.name).read_bytes()]
    for p in candidate.glob("*.py"):
        ast.parse(p.read_text())
    report = dict(reference=str(original), candidate=str(candidate), changed_files=changed,
                  policy_cases=rows, policy_checks_passed=True, cpu_lowering_passed=False,
                  device_checked=False, adopted=False,
                  intent="组合B16上各约5us的两个小收益；不相加预测；原B1半叶与同分顺序保持")
    (here / "SCORE_STREAM_SOURCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
