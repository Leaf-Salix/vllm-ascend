# SPDX-License-Identifier: Apache-2.0
"""组合两个互斥资源区间候选；保留已采用B1，不搬入负性能或无效分支。

B16来源是已测WS+scale的guarded版本，开启/关闭PTO IR均已核对等价。
B28来源是原正反向通过的Score归属映射，只收窄Host区间到144<T<=256。
后者关闭时出现一处纯Scalar cast位置差异，故对B24做真实完整回归，而不
再为逐字一致复制大段调用链。IfExp在本版本DSL不支持的尝试只作历史保留。
"""

import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


def main():
    here = Path(__file__).resolve().parent
    owner = here / "source_score_owner_interval"
    stream = here / "source_score_stream_guarded"
    target = here / "source_focus_combined"
    shutil.copytree(owner, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(stream / "indexer_score_stages.py", target / "indexer_score_stages.py")
    path = target / "policy.py"
    text = path.read_text()
    text = replace(text, "    split_score_rows: bool = False\n",
                   "    split_score_rows: bool = False\n    score_ws_pool_panels: int = 1\n")
    text = replace(text, '            "SPLIT_SCORE_ROWS": self.split_score_rows,\n',
                   '            "SPLIT_SCORE_ROWS": self.split_score_rows,\n'
                   '            "SCORE_WS_POOL_PANELS": self.score_ws_pool_panels,\n')
    text = replace(text, "                               (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS)),",
                   "                               (parallel_small_qa or tokens >= EARLY_GEOMETRY_MIN_TOKENS)),")
    text = replace(text,
                   "        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,",
                   "        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,\n"
                   "        # 64..96现有资源区间：双QK占96KiB Acc，余32KiB合并8个WS列窗。\n"
                   "        # 其它区间编译回原单窗；不跨K重排，不改GM FIFO与同步语义。\n"
                   "        score_ws_pool_panels=8 if (long_history and group == 6 and\n"
                   "                                  EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS) else 1,")
    text = text.replace("中间域保留逐页读取。", "T32..63保留逐页读取，T64..96与WS列池共同启用。")
    path.write_text(text)
    for name in ("entry.py", "cube_schedule.py"):
        path = target / name
        text = path.read_text()
        count = 2 if name == "entry.py" else 1
        text = replace(text, "    SPLIT_SCORE_ROWS: pl.constexpr = False,",
                       "    SPLIT_SCORE_ROWS: pl.constexpr = False,\n    SCORE_WS_POOL_PANELS: pl.constexpr = 1,", count)
        indent = "            " if name == "entry.py" else "        "
        text = replace(text, indent + "SPLIT_SCORE_ROWS,",
                       indent + "SPLIT_SCORE_ROWS,\n" + indent + "SCORE_WS_POOL_PANELS,", count)
        path.write_text(text)
    spec = importlib.util.spec_from_file_location("focus_combined_policy", target / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    controls = {}
    for batch in (1, 4, 8, 16, 24, 28):
        chosen = module.select_policy(tokens=batch * 6, requests=batch, max_query_tokens=6, max_seq_len=131078).constexprs()
        controls[str(batch)] = chosen
        assert chosen["SCORE_PREFIX_BALANCE"] == (batch == 28)
        assert chosen["SCORE_WS_POOL_PANELS"] == (8 if batch == 16 else 1)
    for path in target.glob("*.py"):
        ast.parse(path.read_text())
    original = here / "source_after_b1_half_v2"
    changed = [path.name for path in target.glob("*.py")
               if not (original / path.name).exists() or path.read_bytes() != (original / path.name).read_bytes()]
    report = dict(reference=str(original), candidate=str(target), source_owner=str(owner), source_stream=str(stream),
                  changed_files=changed, controls=controls, adopted=False, full_integration_validated=False,
                  excluded=["source_score_owner_uniform", "source_score_owner_interval_guarded（IfExp不支持）", "source_b1_ob16", "B1连续scale读取"])
    (here / "FOCUS_COMBINED_SOURCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(candidate=str(target), changed_files=changed), ensure_ascii=False))


if __name__ == "__main__":
    main()
