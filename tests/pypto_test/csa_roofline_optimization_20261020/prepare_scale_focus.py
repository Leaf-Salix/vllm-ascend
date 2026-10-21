# SPDX-License-Identifier: Apache-2.0
"""从已采用B1的正式快照生成连续scale读取候选，只改变区间策略。

既有META逐页证明、正/逆序二维读取和碎片回退均原样复用；不改变数学、
叶边界、同分顺序或同步。B1的新半叶为512粒度，仍满足32行页对齐。
"""

import importlib.util
import json
from pathlib import Path
import shutil
import sys


def load_policy(path, label):
    spec = importlib.util.spec_from_file_location(label, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[label] = module
    spec.loader.exec_module(module)
    return module


def main():
    here = Path(__file__).resolve().parent
    reference = here / "source_after_b1_half_v2"
    candidate = here / "source_scale_focus"
    shutil.copytree(reference, candidate, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = candidate / "policy.py"
    old = """        scale_page_coalescing=(long_history and group == 6 and
                               (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS)),"""
    new = """        # 本候选：B1已改为原半叶独立调度，每个AIV只处理一条query，
        # 尝试把其32次逐页scale读取合并。末半叶512粒度仍按32行页对齐。
        # G6则扩到已有前置几何的T>=64区间；保留B8所在中间小档关闭。
        # 页是否连续始终由本轮device完整证明，不能依据Host档位猜页布局。
        scale_page_coalescing=(single_request_leaves or
                               (long_history and group == 6 and
                                (parallel_small_qa or tokens >= EARLY_GEOMETRY_MIN_TOKENS))),"""
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError("连续scale策略上下文已改变，拒绝模糊替换")
    path.write_text(text.replace(old, new))
    before = load_policy(reference / "policy.py", "scale_focus_reference_policy")
    after = load_policy(path, "scale_focus_candidate_policy")
    rows = []
    for history in (8192, 32768, 131072, 262144):
        for batch, query, capacity in ((1, 1, 1), (1, 1, 6), (1, 5, 5), (1, 6, 6),
                                       (4, 6, 6), (8, 6, 6), (10, 6, 6), (11, 6, 6),
                                       (16, 5, 6), (16, 6, 6), (24, 6, 6), (28, 6, 6), (32, 6, 6)):
            args = dict(tokens=batch * query, requests=batch,
                        max_query_tokens=capacity, max_seq_len=history + query)
            previous = before.select_policy(**args).constexprs()
            current = after.select_policy(**args).constexprs()
            changed = {key: [previous[key], current[key]] for key in previous if previous[key] != current[key]}
            if set(changed) - {"SCALE_PAGE_COALESCING"}:
                raise AssertionError((args, changed))
            if batch in (8, 28) and changed:
                raise AssertionError("B8/B28策略不应改变")
            rows.append(dict(arguments=args, changed=changed))
    for batch in (1, 16):
        controls = after.select_policy(tokens=batch * 6, requests=batch,
                                       max_query_tokens=6, max_seq_len=131078).constexprs()
        (here / f"controls_scale_focus_b{batch}.json").write_text(json.dumps(controls, indent=2) + "\n")
    report = {"reference": str(reference), "candidate": str(candidate),
              "changed_files": ["policy.py"], "cases": rows, "passed": True,
              "device_precision_validated": False, "performance_measured": False}
    (here / "SCALE_FOCUS_SOURCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(candidate)


if __name__ == "__main__":
    main()
