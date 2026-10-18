#!/usr/bin/env python3
"""使用PyPTO原清理函数整理全核轨迹，保留原始Scalar与同步图供回查。

直接读取CANN已导出的TRACE JSON，避开历史CANN SOURCE容器长度差异；
不改原始二进制、不改任何timestamp。清理图的箭头来自SET/WAIT_FLAG，
不能据此声称普通GM notify/wait也已被自动识别为跨核箭头。
"""

import importlib.util
import json
from pathlib import Path

from analyze_incore import DESTINATION, WORKSPACE, core_key
from prepare_standalone import HERE


def main():
    path = WORKSPACE / ".worktrees/pypto-csa-memory-reuse-20261018/python/pypto/tools/clean_sim_trace.py"
    spec = importlib.util.spec_from_file_location("pypto_original_trace_cleaner", path)
    cleaner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cleaner)
    coverage = json.loads((HERE / "INCORE_COVERAGE.json").read_text())
    raw = json.loads(Path(coverage["source_trace"]).read_text())
    print("调用原版PyPTO rebuild_trace，72核共同时间轴", flush=True)
    result, skipped = cleaner.rebuild_trace(raw)
    core_names = {event["pid"] for event in result["traceEvents"] if event.get("ph") == "X"}
    assert len(core_names) == 72
    for event in result["traceEvents"]:
        if event.get("name") == "process_sort_index":
            group, role = core_key(event["pid"])
            event["args"]["sort_index"] = group * 3 + role
    target = DESTINATION / "CSA_128K_B16_S6_72cores.clean.json"
    target.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    report = dict(cleaner_source=str(path), raw_source=coverage["source_trace"], cleaned=str(target),
                  cores=len(core_names), raw_events=len(raw["traceEvents"]), clean_events=len(result["traceEvents"]),
                  skipped_sync_arrows=skipped, timestamps_changed=False,
                  scalar_and_original_sync="raw_simulator/trace.json中完整保留",
                  flow_scope="清理箭头仅对应SET/WAIT_FLAG；普通GM notify/wait需结合Scalar原始事件和源码分析")
    (HERE / "TRACE_CLEANING.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (DESTINATION / "trace_cleaning.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
