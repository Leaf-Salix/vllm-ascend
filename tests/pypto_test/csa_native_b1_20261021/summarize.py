# SPDX-License-Identifier: Apache-2.0
"""离线汇总 Native 128K/B1/S6 补测，复用原五口径报告的优化生效检查。

本脚本不启动设备任务，不改变原始样本，不覆盖 10/20 的七档历史表。
主比较值始终是 30 次 PyTorch Profiler 全流设备跨度的中位数；20 次
无 profiler 的 Event 样本另列，不能将两种口径混作一个结果。
"""

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    task = json.loads((HERE / "TASK.json").read_text())
    # 保留任务中的原始绝对路径，同时允许换一个 checkout 离线复核已提交的报告。
    output = REPO / task["raw_output_relative"]
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "MEASURED", report.get("error", report["status"])
    assert (report["history"], report["batch"], report["query_tokens"]) == (131072, 1, 6)
    assert report["tokens"] == 6 and report["layer_index"] == 4

    validator_path = (
        HERE / "source_snapshot/tests/pypto_test/csa_five_way_latest_20261020/summarize.py"
    )
    spec = importlib.util.spec_from_file_location("csa_native_b1_original_validator", validator_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    # 原始报告中的路径属于实测机器；只重定位内存中的副本，不改写原始证据。
    profile_dirs = sorted(output.glob("profile/*/ASCEND_PROFILER_OUTPUT"))
    assert len(profile_dirs) == 1
    report["device_profile"]["raw_csv"] = str(profile_dirs[0] / "kernel_details.csv")
    profile_median, evidence = validator.outer(report_path, report, native=True)
    for guards in (report["initial_guards"], report["timing"]["guards"]):
        assert guards and all(value["status"] == "PASS" for value in guards.values())
    event_samples = report["timing"]["samples_us"]
    assert len(event_samples) == 20
    trace_paths = sorted(output.glob("profile/**/trace_view.json"))
    assert trace_paths, "未找到原始 PyTorch profiling trace"
    profile_samples = evidence["samples_us"]
    summary = {
        "status": "PASS",
        "measured_date": "2026-10-21",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "task": task["task"],
        "history": 131072,
        "batch": 1,
        "query_tokens": 6,
        "tokens": 6,
        "layer_index": 4,
        "scope": task["scope"],
        "fixture": task["fixture"],
        "checkpoint": report["checkpoint"],
        "device": report["device"],
        "input_shapes": report["input_shapes"],
        "versions": report["versions"],
        "cann": report["cann"],
        "native_us": profile_median,
        "primary_metric": "30次PyTorch Profiler完整层全设备流首尾跨度中位数，单位us",
        "profile": {
            "samples_us": profile_samples,
            "median_us": profile_median,
            "min_us": min(profile_samples),
            "max_us": max(profile_samples),
            "trace_paths": [str(path) for path in trace_paths],
        },
        "unprofiled_event": {
            "samples_us": event_samples,
            "median_us": median(event_samples),
            "min_us": min(event_samples),
            "max_us": max(event_samples),
            "scope": "无Profiler、图外Event包围单次重放，包含运行时；另列，不替代主指标",
        },
        "evidence": evidence,
        "source_commit_at_submission": task["source_commit_at_submission"],
        "report": str(report_path),
        "comparison_note": "沿用10/20 Native配置与计量脚本；跨日期与已有PTO L1比较时标明口径和轮次",
    }
    (HERE / "PERFORMANCE.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    text = f"""# Native 128K/B1/S6 性能补测

日期：2026-10-21。仅补这一档；使用原 10/20 Native 五口径测试脚本，没有修改算子。

| 计量方式 | 样本数 | 中位数 μs | 最小值 μs | 最大值 μs |
| --- | ---: | ---: | ---: | ---: |
| PyTorch Profiler 全流整层跨度（主对比值） | 30 | {profile_median:.2f} | {min(profile_samples):.2f} | {max(profile_samples):.2f} |
| 无 Profiler 的图外 NPU Event | 20 | {median(event_samples):.2f} | {min(event_samples):.2f} | {max(event_samples):.2f} |

范围：TP1，正式 W8A8 第4层、S6/T6，HC_pre → input RMSNorm → Native CSA → HC_post。
hidden 与历史 KV 是 seed=1024 的合成输入；不是整模型或服务吞吐测试。

优化生效检查通过：

- 四张核心投影实际 NZ 格式29，保持原 Native DSA 多流。
- npugraph_ex，fullgraph=True、dynamic=False；静态编译实际成功并安装。
- SuperKernel 实际成功返回，trace 存在 sk_*static_kernel 与 MatmulWeightNz。
- 唯一 ACLGraph 直接重放，同初态输出/Top-K/cache/state 对照及保护区通过。
- 30次完整设备跨度保留全部流和图内复制，排除图外 fixture reset/输出毒化。

后续五组表格的 Native B1 列使用 **{profile_median:.2f} μs**。Event 值另列，不能
混入原 Profiler 列；与此前 PTO L1 的百分比仍属于跨轮、外层/核内口径参考。
全部原始样本、优化检查证据及实际 trace 路径见 [PERFORMANCE.json](PERFORMANCE.json)。

任务：`{task['task']}`。
原始结果：`{task['raw_output']}`。
原七档历史数值保持不变；权重、OPP及静态二进制仅本地保留。

本次按用户要求归档脚本快照、完整原始报告、全部样本、运行日志、profiling JSON
及其 CSV 文本导出。文件清单见 [COMMIT_FILES.json](COMMIT_FILES.json)，大产物
路径与字节数见 [LOCAL_ARTIFACTS.json](LOCAL_ARTIFACTS.json)，没有新增设备测试。

可直接查看[原始报告]({Path('../csa_five_way_latest_20261020/results/h131072_b1/native_supplement_20261021_v1/report.json')})、
[PyTorch profiling JSON]({Path('../csa_five_way_latest_20261020/results/h131072_b1/native_supplement_20261021_v1') / trace_paths[0].relative_to(output)})。

复现命令见 [TASK.json](TASK.json)，必须将 attempt 换成新名字，原脚本拒绝覆盖。
原计量与校验代码已保存到 source_snapshot。只重新汇总已有结果：

```bash
python tests/pypto_test/csa_native_b1_20261021/summarize.py
```
"""
    (HERE / "README.md").write_text(text)
    print(json.dumps({"status": "PASS", "native_us": profile_median,
                      "event_median_us": median(event_samples),
                      "trace_paths": summary["profile"]["trace_paths"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
