"""只读正式10步时间戳，定位进入forward前的相对延迟；不删除异常样本。"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIMESTAMP_DOC = "https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_1780.html"
TORCH_SOURCE = "https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/NPUEvent.cpp#L173"


def main():
    forward = json.loads((ROOT / "model/forward.json").read_text())
    result = {"scope": "复用同一次正式forward的设备起始事件，不重跑，不剔除或修正性能样本。",
              "method": "每rank起始时间减rank0，再减该差值的10步中位数，消去稳定的跨设备时钟偏移；"
                        "每步再减16rank中位数，得到相对通常启动关系的瞬时偏移。",
              "units": "torch_npu recorded_time调用aclrtEventGetTimestamp，官方API单位μs。",
              "sources": [TIMESTAMP_DOC, TORCH_SOURCE],
              "limits": "这是短窗口内的相对启动异常，不是绝对同步时间；不校正长期时钟漂移。"
                        "只能定位到forward开始事件之前，不能区分CPU调度、GC、前序草稿或设备队列等待。",
              "cases": []}
    lines = ["# 正式forward尾部：相对进入时刻", "", result["scope"], "", result["method"], "",
             result["limits"], "", "以下只列相对进入异常超过2ms的诊断点；2ms不是验收阈值，JSON保留全部10步。", "",
             "| 档位 | 侧 | 正式step | 相对晚进入rank | 偏移ms | 该rank forward/平时中位ms | 其余rank增加ms |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in forward["cases"]:
        if "forward" not in case:
            continue
        row = {"history": case["history"], "batch": case["batch"]}
        for side in ("native", "pto"):
            folder = ROOT / "model" / f"h{case['history']}" / f"b{case['batch']}" / side
            states = [json.loads((folder / f"rank{r}.performance.json").read_text())["steady_window"][0]
                      for r in range(16)]
            starts = [s["forward"]["start_timestamps_raw"] for s in states]
            durations = [s["forward"]["samples_us"] for s in states]
            relative = [[value - reference for value, reference in zip(values, starts[0])] for values in starts]
            offsets = [statistics.median(values) for values in relative]
            typical = [statistics.median(values) for values in durations]
            samples = []
            for step in range(10):
                values = [relative[r][step] - offsets[r] for r in range(16)]
                center = statistics.median(values)
                excursions = [v - center for v in values]
                late = max(range(16), key=excursions.__getitem__)
                others_delta = statistics.mean(durations[r][step] - typical[r] for r in range(16) if r != late)
                sample = {"step": states[0]["steady_step_indices"][step], "late_rank": late,
                          "relative_entry_excursion_us": excursions, "forward_us": [d[step] for d in durations],
                          "late_rank_typical_forward_us": typical[late], "other_ranks_mean_extra_us": others_delta}
                samples.append(sample)
                if excursions[late] > 2000:
                    lines.append(f"| {case['history']//1024}K/B{case['batch']} | {side} | {sample['step']} | {late} | "
                                 f"{excursions[late]/1000:.3f} | {durations[late][step]/1000:.3f}/"
                                 f"{typical[late]/1000:.3f} | {others_delta/1000:.3f} |")
            row[side] = {"rank_clock_offset_us": offsets, "samples": samples}
        result["cases"].append(row)
    lines += ["", "晚进入rank自身forward正常、其余rank耗时同量级增加，与EP等待放大相符；"
              "这是证据支持的解释，尚未定位晚进入的根因。不会据此直接给CSA加sync_start。",
              "正式均值/P95/max仍原样保留在[RESULTS.md](model/RESULTS.md)。", "",
              f"单位依据：[Ascend API]({TIMESTAMP_DOC})、[torch_npu实现]({TORCH_SOURCE})。"]
    (ROOT / "arrival.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "ARRIVAL.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
