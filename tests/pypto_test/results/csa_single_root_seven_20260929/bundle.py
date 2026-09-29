"""将七档固定窗口的原始PTO泳道汇集并打包，保留来源，不改事件。"""

import csv
import json
import shutil
import zipfile
from pathlib import Path

from collect import CASES, ROOT, finished


def main():
    task, status = finished(ROOT)
    result = json.loads((ROOT / "evidence.json").read_text())
    if (result["task"], result["task_status"]) != (task, status):
        raise ValueError("证据不属于当前已完成任务")
    if [(c["history"], c["batch"]) for c in result["cases"]] != list(CASES):
        raise ValueError("需要完整七档才能发布下载包")
    commit = result["source"]["operator_commit"]
    window = result["source"]["swimlane_window"]
    mapping = []
    for number, case in enumerate(result["cases"], 1):
        source = Path(case["worker_windows"][window]["path"])
        if source.parent.name != f"window_{window}" or not source.is_file() or not source.stat().st_size:
            raise ValueError(f"缺少预先固定的原始窗口：{source}")
        name = (f"{number:02}_{case['history']//1024}K_B{case['batch']}_PTO_Swimlane_{commit}_"
                "SingleCSA_SyntheticHistory.json")
        mapping.append({"file": name, "original": str(source), "task": case["pto_task"],
                        "reused": case["pto_reused"], "window": window,
                        "pytorch_profile": case["sides"]["pto"]["profile_json"]})
    destination = ROOT / "download_pto_swimlanes"
    destination.mkdir(exist_ok=True)
    expected = {item["file"] for item in mapping}
    if {p.name for p in destination.glob("*.json")} - expected:
        raise ValueError("下载目录存在其他版本JSON，不能混入本轮七档")
    for item in mapping:
        shutil.copyfile(item["original"], destination / item["file"])
    with (destination / "SOURCES.tsv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(mapping[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(mapping)
    lines = ["# 本轮七档PTO单次CSA泳道", "",
             f"性能版算子提交：{commit}；七档使用同一冻结私有包。",
             "CANN9.2、NZ mode2、deterministic=0、atomic_add=0，单卡layer4（第二个CSA层）。",
             "真实层权重及合成历史KV；每份JSON含一次HC_pre+norm+CSA+HC_post图重放。",
             f"统一取第{window + 1}个DFX窗口（window_{window}），直接复制原始merged JSON，不改事件、不挑最快窗口。",
             "文件可用Perfetto或Chrome tracing打开；原始四窗与PyTorch profile路径见SOURCES.tsv。", "",
             "| 文件 | 档位 | 来源 |", "| --- | --- | --- |"]
    for case, item in zip(result["cases"], mapping):
        provenance = "复用本轮同源码两档A/B的候选" if case["pto_reused"] else "本轮补测"
        lines.append(f"| [{item['file']}]({item['file']}) | {case['history']//1024}K/B{case['batch']} "
                     f"| {provenance} |")
    lines += ["", "## 独立正式计时参考", "",
              "单位μs，5次预热后20次设备事件；与DFX独立采集，泳道开销不能当成正式CSA性能。",
              "Native复用CANN9.2的最新标准基线：npugraph_ex、dynamic=False、inplace_pass/static/SuperKernel开启。",
              "两侧采样来自不同任务，不用于归因单项优化收益；不是16卡模型forward或token/DSpark验收。", "",
              "| 档位 | Native均值 | PTO均值 | PTO耗时变化 | PTO P95 |", "| --- | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        native, pto = (case["sides"][side] for side in ("native", "pto"))
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {native['mean_us']:.2f} "
                     f"| {pto['mean_us']:.2f} | {case['csa_change_pct']:+.2f}% | {pto['us_p95']:.2f} |")
    (destination / "README.md").write_text("\n".join(lines) + "\n")
    archive = ROOT / f"PTO_CSA_7cases_{commit}_20260929.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name in [item["file"] for item in mapping] + ["README.md", "SOURCES.tsv"]:
            output.write(destination / name, arcname=f"{destination.name}/{name}")
    print(f"七份原始PTO泳道：{destination}\n下载压缩包：{archive}")


if __name__ == "__main__":
    main()
