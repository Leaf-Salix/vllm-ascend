"""汇集完整七档的原始 JSON，明确区分正式比较与核内诊断。"""

import json
import shutil
import subprocess
from pathlib import Path

from collect import CASES, ROOT


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"尚未完成七档，不发布完整下载包：{status}")
    result = json.loads((ROOT / "evidence.json").read_text())
    if (result["task"], result["task_status"]) != (task, status):
        raise ValueError("收集结果不是本轮已完成的矩阵")
    if [(c["history"], c["batch"]) for c in result["cases"]] != list(CASES):
        raise ValueError("下载包需要完整的当前七档")
    mapping = []
    for index, case in enumerate(result["cases"], 1):
        prefix = f"{index:02}_{case['history']//1024}K_B{case['batch']}"
        captures = (
            ("01_Native_SuperKernelOn_Pytorch", case["sides"]["native"]["profile_json"], "主性能独立profile"),
            ("02_PTO_Pytorch", case["sides"]["pto"]["profile_json"], "主性能独立profile"),
            ("03_PTO_Swimlane_Window3", case["worker_windows"][3]["path"], "PTO核内与调度"),
            ("04_Native_SuperKernelOff_StaticOn_Incore", case["native_incore"]["profile_json"], "仅核内诊断"),
        )
        for suffix, original, scope in captures:
            source = Path(original)
            if not source.is_file() or not source.stat().st_size:
                raise ValueError(f"原始profile缺失：{source}")
            mapping.append({"file": f"{prefix}_{suffix}_SyntheticHistory.json",
                            "original": str(source), "scope": scope})
    destination = ROOT / "download"
    destination.mkdir(exist_ok=True)
    for item in mapping:
        shutil.copyfile(item["original"], destination / item["file"])
    (destination / "sources.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n")
    for name in ("RESULTS.md", "evidence.json", "summary.json"):
        shutil.copyfile(ROOT / name, destination / name)
    commit = result["source"]["operator_commit"]
    (destination / "README.md").write_text(
        "# 当前七档profiling下载包\n\n"
        "28份原始JSON：每档Native主性能、PTO主性能、PTO泳道及Native核内诊断各一份。\n"
        "01：Native SuperKernel开、static compile开；04：SuperKernel关、static compile开，仅分析核内。\n"
        "两套Native均显式torch.compile backend=npugraph_ex、dynamic=False、fullgraph=True、inplace_pass=True。\n"
        "03：预先固定第4个DFX窗口（window_3），不挑最快窗口；全部四窗仍在原目录。\n"
        f"生产算子{commit}，CANN9.2/mode2/det0，PTO atomic0，layer4真实权重及独立合成历史。\n"
        "范围为HC_pre+norm+CSA+HC_post；不是16卡模型forward或token/DSpark验收。\n"
        "主性能表来自5预热/20次无profiler设备事件，独立profile不能替代正式样本。\n"
        "Native主性能中的SuperKernel包含多个算子，不可标成单个QLI或Sparse核时。\n\n"
        "[性能及核内对照](RESULTS.md)、[原文件映射](sources.json)、[精简证据](summary.json)。\n"
        "JSON直接复制，未修改或拼接事件，未把诊断组计时混入主性能表。\n"
    )
    print(f"Copied {len(mapping)} original JSON files into {destination}")


if __name__ == "__main__":
    main()
