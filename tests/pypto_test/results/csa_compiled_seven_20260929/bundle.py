"""Gather the completed matrix's original profiler JSONs for downloading."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ((131072, 4), (131072, 8), (131072, 16), (131072, 24), (8192, 24), (8192, 32), (8192, 40))


def main():
    task = (ROOT / "task.txt").read_text().strip()
    if subprocess.check_output(["task-submit", "--status", task], text=True).strip() != "completed (exit=0)":
        raise RuntimeError("Do not publish a partial matrix as a completed download")
    result = json.loads((ROOT / "evidence.json").read_text())
    if [(case["history"], case["batch"]) for case in result["cases"]] != list(CASES):
        raise ValueError("Expected the complete new seven-case matrix")
    destination = ROOT / "download"
    destination.mkdir(exist_ok=True)
    mapping = []
    for index, case in enumerate(result["cases"], 1):
        prefix = f"{index:02}_{case['history']//1024}K_B{case['batch']}"
        originals = [("01_Native_Pytorch", Path(case["sides"]["native"]["profile_json"])),
                     ("02_PTO_Pytorch", Path(case["sides"]["pto"]["profile_json"])),
                     ("03_PTO_Swimlane_Window3", Path(case["worker_windows"][3]["path"]))]
        for suffix, source in originals:
            if not source.is_file() or source.stat().st_size == 0:
                raise ValueError(f"Missing original capture: {source}")
            name = f"{prefix}_{suffix}_SyntheticHistory.json"
            shutil.copyfile(source, destination / name)
            mapping.append({"file": name, "original": str(source)})
    (destination / "sources.json").write_text(json.dumps(mapping, indent=2) + "\n")
    shutil.copyfile(ROOT / "RESULTS.md", destination / "RESULTS.md")
    (destination / "README.md").write_text(
        "# 七档JSON下载包\n\n"
        "21份原始JSON：每档Native/PTO各一份PyTorch profiler导出，另含PTO的第4个独立泳道窗口。\n"
        "窗口编号预先固定为window_3，不选择最快窗口；四个完整窗口仍保留在原目录。\n"
        "源码c93ec723，CANN9.2/mode2/det0，PTO atomic0，正式layer4权重及独立合成历史。\n"
        "范围为HC_pre+norm+CSA+HC_post，不能当作真实权重16卡模型forward或token验收。\n"
        "性能主表来自独立5预热/20次无profiler事件，不能直接用这些独立profile做相减归因。\n\n"
        "[性能结果](RESULTS.md)；[原文件映射](sources.json)。JSON只复制，未修改或拼接事件。\n"
    )
    print(f"Copied {len(mapping)} original JSON files into {destination}")


if __name__ == "__main__":
    main()
