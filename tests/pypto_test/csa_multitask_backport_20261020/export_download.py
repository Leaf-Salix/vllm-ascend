# SPDX-License-Identifier: Apache-2.0
"""按固定第一窗汇集正式多task七档L1/L4，每个文件一次完整CSA执行。"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = "CSA_MultiTask_7Cases_L1_L4_20261021"
MATRIX = HERE / "MATRIX_cooperative_policy_v1_matrix_v1.json"
FORMAL = "plan_workers_v1"
L4_ATTEMPT = "download_l4_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(MATRIX.read_text())
    sys.path.insert(0, matrix["toolchain_identity"]["roots"]["simpler"])
    from simpler_setup.tools.swimlane_converter import read_perf_data

    destination = HERE / "download" / PACKAGE
    destination.mkdir(parents=True, exist_ok=True)
    # 用户改为每档每级一次执行；仅清理本下载包此前生成的多窗口文件。
    for obsolete in destination.glob("*_L[14]_Window0[123].json"):
        obsolete.unlink()
    manifest = {
        "formal_operator_commit": "cc07f727",
        "formal_package": FORMAL,
        "source_matrix": str(MATRIX),
        "scope": "HC_pre + input RMSNorm + CSA + HC_post；真实权重、合成hidden/历史，TP1、layer4、NZ2、S6、CANN9.2",
        "selection": "L1固定取原五窗中的第一窗；L4独立预热后采第一窗。每个文件一次完整CSA执行",
        "toolchain_identity": matrix["toolchain_identity"],
        "l4_warning": "L4增加Scheduler和Orchestrator插桩；不拿L4时长替代L1基线或计算优化收益",
        "files": [],
        "pending_l4": [],
    }

    def copy_window(window, case, level, number):
        source, raw = Path(window["trace"]), Path(window["raw"])
        records = read_perf_data(raw)
        if records["chip_swimlane_level"] != level:
            raise ValueError(f"泳道级别不符：{raw}")
        trace = json.loads(source.read_text())
        if not trace.get("traceEvents"):
            raise ValueError(f"泳道为空：{source}")
        name = f"{case}_L{level}_SingleCSA.json"
        shutil.copy2(source, destination / name)
        manifest["files"].append(
            {
                "file": name,
                "level": level,
                "window": number,
                "source_trace": str(source),
                "source_raw": str(raw),
                "bytes": source.stat().st_size,
                "aicore_records": len(records["tasks"]),
            }
        )

    for index, row in enumerate(matrix["rows"], 1):
        history, batch = row["history"], row["batch"]
        case = f"{index:02d}_{'128K' if history == 131072 else '8K'}_B{batch:02d}"
        if row["current"] != FORMAL:
            raise ValueError("七档中的正式版本不一致")
        result = json.loads((HERE / row["result"]).read_text())
        windows = [window for window in result["l1"]["windows"] if window["side"] == "multitask_current"]
        if not result["passed"] or len(windows) != 5:
            raise ValueError("正式L1原始五窗不完整")
        for number, window in enumerate(windows[:1], 1):
            copy_window(window, case, 1, number)
        l4_result = HERE / "runs" / FORMAL / f"h{history}_b{batch}_{L4_ATTEMPT}" / "result.json"
        l4 = json.loads(l4_result.read_text()) if l4_result.exists() else {}
        if l4.get("passed") and l4.get("stage") == "complete":
            if l4.get("candidate") != FORMAL or l4["environment"] != matrix["environment"]:
                raise ValueError("L4版本或环境与所选正式L1不一致")
            l4_windows = l4["diagnostic_swimlane"]["windows"]
            if not l4_windows:
                raise ValueError("L4没有完整执行窗口")
            for number, window in enumerate(l4_windows[:1], 1):
                copy_window(window, case, 4, number)
        else:
            manifest["pending_l4"].append({"case": case, "expected_result": str(l4_result)})
    manifest["complete"] = not manifest["pending_l4"]
    manifest["trace_count"] = len(manifest["files"])
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    status = (
        "已齐全：14个泳道文件。"
        if manifest["complete"]
        else (
            f"当前已收集{manifest['trace_count']}个泳道文件，"
            f"仍有{len(manifest['pending_l4'])}档L4等待补采；本包尚未齐全。"
        )
    )
    (destination / "README.md").write_text(
        "# CSA正式多task七档：L1与L4\n\n"
        + status
        + "\n\n正式算子：`cc07f727 / plan_workers_v1`。128K的B4/B8/B16/B24/B28，"
        "8K的B16/B32；HC_pre＋input RMSNorm＋CSA＋HC_post，单卡TP1、"
        "真实W8A8权重layer4、NZ2、S6、CANN9.2；hidden与历史数据为单层合成用例，非整网forward。\n\n"
        "- `01_128K_B04_L1_SingleCSA.json`：第1档、L1、一次完整CSA执行。\n"
        "- `01_128K_B04_L4_SingleCSA.json`：同版本第1档的L4单次CSA调度诊断。\n"
        "- 每档每级一个文件。L1固定取原五窗的第一窗，没有挑最快值；"
        "L4另行采集，二者不是同一次执行，也未拼接时间线。\n"
        "- L1展示AICore时间；L4还展示Scheduler派发/完成、调度阶段与"
        "Orchestrator编排。L4有额外插桩，耗时不能代替L1性能口径。\n"
        "- JSON可直接载入Perfetto。原始路径、级别、版本及记录数见`manifest.json`。\n"
        "- 本包为当前已完成七档的正式多task，不含尚未采用的Top-K或WS候选。\n",
        encoding="utf-8",
    )
    if args.require_complete and not manifest["complete"]:
        raise RuntimeError(f"还有{len(manifest['pending_l4'])}档L4未完成")
    suffix = "" if manifest["complete"] else "_PARTIAL"
    archive = destination.parent / f"{PACKAGE}{suffix}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as packed:
        for file in sorted(destination.iterdir()):
            if file.is_file():
                packed.write(file, f"{PACKAGE}/{file.name}")
    if manifest["complete"]:
        partial = destination.parent / f"{PACKAGE}_PARTIAL.zip"
        partial.unlink(missing_ok=True)
    print(destination)
    print(archive)
    print(status)


if __name__ == "__main__":
    main()
