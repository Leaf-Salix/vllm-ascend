"""汇集本轮七档的模型 trace 和单层 DFX；保留来源与不同测量范围。"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ((131072, 4), (131072, 8), (131072, 16), (8192, 16), (8192, 24), (8192, 32), (8192, 40))


def main():
    output = ROOT / "download"
    entries = []
    for history, batch in CASES:
        name = f"h{history}_b{batch}"
        for backend in ("native", "pto"):
            source = ROOT / "model" / f"h{history}" / f"b{batch}" / backend
            traces = list((source / "trace/rank0").glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json"))
            if len(traces) != 1:
                raise ValueError(f"{source}: 需要唯一的 rank0 trace，实际 {len(traces)}")
            entries.append({"file": f"{name}_{backend}_model_rank0.json", "source": str(traces[0]),
                            "scope": "真实权重 EP16 rank0 的独立三步 profile；不是正式十步无 profiler 计时"})
        report_path = ROOT / "swimlanes" / name / "report.json"
        report = json.loads(report_path.read_text())
        windows = report["swimlane_windows"]
        if len(windows) != 1 or not windows[0]["exported"]:
            raise ValueError(f"{report_path}: 需要一个已导出的 DFX 窗口")
        entries.append({"file": f"{name}_pto_single_layer_swimlane.json",
                        "source": windows[0]["merged_swimlane"], "scope": windows[0]["scope"],
                        "input_source": windows[0]["input_source"], "report": str(report_path)})

    # 先检查整套来源，缺失时不交付一套看似完整的目录。不扫描或 hash 大文件。
    for entry in entries:
        if not Path(entry["source"]).is_file():
            raise FileNotFoundError(entry["source"])
    summaries = [ROOT / "model" / name for name in (
        "RESULTS.md", "forward.json", "MODEL_GAP.md", "model_gap_rank0.json")]
    summaries += [ROOT / name for name in (
        "WORKER_GAP.md", "worker_gap.json", "NATIVE_PROJECTION.md", "native_projection.json")]
    for source in summaries:
        if not source.is_file():
            raise FileNotFoundError(source)
    output.mkdir(exist_ok=True)
    for entry in entries:
        shutil.copyfile(entry["source"], output / entry["file"])
    for source in summaries:
        shutil.copyfile(source, output / source.name)
    manifest = {"operator": "71153bb3", "atomic_add": 0, "weight_nz_mode": 2,
                "deterministic_level": 0, "hccl_deterministic": False, "eplb": False,
                "cache": "原 Native 物理页，PTO 内部直接读取", "files": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text(
        "# 71153bb3 / atomic0 七档 profile\n\n"
        "14 份真实 EP16 模型 rank0 PyTorch JSON，加 7 份单卡第二个 CSA 层的 PTO DFX JSON。\n"
        "双方 TP1、DP=EP16、出5验6、mode2、det0、EPLB关；完整来源见 manifest.json。\n\n"
        "正式性能以 RESULTS.md 中预热后十步无 profiler forward 为准，全部 rank 样本见 forward.json。\n"
        "模型 JSON 是独立三步 profile，其他十五个 rank 的原始数据保留在来源目录。\n"
        "PTO DFX 使用第4层真实权重和人工历史，metadata reuse、graph replay，一次根调用；"
        "它与模型输入和计时轮次不同，不能直接拿 DFX 首尾代替正式整网成绩。\n",
        encoding="utf-8",
    )
    print(f"{len(entries)} profiles: {output}")


if __name__ == "__main__":
    main()
