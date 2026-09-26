"""汇集已验真的六档结果；只整理文件，不执行模型、不计算 hash。"""

import json
import os
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUTPUT = BASE.parent / "csa_six_case_profiles_20260926"
CASES = [(131072, batch) for batch in (4, 8, 16)] + [(8192, batch) for batch in (24, 32, 40)]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def include(source, target):
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"缺少有效产物：{source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return
        raise FileExistsError(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "性能版与 Native，128K B4/8/16、8K B24/32/40，TP1/DP=EP16、出5验6、EPLB关闭",
        "metric": "无 profiler；每 rank warmup 后连续 10 次 _model_forward，16 rank 等权均值",
        "excludes": "metadata准备、logits、采样、草稿、步间等待、加载及初始化编译",
        "source_commits": {"128k_model": "a7dc706e", "8k_model": "5d42db04", "single_card_dfx": "7769b3c7"},
        "toolchain": {"pypto": "88297437", "simpler": "a54c05095", "ptoas": "0.66", "pto_isa": "327cd586"},
        "weight_nz_mode": 2, "pto_variant": "performance", "atomic_add": 1,
        "native_deterministic_level": 0, "hccl_deterministic": False, "eplb": False,
        "max_num_seqs": 40, "capture_sizes": [24, 48, 96, 144, 192, 240],
        "dfx": "正式 model.layers.4 权重，合成输入/历史；复用前层 metadata 的完整 HC_pre→norm→CSA→HC_post 根，"
               "独立单卡一次调用；带 DFX 开销，不是整模型原位调用或无 profiler 计时",
        "cases": [],
    }
    files = []
    table = ["| 历史长度 | 每卡 B | Native forward ms | PTO forward ms | PTO 耗时变化 | token / DSpark |",
             "| ---: | ---: | ---: | ---: | ---: | --- |"]
    for history, batch in CASES:
        group = "model_128k_performance" if history == 131072 else "model_8k_large_batch_performance"
        source = BASE / group / "capacity40" / f"b{batch}"
        name = f"h{history}_b{batch}"
        destination = OUTPUT / name
        report = read(source / "performance_comparison.json")
        if report["status"] != "MEASURED_TOKEN_PASS" or len(report["ranks"]) != 16:
            raise ValueError(f"{name} 尚未完成全 rank 的性能与输出核验")
        for backend in ("native", "pto"):
            exports = read(source / backend / "profile_export.json")["exported"]
            if {item["rank"] for item in exports} != {f"rank{rank}" for rank in range(16)}:
                raise ValueError(f"{name}/{backend} 未覆盖 16 rank")
            for item in exports:
                rank = int(item["rank"].removeprefix("rank"))
                target = destination / (f"{backend}_rank0_pytorch.json" if rank == 0 else
                                        f"{backend}_other_ranks/rank{rank:02d}_pytorch.json")
                original = Path(item["trace_view"])
                include(original, target)
                files.append({"file": str(target.relative_to(OUTPUT)), "bytes": target.stat().st_size,
                              "source": str(original), "history": history, "batch": batch, "rank": rank})
        dfx_dir = BASE / "six_case_swimlanes" / name
        diagnostic = read(dfx_dir / "report.json")
        if not diagnostic["swimlane"]["exported"] or diagnostic["status"] != "MEASURED":
            raise ValueError(f"{name} 泳道未成功导出")
        if diagnostic["layer_index"] != 4 or diagnostic["history"] != history or diagnostic["batch"] != batch:
            raise ValueError(f"{name} 泳道配置不符")
        raw = read(dfx_dir / "dfx/chip_swimlane_records.json")
        metadata = raw["metadata"]
        if len(metadata["run_boundaries"]) != 1 or metadata["dropped_run_boundaries"] or not raw["aicore_tasks"]:
            raise ValueError(f"{name} 泳道缺少唯一完整调用")
        for original_name, target_name in (
            ("merged_swimlane.json", "pto_layer4_swimlane.json"),
            ("chip_swimlane_records.json", "dfx_source/chip_swimlane_records.json"),
            ("deps.json", "dfx_source/deps.json"),
            ("name_map.json", "dfx_source/name_map.json"),
            ("kernel_config_source.py", "dfx_source/kernel_config_source.py"),
            ("converter_output.txt", "dfx_source/converter_output.txt"),
        ):
            original, target = dfx_dir / "dfx" / original_name, destination / target_name
            include(original, target)
            files.append({"file": str(target.relative_to(OUTPUT)), "bytes": target.stat().st_size,
                          "source": str(original), "history": history, "batch": batch, "layer": 4})
        include(dfx_dir / "report.json", destination / "dfx_source/report.json")
        entry = {key: report[key] for key in (
            "status", "forward_device", "slowest_rank_forward_device", "compared_tokens",
            "token_mismatches", "spec_decode_mismatched_ranks", "csa")}
        entry.update(history=history, batch=batch, global_batch=batch * 16,
                     token_budget=256 if history == 131072 else 400,
                     scheduled_token_budget=96 if history == 131072 else 240,
                     result_source=str(source), profile_folder=name,
                     dfx_aicore_tasks=len(raw["aicore_tasks"]), per_rank=[])
        for rank in report["ranks"]:
            row = {"rank": rank["rank"]}
            for backend in ("native", "pto"):
                original = read(source / backend / f"rank{rank['rank']}.performance.json")
                row[backend] = {key: rank[backend][key] for key in (
                    "forward_device", "spec_decode", "peak_allocated_bytes", "peak_reserved_bytes")}
                row[backend]["forward_samples_us"] = original["steady_window"][0]["forward"]["samples_us"]
            entry["per_rank"].append(row)
        native, pto = (entry["forward_device"][backend]["mean_us"] / 1000 for backend in ("native", "pto"))
        entry["pto_forward_change_pct"] = (pto / native - 1) * 100
        table.append(f"| {history} | {batch} | {native:.3f} | {pto:.3f} | "
                     f"{entry['pto_forward_change_pct']:+.2f}% | 一致 / 一致 |")
        summary["cases"].append(entry)
    diagnosis = BASE / "event_mode_diagnosis"
    diagnostic_note = ""
    if (diagnosis / "comparison.json").is_file():
        destination = OUTPUT / "b4_diagnosis"
        destination.mkdir(exist_ok=True)
        for filename in ("comparison.json", "dispatch_alignment.json", "ffn_kernel_totals.json",
                         "rank0_ffn_by_layer.json", "probe_default.json", "probe_0.json"):
            original, target = diagnosis / filename, destination / filename
            include(original, target)
            files.append({"file": str(target.relative_to(OUTPUT)), "bytes": target.stat().st_size,
                          "source": str(original), "scope": "B4专项诊断"})
        exports = read(diagnosis / "native_hardware/b4/native/profile_export.json")["exported"]
        for item in exports:
            original = Path(item["trace_view"])
            target = destination / "native_hardware_ranks" / f"{item['rank']}_pytorch.json"
            include(original, target)
            files.append({"file": str(target.relative_to(OUTPUT)), "bytes": target.stat().st_size,
                          "source": str(original), "scope": "B4专项诊断，Native硬件event"})
        (destination / "README.md").write_text(
            (diagnosis / "README.md").read_text().replace(
                "native_hardware/b4/native/trace/", "native_hardware_ranks/").replace(
                "../../csa_six_case_profiles_20260926/", "../"), encoding="utf-8")
        summary["b4_diagnosis"] = {
            "report": "b4_diagnosis/README.md", "native_hardware_forward_ms": 47.95202713012695,
            "conclusion": "排除event差异为回退主因；诊断trace约75%增量在FFN/MoE，"
                          "已见跨rank到达差，GMM增量的路由/分组原因待直接证据。",
            "csa_trace_caution": "默认event模式不同造成profiling扰动差异，撤回据旧trace推断B4 CSA本体更快。",
        }
        diagnostic_note = (
            "\nB4回退专项见 [诊断报告](b4_diagnosis/README.md)：event模式差异不是主因，"
            "主要增量在FFN/MoE；附Native硬件模式的16rank trace。原默认event模式不同会影响"
            "profiling扰动，因此旧CSA区间不能直接外推无profiler的本体快慢。\n")
    save(OUTPUT / "summary.json", summary)
    save(OUTPUT / "files.json", files)
    (OUTPUT / "README.md").write_text(
        "# 六档 decode forward 对比与 profiling\n\n" + "\n".join(table) + "\n\n"
        "正值表示 PTO 较慢。每 rank 预热后连续 10 次纯 forward，再对 16 rank 等权汇总；"
        "逐 rank 原始样本、最慢 rank 指标、CSA 层区间和 DSpark 统计见 summary.json。\n\n"
        "每个 h{历史}_b{batch} 目录直接打开三个主文件：\n\n"
        "- native_rank0_pytorch.json：Native 整模型 rank0，独立 Level0 三步。\n"
        "- pto_rank0_pytorch.json：PTO 整模型 rank0，相同采集档位。\n"
        "- pto_layer4_swimlane.json：PTO 第二个 CSA 层（model.layers.4）单卡 DFX。\n\n"
        "其余 15 rank 的原始 PyTorch JSON 在 native_other_ranks/ 和 pto_other_ranks/。"
        "dfx_source/ 保留泳道原始任务、依赖与本次 JIT 的真实名称表。全部文件可直接下载后打开，"
        "不依赖服务器上的符号链接；files.json 记录来源和大小。\n\n"
        "泳道采用正式第二层权重、合成输入/历史和 Native 页式 cache/state，复用前层 compact metadata；"
        "它与整模型 trace 的输入内容、物理页号及执行模式不同，只用于观察相同形状下的任务与调度。"
        "DFX/Level0 都独立于无 profiler 计时，不能把诊断耗时填入性能主表。\n\n"
        "两侧 mode=2、TP1/DP=EP16、DSpark 出5验6、EPLB关、Native实际level0、HCCL=false；"
        "PTO性能版atomic=1。容量40、捕获24/48/96/144/192/240；128K预算256、8K预算400。"
        "本轮没有精度版性能数据，也不代替 H8192/B16 的750微秒目标或完整数值验收。\n"
        + diagnostic_note,
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
