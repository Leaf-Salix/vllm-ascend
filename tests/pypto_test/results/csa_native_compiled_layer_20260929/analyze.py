"""Summarize completed Native compile comparisons without rerunning NPU work."""
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def analyze_case(path):
    report = json.loads((path / "report.json").read_text())
    if report["status"] != "MEASURED":
        return {"status": report["status"], "error": report.get("error")}
    result = {key: report[key] for key in ("history", "batch", "compiler", "means_us", "change_pct")}
    result["report"] = str(path.relative_to(ROOT) / "report.json")
    result["sides"] = {}
    for side, timing in report["timing"].items():
        csv_path = next((path / "profile" / side).rglob("kernel_details.csv"))
        with csv_path.open() as stream:
            kernels = list(csv.DictReader(stream))
        types = collections.defaultdict(list)
        for kernel in kernels:
            types[kernel["Type"]].append(float(kernel["Duration(us)"]))
        result["sides"][side] = {
            "mean_us": report["means_us"][side],
            **{key: timing[key] for key in ("us_min", "us_p50", "us_p95", "us_max")},
            "guards": dict(collections.Counter(v["status"] for v in timing["guards"].values())),
            "states": {
                name: {key: value.get(key) for key in (
                    "status", "elements", "mismatches", "nonfinite", "max_abs", "rmse")}
                for name, value in timing["eager_comparison"].items()
            },
            "topk": timing["topk_selection"],
            "profile_kernel_count": len(kernels),
            "profile_kernel_span_us": (
                max(float(k["Start Time(us)"]) + float(k["Duration(us)"]) for k in kernels)
                - min(float(k["Start Time(us)"]) for k in kernels)
            ),
            "profile_types_us": dict(types),
            "profile_csv": str(csv_path.relative_to(ROOT)),
            "profile_json": str(next(csv_path.parent.glob("trace_view.json")).relative_to(ROOT)),
        }
    descriptors = list((path / "static_kernel_compile_outputs").glob("*/*_opcompile/*.json"))
    result["static_descriptor_types"] = dict(collections.Counter(
        json.loads(p.read_text())["op_type"] for p in descriptors))
    result["static_descriptors"] = [str(p.relative_to(ROOT)) for p in sorted(descriptors)]
    success_logs = list((path / "static_kernel_compile_outputs").glob("*/*_compile_log/*_compile_succ.log"))
    result["static_success_logs"] = [str(p.relative_to(ROOT)) for p in sorted(success_logs)]
    package = next((path / "static_kernel_compile_outputs").glob("*/*.run"))
    manifest = (Path(report["opp"]) / "static_kernel/ai_core" / package.stem
                / "config/ascend910_93/binary_info_config.json")
    installed = json.loads(manifest.read_text())
    result["installed_manifest"] = str(manifest)
    result["installed_binary_counts"] = {name: len(value["staticList"]) for name, value in installed.items()}
    result["descriptor_types_without_static_binary"] = sorted(set(result["static_descriptor_types"]) - set(installed))
    return result


def main():
    cases = {path.name: analyze_case(path) for path in sorted(ROOT.glob("h*_b*"))
             if (path / "report.json").exists()}
    (ROOT / "evidence.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Native 模板编译：attention 半层对照", "",
        "同进程、同 fixture、正式 layer4 权重、CANN9.2/mode2/det0；",
        "5 次预热后 20 次图重放，状态恢复在计时外。profile 独立采集。",
        "这里是 HC_pre + norm + CSA + HC_post，不包含 MoE/EP16。", "",
        "| 档位 | 手工图 μs | 模板编译 μs | 变化 | P95 μs | 最大值 μs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases.values():
        if "sides" not in case:
            continue
        manual, compiled = (case["sides"][side] for side in ("manual", "compiled"))
        lines.append(
            f"| {case['history'] // 1024}K/B{case['batch']} | {manual['mean_us']:.3f} "
            f"| {compiled['mean_us']:.3f} | {case['change_pct']:+.3f}% "
            f"| {manual['us_p95']:.3f}→{compiled['us_p95']:.3f} "
            f"| {manual['us_max']:.3f}→{compiled['us_max']:.3f} |")
    lines += ["", "短档执行前私有 OPP 已保留长档静态包，手工控制可能复用匹配形状的已有二进制。",
              "短档百分比是这次手工调用→编译半层的增量变化，不是完全隔离静态包有/无的收益归因。",
              "编译后绝对耗时作为当前单卡 Native 配置基线；尚未取得同配置 PTO 对照。",
              "", "## 数值与范围", ""]
    for case in cases.values():
        if "sides" not in case:
            continue
        compiled = case["sides"]["compiled"]
        output = compiled["states"]["x_out"]
        selection = compiled["topk"]
        lines += [
            f"{case['history'] // 1024}K/B{case['batch']}：编译后输出相对 eager 有 "
            f"{output['mismatches']}/{output['elements']} 个非逐 bit 相等元素，"
            f"max_abs={output['max_abs']}，RMSE={output['rmse']}。",
            f"Top-K 有 {selection['position_mismatches']} 个位置不同，"
            f"仅顺序变化 {selection['order_only_rows']} 行，集合不同 {selection['different_set_rows']} 行；"
            f"非法行 {selection['invalid_rows']}。",
            "此配置 det0，尚未区分编译算术变化与执行顺序波动；不将差异默认归因于某个融合或确定性。", "",
        ]
    lines += ["实际编译返回、产物覆盖、状态差异和 profile 路径见 [evidence.json](evidence.json)。",
              "旧手工图结果保留其原始配置含义，不能直接冒充新模板基线。",
              "这只证明单卡编译半层可运行并给出性能诊断，不代表整模型 token/DSpark 验收。", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
