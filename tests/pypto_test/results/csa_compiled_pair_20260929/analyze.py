"""Read terminal paired compile measurements; perform no device work."""
import collections
import csv
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = ((131072, 16), (8192, 24))


def read(path):
    return json.loads(path.read_text())


def analyze_side(path):
    report = read(path / "report.json")
    if report["status"] != "MEASURED" or not report["compiler"]["wrapper_compiled"]:
        raise ValueError(f"Incomplete compiled result: {path}")
    timing = report["timing"]
    samples = timing["samples_us"]
    if len(samples) != 20 or len(set(timing["start_timestamps_raw"])) != 20:
        raise ValueError("Expected 20 independent graph event samples")
    if any(v["status"] != "PASS" for v in timing["guards"].values()):
        raise ValueError("Guard check failed")
    if report["side"] == "pto" and any(
            v["status"] != "PASS" for v in timing["eager_comparison"].values()):
        raise ValueError("PTO compiled replay differs from its eager reference")
    csv_path = next((path / "profile").rglob("kernel_details.csv"))
    with csv_path.open() as stream:
        kernels = list(csv.DictReader(stream))
    types = collections.defaultdict(list)
    for row in kernels:
        types[row["Type"]].append(float(row["Duration(us)"]))
    manifests = list((Path(report["opp"]) / "static_kernel").glob("**/binary_info_config.json"))
    installed = {str(p): {name: len(value["staticList"]) for name, value in read(p).items()}
                 for p in manifests}
    if report["side"] == "native" and not installed:
        raise ValueError("Native static package was not installed")
    descriptors = list((path / "static_kernel_compile_outputs").glob("*/*_opcompile/*.json"))
    return {
        "source": str(path / "report.json"), "device": report["device"],
        "cann": report["cann"], "requested": report["requested"], "compiler": report["compiler"],
        "mean_us": statistics.mean(samples),
        **{key: timing[key] for key in ("us_min", "us_p50", "us_p95", "us_max")},
        "states": timing["eager_comparison"], "topk": timing["topk_selection"],
        "guards": dict(collections.Counter(v["status"] for v in timing["guards"].values())),
        "profile_kernel_count": len(kernels), "profile_types_us": dict(types),
        "profile_csv": str(csv_path), "profile_json": str(csv_path.with_name("trace_view.json")),
        "static_descriptor_types": dict(collections.Counter(read(p)["op_type"] for p in descriptors)),
        "installed_static_binaries": installed,
    }


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"Device task not successfully terminal: {status}")
    result = {"task": task, "operator": "d8627207", "cases": []}
    for history, batch in CASES:
        folder = ROOT / f"h{history}_b{batch}"
        sides = {s: analyze_side(folder / s) for s in ("native", "pto")}
        before, after = (sides[s] for s in ("native", "pto"))
        for field in ("device", "cann", "requested"):
            if before[field] != after[field]:
                raise ValueError(f"Paired configuration differs: {field}")
        result["cases"].append({"history": history, "batch": batch, "sides": sides,
                                "change_pct": 100 * (after["mean_us"] / before["mean_us"] - 1)})
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 同配置编译后的 Native / PTO attention 半层", "",
             "d8627207，CANN9.2、mode2、det0；同一卡，正式 layer4 权重 / 合成历史。",
             "5次预热后的20次设备事件计时，profile独立采集。单位μs。", "",
             "| 档位 | Native | PTO | PTO变化 | Native/PTO P95 | Native/PTO max |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in result["cases"]:
        n, p = (case["sides"][s] for s in ("native", "pto"))
        lines.append(f"| {case['history']//1024}K/B{case['batch']} | {n['mean_us']:.3f} "
                     f"| {p['mean_us']:.3f} | {case['change_pct']:+.3f}% "
                     f"| {n['us_p95']:.3f}/{p['us_p95']:.3f} | {n['us_max']:.3f}/{p['us_max']:.3f} |")
    lines += ["", "两侧开启同一套编译配置；Native追踪并静态编译内部算子，PTO保留生产自定义算子边界，",
              "内部由PyPTO编译；实际CANN描述符、静态包和PTO调用次数分别记录。",
              "PTO复用第二层已有的compact metadata，Native保留自己的metadata调用。",
              "检查PTO自身eager/图重放、Top-K合法性和保护区；未做两侧逐元素精度或模型token/DSpark验收。",
              "范围含HC_pre + norm + CSA + HC_post，不含MoE或EP16，不等于整模型forward。", "",
              "**长档P95尚未通过**：PTO有3次1249.74–1455.72μs的拖尾，P95高于Native；",
              "均值优势不能抵消此问题，异常样本全部保留，单独追加长档profile定位。", "",
              "[完整证据与profile路径](evidence.json)"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
