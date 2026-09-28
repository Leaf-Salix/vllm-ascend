"""只读本轮 EP16 与 DFX，分别呈现完整区间、核内和调度差距。"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "30f2b228"
CASES = ((131072, 4), (131072, 8), (131072, 16), (8192, 16), (8192, 24), (8192, 32), (8192, 40))


def module(name, source):
    spec = importlib.util.spec_from_file_location(name, ROOT.parent / source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def model_report(*, matrix_label="统一七档"):
    data = json.loads((ROOT / "model/model_gap_rank0.json").read_text())
    lines = [f"# {REVISION}：{matrix_label}模型内区间", "", data["scope"], "", data["limits"], "",
             "正式验收见[无profiler十步forward](RESULTS.md)。以下为独立三步rank0 profile，不能精确分账正式计时。",
             "每档63个完整CSA区间（3步×21层），包含首次metadata。", "",
             "| 档位 | Native/PTO CSA均值 μs | PTO变化 | Native/PTO P50 μs | Native/PTO P95 μs |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in data["cases"]:
        n, p = (row[side]["csa"] for side in ("native", "pto"))
        if n["samples"] != 63 or p["samples"] != 63:
            raise ValueError("CSA样本不足")
        lines.append(f"| {row['history']//1024}K/B{row['batch']} | {n['mean_us']:.2f}/{p['mean_us']:.2f} | "
                     f"{(p['mean_us']/n['mean_us']-1)*100:+.2f}% | {n['p50_us']:.2f}/{p['p50_us']:.2f} | "
                     f"{n['p95_us']:.2f}/{p['p95_us']:.2f} |")
    lines += ["", "下表为每步均值ms。CSA本体不含首次metadata，其被计入区间外；四个互斥分段可相加为主图区间。",
              "完整CSA与区间外不能直接相加。专家GMM包含于FFN，其任务时间可能重叠。", "",
              "| 档位 | 侧 | CSA本体 | 其他Attention | FFN | 区间外 | 主图区间 | 专家GMM任务合计 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in data["cases"]:
        for side in ("native", "pto"):
            value = row[side]
            totals = [value["phases"][key] / 1000 for key in (
                "c4_body_sum_us", "other_attention_sum_us", "ffn_sum_us", "outside_half_intervals_us", "main_graph_us")]
            if abs(sum(totals[:4]) - totals[4]) > .001:
                raise ValueError("分段未覆盖主图区间")
            totals.append(sum(v for k, v in value["ffn_task_busy_us_per_step"].items() if "GroupedMatmul" in k) / 1000)
            lines.append(f"| {row['history']//1024}K/B{row['batch']} | {side} | " +
                         " | ".join(f"{v:.3f}" for v in totals) + " |")
    lines += ["", "不同event模式会影响profile；未采实际专家索引，不将GMM时间变化单独归因于路由或atomic。",
              "主图区间不是完整_model_forward事件边界；rank0局部区间不能代替EP16最慢rank验收。"]
    (ROOT / "model/MODEL_GAP.md").write_text("\n".join(lines) + "\n")


def worker_report():
    worker = module("worker", "csa_scheduling_20260927/upstream_725/compare.py")
    upstream = worker.summarize(worker.UPSTREAM)
    current = []
    for history, batch in CASES:
        path = ROOT / "swimlanes" / f"h{history}_b{batch}" / "report.json"
        report = json.loads(path.read_text())
        if len(report["swimlane_windows"]) != 1:
            raise ValueError(f"{path}: 本轮要求一个窗口")
        current.append({"history": history, "batch": batch,
                        **worker.summarize(Path(report["swimlane_windows"][0]["merged_swimlane"]))})
    value = {"operator": REVISION, "source_reference": "pypto-lib main2164563",
             "scope": "layer4真实权重、合成历史，mode2/atomic0/det0，graph replay，每档一个DFX窗口。",
             "limits": "历史727.98μs上游图缺完整配置，非同输入A/B；核内含DMA/等待，setup不是纯调度开销。",
             "upstream": upstream, "current": current}
    write_json(ROOT / "worker_gap.json", value)
    lines = [f"# {REVISION}：七档DFX与上游参考", "", value["scope"], "", value["limits"], "",
             "单位μs，单窗口不替代无profiler均值/P95或整模型验收。", "",
             "| 档位 | Worker首尾 | QR核内均值 | KV核内均值 | Indexer query核内均值 | query启动分散 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in current:
        t = row["tasks"]
        values = [row["worker_span_us"], t["qr_proj_matmul"]["kernel_mean_us"],
                  t["kv_proj_matmul"]["kernel_mean_us"], t["idx_qr_proj_matmul"]["kernel_mean_us"],
                  t["idx_qr_proj_matmul"]["start_spread_us"]]
        lines.append(f"| {row['history']//1024}K/B{row['batch']} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    short = next(c for c in current if c["history"] == 8192 and c["batch"] == 16)
    lines += ["", "8K/B16与历史上游的互斥区间：", "", "| 分段 μs | 上游 | 当前 |", "| --- | ---: | ---: |"]
    for key in upstream["phases_us"]:
        lines.append(f"| {key} | {upstream['phases_us'][key]:.2f} | {short['phases_us'][key]:.2f} |")
    lines += ["", "| Task | 上游/当前数量 | 核内均值 μs | 启动分散 μs | setup均值 μs |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name in dict.fromkeys([*upstream["tasks"], *short["tasks"]]):
        pair = (upstream["tasks"].get(name), short["tasks"].get(name))
        values = ["/".join(f"{t[key]:.2f}" if t else "—" for t in pair)
                  for key in ("blocks", "kernel_mean_us", "start_spread_us", "setup_mean_us")]
        lines.append("| " + " | ".join([name, *values]) + " |")
    lines += ["", "与pypto-lib不同的原因：上游QA/KV split-K2/4并使用atomic与清零种子；当前固定K、免种子、"
              "按行数选择M32/M64和最多3个M组。该方向以真实EP16结果验收，核内缩短并不自动代表整网收益。",
              "原生物理cache页、页表/slot、Native metadata与状态接口均保留，因此仍包含上游连续cache用例没有的工作。",
              "本轮未包含独立的Indexer query sync_start候选；其结果不能混入本表。"]
    (ROOT / "WORKER_GAP.md").write_text("\n".join(lines) + "\n")
    projection = module("projection", "csa_atomic_matrix_20260928/native_projection.py")
    projection.ROOT = ROOT
    projection.main()


def bundle():
    output = ROOT / "download"
    entries = []
    for history, batch in CASES:
        name = f"h{history}_b{batch}"
        for side in ("native", "pto"):
            traces = list((ROOT / "model" / f"h{history}" / f"b{batch}" / side / "trace/rank0").glob(
                "*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json"))
            if len(traces) != 1:
                raise ValueError(f"{name}/{side}: 缺少唯一trace")
            entries.append({"file": f"{name}_{side}_model_rank0.json", "source": str(traces[0]),
                            "scope": "真实EP16 rank0独立3步PyTorch profile，非正式10步计时"})
        report = json.loads((ROOT / "swimlanes" / name / "report.json").read_text())
        w = report["swimlane_windows"][0]
        entries.append({"file": f"{name}_pto_single_layer_swimlane.json", "source": w["merged_swimlane"],
                        "scope": w["scope"], "input_source": w["input_source"]})
    summaries = [ROOT / "model" / name for name in (
        "RESULTS.md", "forward.json", "MODEL_GAP.md", "model_gap_rank0.json")]
    summaries += [ROOT / name for name in (
        "WORKER_GAP.md", "worker_gap.json", "NATIVE_PROJECTION.md", "native_projection.json",
        "ARRIVAL.md", "arrival.json")]
    for path in [Path(e["source"]) for e in entries] + summaries:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(exist_ok=True)
    for entry in entries:
        shutil.copyfile(entry["source"], output / entry["file"])
    for path in summaries:
        shutil.copyfile(path, output / path.name)
    write_json(output / "manifest.json", {"operator": REVISION, "atomic_add": 0, "weight_nz_mode": 2,
                                          "deterministic_level": 0, "eplb": False,
                                          "actual_cann_event_work_mode": {"native": 0, "pto": 1}, "files": entries})
    (output / "README.md").write_text(f"# {REVISION} 七档对照\n\n"
        "14份真实EP16模型rank0 PyTorch JSON、7份单卡第二CSA层PTO泳道。来源与范围见manifest.json。\n"
        "TP1、DP=EP16、出5验6、mode2、det0、EPLB关，PTO原始cache物理页/atomic0。\n"
        "正式结果用RESULTS.md的预热后10步forward；模型profile独立3步，单层DFX使用真实权重和合成历史。\n"
        "七档均值更快，但128K/B8、B16的P95较高；ARRIVAL.md保留异常，尾部问题未关闭。\n")
    print(f"{len(entries)} profiles: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("model", "worker", "bundle"))
    phase = parser.parse_args().phase
    {"model": model_report, "worker": worker_report, "bundle": bundle}[phase]()
