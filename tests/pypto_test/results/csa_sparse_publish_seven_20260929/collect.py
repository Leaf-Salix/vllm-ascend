"""收集新 Native 后端图与融合 Sparse 的七档，保留真实融合范围。"""

import collections
import csv
import functools
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
CASES = ((131072, 4), (131072, 8), (131072, 16), (131072, 24), (8192, 16), (8192, 24), (8192, 32))
STATES = {"x_out", "idx_topk", "swa.0", "compressed.0", "state.0",
          "indexer.0", "indexer.1", "indexer_state.0"}
TASKS = ("indexer_head_coefficients", "indexer_score_topk_native_pair_aic",
         "indexer_score_topk_native_pair_aiv", "indexer_topk_query_merge", "qk_pv_aic", "qk_pv_aiv")


def read(path):
    return json.loads(path.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_scope(report, csv_path, *, super_kernel=True):
    """Require the adopted graph path; do not relabel fused PMU as one operator."""
    options, measurement = report["backend_options"], report["measurement"]
    if (report["compile_entry"] != "torch.compile(backend=npugraph_ex)"
            or options["force_eager"] or not options["static_kernel_compile"]
            or options["super_kernel_optimize"] != super_kernel or report["super_kernel"] != super_kernel):
        raise ValueError("Native未使用已校准的显式后端/static/superkernel配置")
    flags = report["compiler"]["static_super_flags"]
    calls = report["super_kernel_graph_calls"]
    if (not flags or any(v != super_kernel for v in flags) or bool(calls) != super_kernel
            or any(c["status"] != "RETURNED_SUCCESSFULLY" for c in calls)):
        raise ValueError("Native超级核优化未实际生效")
    if (measurement["graph_count"] != 1 or measurement["host_updated_nodes"] != 0
            or measurement["timed_call"] != "backend-created NPUGraph.replay"):
        raise ValueError("Native直接重放的固定地址边界发生改变")
    checks = report["timing"]["eager_comparison"]
    if set(checks) != STATES or any(v["status"] != "PASS" for v in checks.values()):
        raise ValueError("Native图重放与同初态compiled callable不一致")
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    streams = collections.Counter(row["Stream ID"] for row in rows)
    if (not report["multistream"]["dsa_overlap"] or len(streams) < 2
            or any(row["Type"] == "SuperKernel" for row in rows) != super_kernel):
        raise ValueError("Native profile缺少实际多流或超级核证据")
    fields = ("Name", "Type", "Duration(us)", "aicore_time(us)", "aiv_time(us)", "Block Num", "Mix Block Num")
    kernels = [{key: row[key] for key in fields} for row in rows]
    attention = []
    for row in kernels:
        markers = [name for name in ("VllmQuantLightningIndexer", "SparseAttnSharedkv")
                   if name in row["Name"] or row["Type"] == name]
        if markers:
            attention.append({**row, "visible_endpoints": markers,
                              "scope": "完整SuperKernel；端点不代表完整内部算子清单"
                              if row["Type"] == "SuperKernel" else "独立算子"})
    if set(name for row in attention for name in row["visible_endpoints"]) != {
            "VllmQuantLightningIndexer", "SparseAttnSharedkv"}:
        raise ValueError("请检查实际融合范围，不能用缺失的Native单算子PMU填表")
    return {"attention": attention, "kernels": kernels, "streams": dict(streams),
            "backend_options": options, "measurement": measurement,
            "profile_csv": str(csv_path), "profile_json": str(csv_path.with_name("trace_view.json"))}


def main():
    task = (ROOT / "task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"等待同一任务终态：{status}")
    source = read(ROOT / "source.json")
    pair = load("sparse_seven_pair", ROOT.parent / "csa_compiled_pair_20260929/analyze.py")
    metrics = load("sparse_seven_schedule", ROOT.parent / "csa_score_segment_ub_20260929/collect.py")
    worker = load("sparse_seven_worker", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    worker.summarize = functools.partial(worker.summarize, publication_task="qk_pv_aiv")
    helper = load("sparse_seven_critical_path", WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    result = {"task": task, "task_status": status, "source": source, "cases": [],
              "scope": "单卡HC_pre+norm+CSA+HC_post；独立profile与四窗DFX，不等同模型/EP16验收"}
    for history, batch in CASES:
        folder = ROOT / f"h{history}_b{batch}"
        reports = {s: read(folder / s / "report.json") for s in ("native", "pto")}
        sides = {s: pair.analyze_side(folder / s) for s in reports}
        for side, report in reports.items():
            if (report["source"], report["variant"], report["history"], report["batch"], report["side"]) != (
                    source["source"], source["variant"], history, batch, side):
                raise ValueError("源码、私有包或档位不一致")
            if report["timing"]["topk_selection"]["structural_errors"]:
                raise ValueError("Top-K结构错误")
            samples = report["timing"]["samples_us"]
            median = statistics.median(samples)
            sides[side].update(samples_us=samples, p95_over_p50=sides[side]["us_p95"] / median,
                               max_over_p50=sides[side]["us_max"] / median,
                               over_p50_5pct=sum(v > 1.05 * median for v in samples))
            trace = read(Path(sides[side]["profile_json"]))
            events = trace["traceEvents"] if isinstance(trace, dict) else trace
            if not any(e.get("ph") == "X" for e in events):
                raise ValueError("缺少可查看的PyTorch trace")
        for key in ("device", "cann", "requested"):
            if sides["native"][key] != sides["pto"][key]:
                raise ValueError(f"两侧配置不同：{key}")
        pto = reports["pto"]
        package = "vllm_ascend.ops.pypto." + source["variant"].removeprefix("pkg:")
        if (pto["implementation_package"] != package or not pto["compiler"]["pto_dispatch_calls"]
                or pto["implementation_source"] != str(Path(source["source"]) / (package.replace(".", "/")
                                                                                            + "/service.py"))):
            raise ValueError("PTO实际加载了不同源码或未执行")
        native = native_scope(reports["native"], Path(sides["native"]["profile_csv"]))
        incore_report = read(folder / "native_incore/report.json")
        if (incore_report["status"], incore_report["diagnostic_only"], incore_report["source"],
                incore_report["variant"], incore_report["history"], incore_report["batch"],
                incore_report["requested"], incore_report["device"], incore_report["cann"]) != (
                "PROFILED", True, source["source"], source["variant"], history, batch,
                reports["native"]["requested"], reports["native"]["device"], reports["native"]["cann"]):
            raise ValueError("Native核内诊断与主性能配置的形状/来源/设备不一致")
        incore_csv = list((folder / "native_incore/profile").rglob("kernel_details.csv"))
        if len(incore_csv) != 1:
            raise ValueError("Native核内诊断profile不唯一")
        native_incore = native_scope(incore_report, incore_csv[0], super_kernel=False)
        for name in ("VllmQuantLightningIndexer", "SparseAttnSharedkv"):
            if sum(row["Type"] == name for row in native_incore["kernels"]) != 1:
                raise ValueError(f"Native static核内诊断需要单独的{name}")
        dfx = read(folder / "swimlane/report.json")
        windows = dfx["swimlane_windows"]
        if (dfx["status"], dfx["variant"], dfx["history"], dfx["batch"], len(windows),
                dfx["effective_weight_nz_mode"], dfx["deterministic_level"],
                dfx["pto_reduction"]["atomic_add"]) != ("MEASURED", source["variant"], history, batch, 4, 2, 0, 0):
            raise ValueError("DFX档位/配置/窗口不完整")
        for window in windows:
            if not window["exported"] or (window["layer_index"], window["compact_metadata_policy"],
                                          window["input_source"], window["execution"]) != (
                    4, "reuse", "formal_layer_weights_synthetic_history", "graph_replay"):
                raise ValueError("DFX捕获范围变化")
        # Existing operator uses S6 for long cases and 8K/B24, dual-query for
        # 8K/B16/B32. The latter saturates all 48 coefficient workers.
        coefficient_workers = 48 if history == 8192 and batch in (16, 32) else batch
        summaries = [metrics.schedule_window(Path(w["merged_swimlane"]), coefficient_workers, helper, worker)
                     for w in windows]
        for w in summaries:
            if "merge_norm" in w["tasks"] or (w["tasks"]["qk_pv_aic"]["blocks"],
                                               w["tasks"]["qk_pv_aiv"]["blocks"]) != (24, 48):
                raise ValueError("Sparse融合结构或核心覆盖不符")
        means = {name: statistics.mean(w["tasks"][name]["kernel_mean_us"] for w in summaries) for name in TASKS}
        result["cases"].append({"history": history, "batch": batch, "sides": sides, "native_profile": native,
                                "native_incore": native_incore,
                                "worker_windows": summaries, "incore_means": means,
                                "csa_change_pct": 100 * (sides["pto"]["mean_us"] / sides["native"]["mean_us"] - 1)})
    if len({c["sides"]["native"]["device"] for c in result["cases"]}) != 1:
        raise ValueError("矩阵必须在同一个队列分配的设备执行")
    groups = {str(h): statistics.mean(c["csa_change_pct"] for c in result["cases"] if c["history"] == h)
              for h in (131072, 8192)}
    result["history_changes_pct"] = groups
    result["weighted_8_2_csa_change_pct"] = .8 * groups["131072"] + .2 * groups["8192"]
    (ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_report(result)


def write_report(result):
    lines = ["# 新Native基线与Sparse融合后的七档", "",
             "CANN9.2/mode2/det0；PTO atomic0；每档5预热20次设备事件，单位μs。",
             "Native显式torch.compile backend=npugraph_ex，自管图/static/superkernel；PTO生产自定义算子图。", "",
             "| 档位 | Native | PTO | 耗时变化 | Native/PTO P95 | Native/PTO max |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in result["cases"]:
        n, p = (c["sides"][s] for s in ("native", "pto"))
        lines.append(f"| {c['history']//1024}K/B{c['batch']} | {n['mean_us']:.3f} | {p['mean_us']:.3f} "
                     f"| {c['csa_change_pct']:+.3f}% | {n['us_p95']:.3f}/{p['us_p95']:.3f} "
                     f"| {n['us_max']:.3f}/{p['us_max']:.3f} |")
    lines += ["", f"各上下文内batch等权，长短8:2：{result['weighted_8_2_csa_change_pct']:+.3f}%。", "",
              "核内数据来自独立profile/DFX；以下完整融合范围不能拆成单个Native QLI或Sparse核时，",
              "也不能与各PTO均值直接相减计算纯算术收益。完整原始kernel名称和范围见evidence。", "",
              "| 档位 | Native可见端点及完整范围AIC/AIV | PTO系数 | PTO Score AIC/AIV | PTO Indexer merge | "
              "PTO融合Sparse AIC/AIV |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for c in result["cases"]:
        scopes = ["→".join(r["visible_endpoints"]) + f"（{r['Type']}）"
                  + f" {float(r['aicore_time(us)']):.3f}/{float(r['aiv_time(us)']):.3f}"
                  for r in c["native_profile"]["attention"]]
        m = c["incore_means"]
        lines.append(f"| {c['history']//1024}K/B{c['batch']} | {'<br>'.join(scopes)} | {m[TASKS[0]]:.3f} "
                     f"| {m[TASKS[1]]:.3f}/{m[TASKS[2]]:.3f} | {m[TASKS[3]]:.3f} "
                     f"| {m[TASKS[4]]:.3f}/{m[TASKS[5]]:.3f} |")
    lines += ["", "PTO已融合最终归一化、逆RoPE和发布，没有独立merge_norm；不是把缺失样本填0。", "",
              "核内细节另采Native static compile开启、SuperKernel关闭的profile；不把其区间填入主性能表。", "",
              "| 档位 | Native QLI AIC/AIV | PTO Score AIC/AIV | PTO系数/merge | "
              "Native Sparse AIC/AIV | PTO融合Sparse AIC/AIV |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in result["cases"]:
        native = {row["Type"]: row for row in c["native_incore"]["attention"]}
        qli, sparse = (native[n] for n in ("VllmQuantLightningIndexer", "SparseAttnSharedkv"))
        m = c["incore_means"]
        lines.append(f"| {c['history']//1024}K/B{c['batch']} "
                     f"| {float(qli['aicore_time(us)']):.3f}/{float(qli['aiv_time(us)']):.3f} "
                     f"| {m[TASKS[1]]:.3f}/{m[TASKS[2]]:.3f} | {m[TASKS[0]]:.3f}/{m[TASKS[3]]:.3f} "
                     f"| {float(sparse['aicore_time(us)']):.3f}/{float(sparse['aiv_time(us)']):.3f} "
                     f"| {m[TASKS[4]]:.3f}/{m[TASKS[5]]:.3f} |")
    lines += ["", "Native Sparse不含独立逆RoPE，而PTO融合Sparse包含它；以上仍是不同边界的参考，不能机械相减。", "",
              "| 档位 | PTO P95/P50 | PTO max/P50 | 超过P50的105% |",
              "| --- | ---: | ---: | ---: |"]
    for c in result["cases"]:
        p = c["sides"]["pto"]
        lines.append(f"| {c['history']//1024}K/B{c['batch']} | {p['p95_over_p50']:.4f} "
                     f"| {p['max_over_p50']:.4f} | {p['over_p50_5pct']}/20 |")
    lines += ["", "不删拖尾样本；20次正常采样不能关闭历史间歇拖尾或EP16稳定性问题。",
              "Native重放对同图compiled callable、PTO对自身eager；不冒称Native/PTO跨实现精度通过。",
              "全部配置、样本、28个官方join核对窗口、主性能21个JSON及7个Native核内诊断JSON见"
              "[evidence.json](evidence.json)。"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
