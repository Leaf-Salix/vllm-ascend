"""只用旧版 B16/S6，区分 DFX 窗口前预热与窗口内预热；保留所有原始采样。"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench_before_warmup_fix as bench

original_measure = bench.measure_incore
original_profile = bench.profile_graph


def audit_measure(graphs, args, case_name, window, variant=None):
    import pypto.torch
    import torch
    from dsv4_csa_single_card_bench import _export_swimlane
    from dsv4_hc_pre_timing import summarize_swimlane

    def profile_then_warmup(graph, output):
        trace = original_profile(graph, output)
        for _ in range(5):
            graphs["pto"].replay()
            torch.npu.synchronize()
        return trace

    bench.profile_graph = profile_then_warmup
    try:
        result = original_measure(graphs, args, case_name, window, variant)
    finally:
        bench.profile_graph = original_profile
    result["audit_condition"] = "Native profiler 完成后、begin_dfx 之前追加 5 次 PTO 预热，再采 3 次"

    # 另开窗口；前 5 次事先固定为预热，后 3 次为测量。原始 8 次全部导出，禁止按数值挑样。
    start = time.perf_counter()
    pypto.torch.begin_dfx()
    begin_ms = (time.perf_counter() - start) * 1000
    try:
        for _ in range(8):
            graphs["pto"].replay()
            torch.npu.synchronize()
    finally:
        pypto.torch.end_dfx()
    directory = args.output / "dfx" / "window_1"
    exported = _export_swimlane(directory, kernel_pattern="_jit_hc_pre_*/kernel_config.py")
    if not exported["exported"]:
        raise RuntimeError(exported)
    trace = directory / "merged_swimlane.json"
    all_samples = summarize_swimlane(trace, 8)
    measured = all_samples["samples"][5:]
    complete = all(sample["blocks"] == 72 for sample in all_samples["samples"])

    audit = {"scope": "仅 L1 incore 首尾；不计 Simpler/Orchestrator 外层头尾",
             "before_begin_warmup": result["pto"],
             "inside_window_warmup": {"warmup_epochs": [0, 1, 2, 3, 4], "measured_epochs": [5, 6, 7],
                                      "all_raw_samples": all_samples,
                                      "measurement_valid": complete,
                                      "invalid_reason": None if complete else "部分 epoch 少于预期的 24 AIC + 48 AIV 记录，不作为性能证据",
                                      "measured_span_p50_us": statistics.median(x["span_us"] for x in measured) if complete else None},
             "begin_dfx_host_ms_context_only": begin_ms}
    (args.output / "warmup_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print("WARMUP_AUDIT " + json.dumps(audit, ensure_ascii=False), flush=True)
    return result


bench.measure_incore = audit_measure
bench.main()
