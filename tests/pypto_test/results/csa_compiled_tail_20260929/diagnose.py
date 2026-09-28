"""Locate the compiled-entry tail in device execution versus graph launch gaps."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "csa_compiled_pair_20260929"))
import compiled_case as case  # noqa: E402

ORIGINAL_MEASURE = case.measure_graph_interval


def diagnose(fixture, run, output, topk, reference, *, iters, warmup, require_exact, profile_dir):
    torch, torch_npu = case.torch, case.torch_npu
    # Retain the existing unprofiled event, state and guard contract once.
    result = ORIGINAL_MEASURE(fixture, run, output, topk, reference,
                              iters=iters, warmup=warmup, require_exact=require_exact, profile_dir=None)
    initial = {name: group["initial"].to(group["allocation"].device)
               for name, group in fixture["groups"].items()}

    def reset():
        for name, group in fixture["groups"].items():
            group["allocation"].copy_(initial[name])
        output.fill_(float("nan"))

    reset()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for _ in range(warmup):
        reset()
        graph.replay()
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    windows = []
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
    ) as trace:
        for index in range(iters):
            reset()
            torch.npu.synchronize()
            with torch.autograd.profiler.record_function(f"CSA_REPLAY_{index:02d}"):
                start.record()
                graph.replay()
                end.record()
                torch.npu.synchronize()
            windows.append({"index": index, "event_us": start.elapsed_time(end) * 1000,
                            "start_timestamp_raw": start.recorded_time()})
        trace.step()
    result["profiled_replays"] = windows
    result["profile"] = {"directory": str(profile_dir), "count": iters,
                         "scope": "Diagnostic profile; not mixed into unprofiled timing"}
    return result


if __name__ == "__main__":
    case.measure_graph_interval = diagnose
    case.main()
