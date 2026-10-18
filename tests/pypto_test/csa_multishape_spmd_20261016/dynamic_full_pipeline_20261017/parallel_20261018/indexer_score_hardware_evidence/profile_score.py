# SPDX-License-Identifier: Apache-2.0
"""使用完整CSA的真实中间输入，测Score流水利用率；没有合成替代Score数学。"""

import csv
import json
import statistics
from pathlib import Path


def measure(prepared, compile_options, output):
    import torch
    import torch_npu
    from score_kernel import score_entry

    output.mkdir(parents=True, exist_ok=False)
    buffers = prepared.buffers
    policy = prepared.policy.constexprs()
    constants = dict(G=policy["QUERY_GROUP"], N=policy["QK_PANEL"], STEP=policy["SCORE_TILE"],
                     PREFETCH=policy["PREFETCH"], TO_L0=policy["TO_L0"], BALANCED=policy["BALANCED"])
    # 从同一份已运行的完整CSA读取原query/系数/cache/metadata；只另分配本段输出。
    keys = ("idx_query_i8", "coefficients", "idx_cache", "indexer_table", "positions",
            "idx_lengths", "idx_bounds", "group_offsets", "group_plan")
    source = [buffers[key] for key in keys]
    transfer = torch.full_like(buffers["indexer_transfer"], 37)
    scores = torch.full_like(buffers["score_arena"], 37)
    pairs = torch.full_like(buffers["pair_arena"], 37)
    arguments = [*source, transfer, scores, pairs, buffers["ffts"]]
    reference_pairs = buffers["pair_arena"].cpu()
    reference_scores = buffers["score_arena"].cpu()
    report = dict(passed=False, diagnostic_only=True, eligible_for_full_csa_performance=False,
                  constants=constants, input_origin="同进程完整CSA正式权重A→B→A后的真实中间Tensor",
                  input_shapes={key: list(buffers[key].shape) for key in keys}, checks=[])

    def verify(label):
        pair_equal = torch.equal(pairs.cpu(), reference_pairs)
        scores_equal = torch.equal(scores.cpu(), reference_scores)
        check = dict(label=label, full_pair_allocation_exact=pair_equal,
                     full_score_allocation_exact=scores_equal)
        report["checks"].append(check)
        if not pair_equal or not scores_equal:
            raise AssertionError("独立Score与完整CSA中同一函数输出不同：" + str(check))

    def run():
        score_entry(*arguments, **constants, config=compile_options)

    try:
        run()
        torch.npu.synchronize()
        verify("eager")
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                run()
            graph.replay()
            stream.synchronize()
            verify("graph")
            for _ in range(5):
                graph.replay()
                stream.synchronize()
            start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            samples = []
            for _ in range(10):
                start.record()
                graph.replay()
                end.record()
                stream.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            report["event_us"] = dict(samples=samples, median=statistics.median(samples),
                                      metric="独立Score图外Event，含外围调度，不是完整CSA或L1")
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
                record_shapes=False, profile_memory=False, with_stack=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output / "profile")),
            ) as profiler:
                for step in range(3):
                    with torch.profiler.record_function(f"CSA_INDEXER_SCORE_STEP_{step}"):
                        graph.replay()
                        stream.synchronize()
                    profiler.step()
            verify("after_profile")
        files = list((output / "profile").glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
        if len(files) != 1:
            raise RuntimeError("未生成唯一硬件profiling明细")
        rows = list(csv.DictReader(files[0].open()))
        report["profile_csv"] = str(files[0])
        report["profile_columns"] = list(rows[0]) if rows else []
        report["kernel_rows"] = rows
        if not any("aicore_kernel_mode" in row.get("Name", "") for row in rows):
            raise RuntimeError("profiling没有记录Score设备kernel")
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {"passed": True, "source": str(output / "result.json"), "diagnostic_only": True}
