# SPDX-License-Identifier: Apache-2.0
"""同行进程核对行依赖候选与此前所选单 SPMD，仅采完整无插桩 L1。"""

import importlib.util
import json
import sys
from pathlib import Path

from shape_specialization import build_variant, descriptors_from_tensors
from signal_polling import bypass_signal_wait


def compare(*, args, call, options, graphs, stream, reset, check, reference,
            export_window, statistics_us, reference_row_dependencies=False,
            candidate_label="row_dependency_spmd", folder="paired_selected_spmd"):
    from test_static_jit_device import (
        PreparedStaticArguments, progress_expectation, pypto, torch,
    )

    destination = args.output / folder
    case, _ = build_variant(
        descriptors_from_tensors(call.args), max_seq_len=args.history + 6,
        output_root=destination / "specialization", case_name=f"h{args.history}_b{args.batch}",
        balanced_schedule=True, schedule_policy=args.schedule_policy,
        cost_profile=args.schedule_cost_profile, qk_vector_indices=args.qk_vector_indices,
        row_dependencies=reference_row_dependencies)
    layout = json.loads((case / "layout.json").read_text())
    prepared = PreparedStaticArguments(case, call.args, workspace_guard_bytes=512)
    prepared.require_core_capacity(256)
    spec = importlib.util.spec_from_file_location("csa_paired_selected_entry", case / "entry_jit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    reset()
    with bypass_signal_wait(destination / "signal_polling", args.wait_backoff_ticks, cached=True):
        module.entry(*prepared.core_args, config=options)
    stream.synchronize()
    checks = {"selected_eager": check(reference)}
    reset()
    stream.synchronize()
    selected_graph = torch.npu.NPUGraph()
    with torch.npu.graph(selected_graph, stream=stream):
        module.entry(*prepared.core_args, config=options)
    reset()
    selected_graph.replay()
    stream.synchronize()
    checks["selected_graph"] = check(reference)
    selected_generations, candidate_generations = 2, 0
    replay_graphs = {"selected_spmd": selected_graph, candidate_label: graphs["single_spmd"]}
    # 两边均已编译；相同权重/输入/输出/metadata，只保留独立工作区和完成信号。
    for iteration in range(2):
        order = list(replay_graphs) if iteration % 2 == 0 else list(reversed(replay_graphs))
        for side in order:
            reset()
            replay_graphs[side].replay()
            stream.synchronize()
            selected_generations += side == "selected_spmd"
            candidate_generations += side == candidate_label
    windows = []
    window = 0
    for sample in range(3):
        order = list(replay_graphs) if sample % 2 == 0 else list(reversed(replay_graphs))
        for side in order:
            reset()
            stream.synchronize()
            pypto.torch.begin_dfx()
            try:
                replay_graphs[side].replay()
            finally:
                pypto.torch.end_dfx()
            selected_generations += side == "selected_spmd"
            candidate_generations += side == candidate_label
            source = args.output / "dfx"
            if window:
                source /= f"window_{window}"
            # 两侧都要求72个incore成员、同一个task；导出函数的single_spmd校验适用于两侧。
            record = export_window(source, destination / side / f"step_{sample + 1}", "single_spmd")
            record.update(side=side, step=sample + 1, dfx_window=window)
            windows.append(record)
            window += 1
            if sample == 2:
                checks[side + "_final"] = check(reference)
    progress = prepared.sections["progress"].cpu()[::layout["protocol"]["progress_words_per_core"]]
    guard = prepared.workspace_guard_bytes
    progress_passed = torch.equal(progress, progress_expectation(layout, selected_generations))
    guards_passed = bool((prepared.workspace_allocation[:guard].cpu() == 37).all()
                        and (prepared.workspace_allocation[-guard:].cpu() == 37).all())
    if not (progress_passed and guards_passed):
        raise AssertionError("同进程selected单SPMD的信号或工作区保护区不符")
    measurements = {side: statistics_us([r["incore_span_us"] for r in windows if r["side"] == side])
                    for side in replay_graphs}
    selected = measurements["selected_spmd"]["median_us"]
    candidate = measurements[candidate_label]["median_us"]
    report = {"passed": True, "performance_measured": True,
              "metric": "无插桩L1的min(incore start)至max(incore end)，包括等待与间隙",
              "same_process": True, "same_root_addresses": True, "native_forward_executed": False,
              "reference_row_dependencies": reference_row_dependencies,
              "candidate_label": candidate_label,
              "windows": windows, "l1_incore": measurements,
              "candidate_change_percent": (candidate / selected - 1) * 100,
              "selected_signal_generations": selected_generations,
              "candidate_additional_generations": candidate_generations,
              "selected_progress_passed": progress_passed, "selected_workspace_guards_passed": guards_passed,
              "exact": checks}
    (destination / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report, candidate_generations
