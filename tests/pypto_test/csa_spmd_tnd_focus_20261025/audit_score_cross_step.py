# SPDX-License-Identifier: Apache-2.0
"""Audit emitted cross-step L1 ownership, unchanged math, resources and ABI."""

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASES = {
    "C04": "codegen_score_cross_step_baseline_C04_v1",
    "C05": "codegen_score_cross_step_baseline_C05_v1",
}


def normalized(source):
    source = re.sub(r"//[^\n]*", "", source)
    names = {}
    source = re.sub(r"\bv\d+\b", lambda m: names.setdefault(m[0], f"var{len(names)}"), source)
    return re.sub(r"\s+", "", source)


def resources(source):
    peaks, regions = {}, {}
    pattern = r"(%[\w]+) = pto.alloc_tile addr = %c(\d+)_i64[^\n]*loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)"
    for match in re.finditer(pattern, source):
        name, offset, kind, dtype, rows, cols = match.groups()
        offset = int(offset)
        size = int(rows) * int(cols) * max(int(re.search(r"\d+", dtype)[0]) // 8, 1)
        peaks[kind] = max(peaks.get(kind, 0), offset + size)
        if re.match(r"%buf_(key_l1_pool|current_key|prefetched_key)_inline|%split_(qk_acc[01]|ws_acc|right[01])", name):
            regions.setdefault(name.split("_inline")[0], set()).add((kind, offset, size))
    return peaks, {name: sorted(value) for name, value in regions.items()}


def full_step(path):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if re.search(r"// pto: %buf_key_l1_pool_inline\d+__ssa_v0", line))
    end = next(i for i in range(start, len(lines)) if "ffts_cross_core_sync(PIPE_FIX" in lines[i])
    # Include the preceding QK and WS accumulator declarations for operand mapping.
    acc_start = max(i for i in range(start) if re.search(r"// pto: %split_query0_inline\d+__ssa_v0", lines[i]))
    acc_vars = {}
    for role in ("qk_acc0", "qk_acc1", "ws_acc"):
        marker = next(i for i in range(acc_start, start) if f"// pto: %split_{role}_inline" in lines[i])
        acc_vars[role] = re.search(r"(v\d+) = Tile<TileType::Acc", "\n".join(lines[marker:start]))[1]
    body = "\n".join(lines[start : end + 1])
    trace = []
    for line in lines[start : end + 1]:
        if match := re.search(r"(TMATMUL(?:_ACC)?)\((v\d+),", line):
            trace.append((match[1], next(role for role, value in acc_vars.items() if value == match[2])))
        if "TINSERT<" in line:
            match = re.search(r">\(v\d+, (v\d+),", line)
            trace.append(("TINSERT", next(role for role, value in acc_vars.items() if value == match[1])))
        if "TSTORE(" in line:
            trace.append(("TSTORE", "ws_acc"))
    counters = Counter(re.findall(r"\b(T[A-Z_]+)(?:<[^\n]*>)?\(", body))
    return lines, start, end, body, trace, counters


def audit(case):
    run = HERE / "runs" / f"codegen_score_cross_step_{case}_v1"
    result = json.loads((run / "result.json").read_text())
    assert result["passed"] and result["stage"] == "cpu_backend_compiled" and not result["device_execution"]
    assert result["shared_toolchain"]["release"] == "csa-shared-20261024-group-dims"
    assert result["policy"]["PREFETCH"] == 1 and result["policy"]["TO_L0"] is True
    changed = [
        name
        for name, digest in result["source_hashes"]["reference"].items()
        if result["source_hashes"]["candidate"][name] != digest
    ]
    assert changed == ["indexer_score_stages.py"]
    (kernel,) = run.glob("build_output/*")
    baseline_run = HERE / "runs" / BASES[case]
    baseline_result = json.loads((baseline_run / "result.json").read_text())
    assert baseline_result["passed"] and baseline_result["stage"] == "cpu_backend_compiled"
    assert baseline_result["shared_toolchain"] == result["shared_toolchain"]
    (baseline,) = baseline_run.glob("build_output/*")
    cpp = kernel / "ptoas/dynamic_csa_full.cpp"
    old_lines, old_start, old_end, old_body, old_math, old_counts = full_step(baseline / "ptoas/dynamic_csa_full.cpp")
    lines, start, end, body, math, counts = full_step(cpp)
    assert math == old_math and len(math) == 112
    assert counts["TMOV"] == old_counts["TMOV"] == 0
    assert "PIPE_ALL" not in body and "PIPE_S" not in body
    assert counts["TEXTRACT"] == old_counts["TEXTRACT"] == 48
    assert counts["TLOAD"] == 68 and old_counts["TLOAD"] == 64
    assert len(re.findall(r"= \(v\d+\)\[", body)) == 34
    assert Counter(re.findall(r"pipe_barrier\((PIPE_\w+)\)", body)) == Counter(
        re.findall(r"pipe_barrier\((PIPE_\w+)\)", old_body)
    )
    new_peaks, new_regions = resources((kernel / "ptoas/dynamic_csa_full.pto").read_text())
    old_peaks, old_regions = resources((baseline / "ptoas/dynamic_csa_full.pto").read_text())
    assert new_peaks == old_peaks
    assert new_regions == old_regions
    assert json.loads((kernel / "compiled_meta.json").read_text()) == json.loads(
        (baseline / "compiled_meta.json").read_text()
    )
    orch = "orchestration/decode_csa_spmd.cpp"
    assert normalized((kernel / orch).read_text()) == normalized((baseline / orch).read_text())
    aiv = "kernels/aiv/vector_schedule.cpp"
    marker = "static __aicore__ void vector_schedule("
    assert normalized((kernel / aiv).read_text().split(marker, 1)[1]) == normalized(
        (baseline / aiv).read_text().split(marker, 1)[1]
    )

    # Concrete terminal write and next-step read edges. All phi aliases have the
    # same 16KiB L1 interval, so no hidden Mat copy crosses the loop back edge.
    step = next(i for i in range(start, end) if "for (int64_t" in lines[i])
    loop_var = re.search(r"for \(int64_t (v\d+)", lines[step])[1]
    prime_guard = next(i for i in range(step, end) if f"if ({loop_var} ==" in lines[i])
    cross_guard = max(i for i in range(step, end) if "if ((int64_t)" in lines[i] and loop_var in lines[i])
    next_key_loads = [i for i in range(cross_guard, end) if "TLOAD(" in lines[i]]
    assert len(next_key_loads) == 4  # Two pages, each with mutually exclusive 0/64B views.
    key_types = set(re.findall(r"(v\d+) = Tile<TileType::Right, int8_t, 128, 64,", body))
    extracts = [i for i in range(start, end) if (m := re.search(r"TEXTRACT\((v\d+),", lines[i])) and m[1] in key_types]
    assert len(extracts) == 16
    first_extract, last_slot0_extract = extracts[0], extracts[14]
    assert last_slot0_extract < cross_guard < next_key_loads[0]
    assert all(i < cross_guard for i in extracts), "cross-step Key must not enter Right early"
    free_set = next(
        i for i in range(last_slot0_extract, cross_guard) if "set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2)" in lines[i]
    )
    free_wait = max(i for i in range(free_set, cross_guard) if "wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2)" in lines[i])
    acquire_set = max(
        i for i in range(prime_guard, first_extract) if "set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0)" in lines[i]
    )
    acquire_wait = max(
        i for i in range(acquire_set, first_extract) if "wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0)" in lines[i]
    )
    # set/wait is outside prime's dynamic branch: next step still acquires the
    # preceding iteration's terminal DMA when its prime branch is skipped.
    loop_indent = len(lines[step]) - len(lines[step].lstrip()) + 2
    assert len(lines[acquire_set]) - len(lines[acquire_set].lstrip()) == loop_indent
    assert len(lines[acquire_wait]) - len(lines[acquire_wait].lstrip()) == loop_indent
    right_wait = max(i for i in range(step, first_extract) if "wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1)" in lines[i])
    right_set = max(i for i in range(first_extract, end) if "set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1)" in lines[i])
    final_qk = [i for i in range(next_key_loads[-1], end) if re.search(r"TMATMUL\(", lines[i])]
    assert len(final_qk) == 4  # QK0, QK1, previous WS0, final WS0; both WS1 use ACC.
    step_events = "\n".join(lines[step : end + 1])
    sets = Counter(re.findall(r"set_flag\((PIPE_\w+), (PIPE_\w+), (EVENT_ID\d+)\)", step_events))
    waits = Counter(re.findall(r"wait_flag\((PIPE_\w+), (PIPE_\w+), (EVENT_ID\d+)\)", step_events))
    assert sets == waits
    evidence = {
        "case": case,
        "cpu_backend_compiled": True,
        "device_execution": False,
        "generated_cpp": str(cpp),
        "baseline_cpp": str(baseline / "ptoas/dynamic_csa_full.cpp"),
        "full_score_span": [start + 1, end + 1],
        "actual_policy": {
            key: result["policy"][key]
            for key in ("QUERY_GROUP", "QK_PANEL", "SCORE_TILE", "PREFETCH", "TO_L0", "BALANCED")
        },
        "math_trace_identical": True,
        "math_trace_length": len(math),
        "aiv_function_equal_modulo_generated_variable_names": True,
        "orchestration_and_compiled_metadata_unchanged": True,
        "tile_raw_address_ends_bytes": new_peaks,
        "resource_delta_bytes": {kind: 0 for kind in new_peaks},
        "score_regions": new_regions,
        "no_score_TMOV_PIPE_ALL_or_PIPE_S": True,
        "per_pipe_barriers_unchanged": True,
        "static_counts": {"baseline": dict(old_counts), "candidate": dict(counts)},
        "runtime_work": (
            "For N steps: 2 prime pages once + 30*N within-step pages + 2*(N-1) cross-step pages = 32*N; "
            "exactly 16*N Key TEXTRACT, 64*N MMA, 16*N score stores."
        ),
        "cross_step_events": {
            "prime_guard": prime_guard + 1,
            "next_step_guard": cross_guard + 1,
            "slot0_last_read": last_slot0_extract + 1,
            "slot0_release_set": free_set + 1,
            "slot0_reuse_wait": free_wait + 1,
            "next_key_loads_mutually_exclusive_views": [i + 1 for i in next_key_loads],
            "next_iteration_acquire_set": acquire_set + 1,
            "next_iteration_acquire_wait": acquire_wait + 1,
            "next_iteration_first_key_extract": first_extract + 1,
            "previous_iteration_Right_release_set": right_set + 1,
            "next_iteration_Right_reuse_wait": right_wait + 1,
            "step_local_set_wait_counts_balanced": True,
        },
        "mechanism_gate": (
            "The next-step Key TLOADs are emitted before terminal QK/FIX/WS; "
            "next iteration acquires them without reissuing prime DMA. Right lifetime is unchanged."
        ),
        "limitations": [
            "No NPU correctness, repeated graph, or complete L1 timing has been run.",
            "Static ownership/event audit is not a proof of all asynchronous device executions.",
            "Same bytes/MMA plus added dynamic guards: emitted overlap opportunity does not imply speedup.",
        ],
        "adopted": False,
        "performance_eligible": False,
    }
    (HERE / f"score_cross_step_{case}_v1_codegen.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "case": case,
        "passed": True,
        "resource_delta_bytes": evidence["resource_delta_bytes"],
        "device_execution": False,
    }


if __name__ == "__main__":
    print(json.dumps([audit(case) for case in BASES], ensure_ascii=False, indent=2))
