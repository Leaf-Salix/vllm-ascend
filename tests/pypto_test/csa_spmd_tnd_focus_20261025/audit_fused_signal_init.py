# SPDX-License-Identifier: Apache-2.0
"""Audit real guarded-MIX codegen, clear coverage, aliases, and FFTS ordering."""

import argparse
import ast
import json
import re
from pathlib import Path


def function_parts(text, pattern):
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    return {
        match.group(1): text[match.start() : matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        for i, match in enumerate(matches)
    }


def source_checks(here, candidate, parent=None):
    if parent is None:
        parent = here / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd"
    changes = [p.name for p in sorted(candidate.glob("*.py")) if p.read_bytes() != (parent / p.name).read_bytes()]
    assert changes == ["cube_schedule.py", "entry.py", "vector_schedule.py"], changes
    for role, tail in (
        ("cube", "    tokens = pl.tensor.dim(x_fp32, 0)\n"),
        ("vector", "    tokens = pl.tensor.dim(x, 0)\n"),
    ):
        before = (parent / f"{role}_schedule.py").read_text()
        after = (candidate / f"{role}_schedule.py").read_text()
        assert before[before.index(tail) :] == after[after.index(tail) :], role
        assert after.count("pl.system.syncall(core_type=pl.KernelType.MIX)") == 1
        assert after.index("pl.system.syncall(") < after.index("signal, generation = begin_invocation(")
        assert "FUSED_SIGNAL_INIT: pl.constexpr = False" in after
    before_ast = ast.parse((parent / "entry.py").read_text())
    after_ast = ast.parse((candidate / "entry.py").read_text())
    entry_checks = {}
    for name in ("decode_csa_spmd", "decode_csa_spmd_debug"):
        before = next(n for n in before_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
        after = next(n for n in after_ast.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(before.args) == ast.dump(after.args), name
        allocations = lambda n: [
            ast.dump(c)
            for c in ast.walk(n)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "create_tensor"
        ]
        assert allocations(before) == allocations(after), name
        guard = next(
            n for n in after.body if isinstance(n, ast.If) and "available_cluster_count" in ast.unparse(n.test)
        )
        assert ast.unparse(guard.test) == "pl.system.available_cluster_count() == GROUPS"
        fused_calls = [
            n
            for n in ast.walk(guard)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_csa_core_group"
        ]
        assert len(fused_calls) == 2
        assert [next(k.value.value for k in n.keywords if k.arg == "FUSED_SIGNAL_INIT") for n in fused_calls] == [
            True,
            False,
        ]
        assert "_clear_internal_signal" not in ast.unparse(guard.body)
        assert "_clear_internal_signal" in ast.unparse(guard.orelse)
        entry_checks[name] = {
            "external_signature_unchanged": True,
            "GM_allocations_unchanged": len(allocations(after)),
            "capability_guard": ast.unparse(guard.test),
            "fused_has_one_group_no_clear_task": True,
        }
    words = 60 * 72 * 128
    per_worker = words // 48
    intervals = [(worker * per_worker, (worker + 1) * per_worker) for worker in range(48)]
    assert intervals[0][0] == 0 and intervals[-1][1] == words
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    assert all(start % 128 == 0 and end % 128 == 0 for start, end in intervals)
    return {
        "changed_files": changes,
        "both_schedule_math_and_business_sync_suffixes_identical": True,
        "entries": entry_checks,
        "clear_words": words,
        "clear_bytes": words * 4,
        "words_per_AIV": per_worker,
        "intervals_words": intervals,
        "slot_bytes": 512,
        "coverage_exact_disjoint_and_512_byte_aligned": True,
    }


def inspect_run(run):
    result = json.loads((run / "result.json").read_text())
    assert result["stage"] == "cpu_backend_compiled" and result["passed"]
    assert not result["device_execution"]
    builds = [p for p in (run / "build_output").iterdir() if p.is_dir()]
    assert len(builds) == 1
    build = builds[0]
    orch_path = next((build / "orchestration").glob("*.cpp"))
    orch = orch_path.read_text()
    condition = re.search(r"if \(\(static_cast<int64_t>\(\w+\) == 24\)\) \{", orch)
    assert condition
    branch = orch[condition.end() :]
    fused, fallback = branch.split("} else {", 1)
    assert fused.count("rt_submit_task(") == 1 and "rt_submit_aiv_task(" not in fused
    assert "set_block_num(rt_available_cluster_count())" in fused
    assert "set_require_sync_start(true)" in fused
    mix = re.search(r"MixedKernels \w+ = \{(\d+), (\d+), (\d+)\};", fused)
    assert mix and mix.group(2) == mix.group(3) and mix.group(1) != mix.group(2)
    assert fallback.count("rt_submit_aiv_task(") == 1 and fallback.count("rt_submit_task(") == 1
    assert "set_block_num(48)" in fallback and "set_block_num(24)" in fallback
    assert "Tensor signal_init_matrix = signal.reshape(signal_init_matrix_shapes, 2);" in orch
    assert "signal_init_matrix_shapes[2] = {48, 11520}" in orch
    assert ".add_inout(signal);" in fused and ".add_inout(signal_init_matrix);" in fused

    fused_pto_path = build / "ptoas/dynamic_csa_full.pto"
    pto = fused_pto_path.read_text()
    functions = function_parts(pto, r"^  func\.func @(\w+)\(")
    assert set(functions) == {"cube_schedule", "vector_schedule"}
    fallback_pto = (build / "ptoas/dynamic_csa_full_0.pto").read_text()
    assert "pto.syncall" not in fallback_pto
    native_path = build / "ptoas/dynamic_csa_full.cpp"
    native = function_parts(native_path.read_text(), r"^static __aicore__ void (\w+)\(")
    # PTOAS emits global functions before PyPTO adds static wrappers on some versions.
    if not native:
        native = function_parts(native_path.read_text(), r"^(?:__aicore__ )?void (\w+)\(")
    if not {"cube_schedule", "vector_schedule"} <= native.keys():
        wrapper = build / "kernels/aiv/vector_schedule.cpp"
        native_path = wrapper
        native = function_parts(wrapper.read_text(), r"^static __aicore__ void (\w+)\(")

    roles = {}
    all_events = set()
    for role in ("cube_schedule", "vector_schedule"):
        p = functions[role]
        n = native[role]
        assert p.count("pto.syncall()") == 1
        assert n.count("SYNCALL<SyncCoreType::Mix>();") == 1
        barrier = n.index("SYNCALL<SyncCoreType::Mix>();")
        prior = n.index("// pto: %prior_")
        notify = n.index("pto::comm::TNOTIFY(")
        assert barrier < prior < notify
        assert "dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);" in n[barrier:prior]
        assert "dsb(DSB_DDR);" in n[max(0, barrier - 300) : barrier]
        if role == "vector_schedule":
            first_store = n.index("TSTORE(")
            assert first_store < barrier
            assert "pipe_barrier(PIPE_ALL);" in n[first_store:barrier]
            assert "dsb(DSB_DDR);" in n[first_store:barrier]
            assert "TLOAD(" not in n[:barrier]
            assert "TNOTIFY(" not in n[:barrier] and "TWAIT(" not in n[:barrier]
            zero = re.search(r"%zero__\w+ = pto.alloc_tile addr = %c(\d+)_i64.*rows=1, cols=11520", p)
            assert zero
            zero_start = int(zero.group(1))
            assert zero_start + 46080 <= 192 * 1024
        else:
            assert "TNOTIFY(" not in n[:barrier] and "TWAIT(" not in n[:barrier]
            zero_start = None
        event_lines = [line for line in p.splitlines() if "pto.sync.set " in line or "pto.sync.wait " in line]
        events = set()
        for line in event_lines:
            event = re.search(r"%c(\d+)_index", line)
            assert event, line
            events.add(int(event.group(1)))
        all_events |= events
        prefix = n[:notify]
        roles[role] = {
            "one_hardware_mix_barrier": True,
            "barrier_precedes_first_generation_read_and_notify": True,
            "barrier_followed_by_consumer_invalidate": True,
            "clear_store_PIPE_ALL_DSB_before_barrier": role == "vector_schedule",
            "zero_UB_range_bytes": None if zero_start is None else [zero_start, zero_start + 46080],
            "explicit_business_FFTS_events": sorted(events),
            "native_prefix_tail": prefix.splitlines()[-85:],
        }
    assert all_events <= {0, 1, 2, 3, 4, 5}
    assert not (all_events & {11, 12, 13})
    return {
        "case": result["case"]["id"],
        "case_name": result["case"]["name"],
        "build": str(build),
        "compile_result": str(run / "result.json"),
        "orchestration": str(orch_path),
        "fused_active_tasks": 1,
        "fused_active_roles": {"AIC": 24, "AIV": 48},
        "fused_MixedKernels": list(map(int, mix.groups())),
        "fallback_tasks": 2,
        "runtime_device_width_guard_and_query_launch": True,
        "signal_and_matrix_alias_same_GM_allocation": True,
        "fallback_has_no_hardware_syncall": True,
        "full_PyPTO_memory_planner_and_PTOAS_codegen_passed": True,
        "no_new_GM_allocations": True,
        "device_execution": False,
        "PTO": str(fused_pto_path),
        "native_code": str(native_path),
        "roles": roles,
    }


def toolchain_checks(here):
    workspace = next(p for p in here.parents if (p / "env-dsv4-0251rc1.sh").is_file())
    manifest_path = workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["release"] == "csa-shared-20261024-group-dims" and manifest["status"] == "validated"
    simpler = workspace / manifest["simpler"]["source"]
    pypto = workspace / manifest["pypto"]["source"]
    isa_pin = (simpler / "pto_isa.pin").read_text().strip()
    metadata_path = simpler / "build/lib/pto_isa_build.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["actual_checkout_commit"] == metadata["required_commit_from_pin"] == isa_pin
    isa = Path(metadata["checkout_path"])
    impl_path = isa / "include/pto/npu/a2a3/SyncAll.hpp"
    impl = impl_path.read_text()
    for fragment in (
        "wait_flag_dev(SYNC_AIV_FLAG);",
        "ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));",
        "wait_flag_dev(SYNC_AIC_FLAG);",
        "ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x2, SYNC_AIC_AIV_FLAG));",
        "ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x2, SYNC_AIV_FLAG));",
        "wait_flag_dev(SYNC_AIC_AIV_FLAG);",
    ):
        assert fragment in impl
    constants = (isa / "include/pto/common/type.hpp").read_text()
    for name, value in (("SYNC_AIC_FLAG", 11), ("SYNC_AIV_FLAG", 12), ("SYNC_AIC_AIV_FLAG", 13)):
        assert f"constexpr uint16_t {name} = {value};" in constants
    compiler = (simpler / "simpler_setup/toolchain.py").read_text()
    assert '"-D__PTO_AUTO__"' not in compiler
    ffts_host = simpler / "src/a2a3/platform/onboard/host/device_runner.cpp"
    ffts_core = simpler / "src/a2a3/platform/onboard/aicore/kernel.cpp"
    assert "rtGetC2cCtrlAddr(&ffts_base_addr, &ffts_len)" in ffts_host.read_text()
    assert "set_ffts_base_addr((uint64_t)k_args->ffts_base_addr);" in ffts_core.read_text()
    preprocess = pypto / "python/pypto/backend/_ptoas_preprocess.py"
    assert 'if stripped.startswith("set_ffts_base_addr("):\n            continue' in preprocess.read_text()
    occupancy = pypto / "src/ir/verifier/verify_hard_syncall_occupancy.cpp"
    assert "kernel->func_type_ == FunctionType::Group" in occupancy.read_text()
    return {
        "manifest": str(manifest_path),
        "release": manifest["release"],
        "isa_pin": isa_pin,
        "isa_build_metadata": str(metadata_path),
        "actual_isa_implementation": str(impl_path),
        "MIX_protocol": "AIV mode2 event12 -> AIC all-core mode0 event11 -> AIC mode2 event13 -> AIV wait13",
        "hardware_events": [11, 12, 13],
        "GM_bootstrap_workspace": False,
        "runtime_FFTS_source": [str(ffts_host), str(ffts_core)],
        "dummy_ffts_operand_is_not_hardware_base": str(preprocess),
        "group_full_occupancy_and_sync_start_verifier": str(occupancy),
        "PTO_AUTO_not_enabled_by_incore_toolchain": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    report = {
        "status": "CPU_protocol_codegen_gate_passed",
        "candidate": str(args.candidate.resolve()),
        "shared_toolchain_evidence": toolchain_checks(here),
        "source_contract": source_checks(here, args.candidate),
        "runs": [inspect_run(run) for run in args.run],
        "scope": "Source and complete CPU codegen only; no NPU correctness, graph replay or performance claim",
        "measurement": "fused branch includes clear + fence + hard barrier + all math to MIX exit",
        "device_execution": False,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
