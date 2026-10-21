# SPDX-License-Identifier: Apache-2.0
"""Reproduce CPU structural checks for the O_A/O_B ownership experiment."""

import collections
import json
import re
from pathlib import Path


def role_function(text, name):
    start = text.index("AICORE void " + name + "(")
    opening = text.index("{", start)
    depth = 1
    for end in range(opening + 1, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return re.sub(r"^\s*// pto:.*\n", "", text[start:end + 1], flags=re.M)
    raise ValueError("Unclosed role function")


def main():
    here = Path(__file__).resolve().parent
    runs = here / "runs"
    original = here.parents[2] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
    candidate = runs / "output_work_balance_native_candidate_v1"
    changed = sorted(p.name for p in original.glob("*.py") if p.read_bytes() != (candidate / p.name).read_bytes())
    assert changed == ["cube_schedule.py", "entry.py", "policy.py"]
    before = runs / "current_native_typefix_control_b16_v58/kernel/ptoas"
    after = runs / "output_work_balance_cpu_b16_v66/kernel/ptoas"
    cpp = [(p / "dynamic_csa_full.cpp").read_text() for p in (before, after)]
    pto = [(p / "dynamic_csa_full.pto").read_text() for p in (before, after)]
    assert role_function(cpp[0], "vector_schedule") == role_function(cpp[1], "vector_schedule")
    allocation = [collections.Counter(re.findall(
        r"pto.alloc_tile addr = (\S+) [^\n]* : (!pto.tile_buf<[^>]+>)", text,
    )) for text in pto]
    assert allocation[0] == allocation[1]
    tile_ops = [re.findall(r"\b(pto\.t[a-z0-9_]+)\b", text) for text in pto]
    assert tile_ops[0] == tile_ops[1]
    sync = {}
    for symbol in ("PTOAS__WAIT_GE_BACKOFF<150>(", "pto::comm::TNOTIFY(",
                   "wait_flag_dev(", "ffts_cross_core_sync("):
        counts = [text.count(symbol) for text in cpp]
        assert counts[0] == counts[1]
        sync[symbol] = counts
    for batch in (1, 16):
        result = json.loads((runs / f"output_work_balance_cpu_b{batch}_v66/result.json").read_text())
        assert result["passed"] and result["full_codegen_passed"]
        launch_path = runs / f"output_work_balance_cpu_b{batch}_v66/kernel/orchestration/decode_csa_spmd.cpp"
        orchestration = launch_path.read_text()
        assert orchestration.count("rt_submit_task(") == 1
        assert re.findall(r"set_block_num\((\d+)\)", orchestration) == ["24"]
        assert "MixedKernels (AIC + AIV lanes)" in orchestration
    report = {
        "passed": True, "device_execution": False, "changed_files": changed,
        "all_math_helpers_identical": True,
        "whole_vector_function_equal_except_pto_comments": True,
        "all_tile_allocation_layout_multisets_equal": True,
        "whole_module_tile_operation_sequence_equal": True,
        "tile_operation_sites": len(tile_ops[0]), "sync_static_sites": sync,
        "single_mix_task_with_24_physical_groups": True,
        "scope": "32 software O_B ranks on 24 physical AICs; group quant writers and final PHASE_OB preserved",
        "limitation": "CPU structure only; no claim about machine scheduling or device performance",
    }
    (here / "output_work_balance_codegen_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
