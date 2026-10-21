"""CPU-only recomputation of saved T168 QA/full-CSA evidence, not NPU replay."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    for name, digest in json.loads((ROOT / "SHA256.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    qa = json.loads((ROOT / "qa-packet-result.json").read_text())
    assert qa["complete"] and qa["pass"]
    assert all(qa["checks"].values())
    assert qa["actual_torch_deterministic"] and qa["actual_npu_deterministic_level"] == 1
    assert len(qa["cases"]) == 5
    for row in qa["cases"].values():
        assert row["metrics"]["packet"]["bit_equal"]
        assert row["metrics"]["packet"]["different"] == 0
        for flag in (
            "actual_raw_graph",
            "output_poison_restored",
            "poisoned_repeat_exact",
            "readonly_and_outer_guards",
        ):
            assert row[flag], flag
    full = json.loads((ROOT / "full-csa-result.json").read_text())
    assert full["complete"] and full["pass"] and full["precision_pass"]
    assert full["timing_executed"] is False
    assert full["true_tokens"] == full["physical_graph_tokens"] == 168
    assert len(full["original_native_A_invariance"]) == 8 and all(full["original_native_A_invariance"].values())
    for key in ("comparisons", "B_comparisons"):
        rows = full[key]["candidate"]
        assert len(rows) == 8
        for row in rows.values():
            assert row["bit_equal"] and row["different_elements"] == 0
            assert row["max_abs"] == row["relative_l2"] == 0
    for graph in full["graphs"].values():
        for flag in (
            "actual_acl_graph",
            "fullgraph",
            "compiled_raw_equal",
            "topk_poison_replay_restored",
            "live_input_ABA",
        ):
            assert graph[flag], flag
    assert full["packet_checks"]["readonly_after_replay"]
    compilation = json.loads((ROOT / "canonical-compile-receipt.json").read_text())
    for flag in ("compiled", "kernel_abi_pass", "orchestration_compiled", "prepare_compiled"):
        assert compilation[flag], flag
    assert compilation["npu_initialized_after"] is False
    for name in ("CPU_BANK_RECEIPT.json", "CPU_PREFLIGHT_RECEIPT.json"):
        cpu = json.loads((ROOT / name).read_text())
        assert cpu["pass_cpu"] and cpu["npu_initialized"] is False
    return {
        "status": "PASS_SAVED_SINGLE_LAYER_AND_CPU_PREPARATION",
        "qa_inputs_exact": 3,
        "full_csa_boundaries_exact_per_input": 8,
        "whole_model_tokens": "not checked by this verifier",
        "performance": "not measured",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
