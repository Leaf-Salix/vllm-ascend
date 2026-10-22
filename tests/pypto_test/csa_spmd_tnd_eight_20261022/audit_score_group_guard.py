# SPDX-License-Identifier: Apache-2.0
"""Prove enabled measured C08 and all disabled generated-code classes for v3."""

import hashlib
import json
from pathlib import Path

CPP_FILES = (
    "ptoas/dynamic_csa_full.cpp",
    "kernels/aic/cube_schedule.cpp",
    "kernels/aiv/vector_schedule.cpp",
    "orchestration/decode_csa_spmd.cpp",
)


def content(root):
    data = {
        name: "\n".join(
            line for line in (root / name).read_text().splitlines() if not line.lstrip().startswith("// pto:")
        )
        for name in CPP_FILES
    }
    data["kernel_abi"] = json.loads((root / "compiled_meta.json").read_text())["kernel_abi"]
    return data


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    baseline = json.loads((here / "baseline.json").read_text())
    classes = json.loads((here / "score_group_guard_baseline_classes.json").read_text())["classes"]
    rows = {}
    for expected_hash, keys in classes.items():
        representatives = [
            key for key in keys if (here / f"runs/compile_score_group_order_{key}_v3/result.json").exists()
        ]
        assert representatives
        representative = representatives[0]
        compiled = here / f"runs/compile_score_group_order_{representative}_v3"
        assert json.loads((compiled / "result.json").read_text())["full_codegen_passed"]
        actual = content(compiled / "kernel")
        assert fingerprint(actual) == expected_hash, representative
        for key in keys:
            (historical,) = Path(baseline["cases"][key]["source_report"]).parent.glob("build_output/decode_csa_spmd_*")
            assert actual == content(historical), key
            policy = json.loads((here / f"descriptors/{key}_score_group_order_v3.json").read_text())["policy"]
            assert policy.pop("SCORE_GROUP_ORDER") is False
            assert policy == json.loads(Path(baseline["cases"][key]["source_report"]).read_text())["policies"]["spmd"]
            rows[key] = dict(
                passed=True,
                enabled=False,
                compiled_representative=representative,
                compile=str(compiled),
                historical=str(historical),
                executable_contract_sha256=expected_hash,
                code_equivalence_basis="Same generated source, packed wrappers, orchestration and kernel ABI class",
            )
    compiled = here / "runs/compile_score_group_order_C08_v3"
    assert json.loads((compiled / "result.json").read_text())["full_codegen_passed"]
    actual = content(compiled / "kernel")
    measured = here / "runs/h131072_b16_mixed_score_group_order_v2"
    report = json.loads((measured / "result.json").read_text())
    assert report["passed"] and len(report["windows"]) == 10
    matching = [p for p in (measured / "build_output").glob("decode_csa_spmd_*") if content(p) == actual]
    assert len(matching) == 1
    rows["C08"] = dict(
        passed=True,
        enabled=True,
        compile=str(compiled),
        measured_artifact=str(matching[0]),
        source_report=str(measured / "result.json"),
        executable_contract_sha256=fingerprint(actual),
        code_equivalence_basis="Full executable-source/ABI match to the actually measured v2 candidate",
    )
    assert len(rows) == 8
    output = dict(
        passed=True,
        device_execution=False,
        candidate="score_group_order_v3",
        scope="Generated-code equivalence only; not authorization to reuse cross-SDK performance",
        cross_sdk_performance_reuse_permitted=False,
        cases=rows,
        normalization="Remove only full-line // pto: location comments from C++; compare kernel ABI as JSON",
        limitations="No ELF identity claim; metadata transitions and service integration still require validation",
    )
    with (here / "score_group_order_v3_codegen.json").open("x") as stream:
        stream.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(passed=True, matched_cases=list(rows), disabled_classes=len(classes))))


if __name__ == "__main__":
    main()
