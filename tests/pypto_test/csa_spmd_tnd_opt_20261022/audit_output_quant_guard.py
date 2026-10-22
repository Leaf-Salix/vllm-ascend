# SPDX-License-Identifier: Apache-2.0
"""Compare full guarded generated programs to their actually measured sources."""

import hashlib
import json
from pathlib import Path

from audit_query_head_guard import normalized, sha


def main():
    here = Path(__file__).resolve().parent
    cases = []
    for mode, trial, source_label, manifest, batch in (
        ("enabled", "output_quant_reuse_b1_matrix", "spmd", "output_quant_reuse.json", 1),
        ("disabled", "output_quant_reuse_b16", "original_spmd", "frozen_v122.json", 16),
    ):
        job = here / "runs" / trial
        result = json.loads((job / "result.json").read_text())
        assert result["stage"] == "complete" and result["passed"] and result["functional_passed"]
        assert result["comparison_reference"]["kind"] == "frozen_tnd_v122"
        expected = json.loads((here / manifest).read_text())["files"]
        source = job.with_name(job.name + "_source") / source_label
        assert {p.name: sha(p) for p in source.glob("*.py")} == expected
        pto_files = [
            p for p in (job / "build_output").glob("*/ptoas/dynamic_csa_full.pto") if str(source) + "/" in p.read_text()
        ]
        assert len(pto_files) == 1
        old = pto_files[0].parent.parent
        new_root = here / "runs" / f"compile_output_quant_guard_b{batch}"
        assert json.loads((new_root / "result.json").read_text())["full_codegen_passed"]
        new = new_root / "kernel"
        artifacts = []
        for relative in (
            "ptoas/dynamic_csa_full.cpp",
            "kernels/aic/cube_schedule.cpp",
            "kernels/aiv/vector_schedule.cpp",
            "orchestration/decode_csa_spmd.cpp",
            "compiled_meta.json",
        ):
            left, right = old / relative, new / relative
            a, b = normalized(left), normalized(right)
            artifacts.append(
                dict(
                    relative=relative,
                    identical=a == b,
                    old=str(left),
                    new=str(right),
                    old_sha256=sha(left),
                    new_sha256=sha(right),
                    normalized_sha256=hashlib.sha256(b.encode()).hexdigest(),
                )
            )
        cases.append(
            dict(
                mode=mode,
                passed=all(a["identical"] for a in artifacts),
                trial=str(job),
                source_verified=str(source),
                artifacts=artifacts,
            )
        )
    report = dict(
        passed=all(c["passed"] for c in cases),
        cases=cases,
        proof="Full C++ with only pto location comments removed; exact packed wrappers/host/meta",
        final_nine_case_gate_passed=False,
        limitations="No ELF identity or unmeasured-case performance claim; not production adoption",
    )
    with (here / "output_quant_guard_codegen_audit.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"passed": report["passed"], "cases": [{"mode": c["mode"], "passed": c["passed"]} for c in cases]})
    )
    assert report["passed"]


if __name__ == "__main__":
    main()
