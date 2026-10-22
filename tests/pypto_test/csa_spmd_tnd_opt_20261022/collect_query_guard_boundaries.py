# SPDX-License-Identifier: Apache-2.0
"""Collect specific non-cohort Query boundaries without relaxing the nine-case gate."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import median

from audit_query_head_guard import normalized
from collect_trials import REFERENCE_KIND, shared_toolchain_fingerprint, source_hashes

BOUNDARIES = {
    (4,) * 16: False,
    (5,) * 16: False,
    (6, 2) + (4,) * 14: True,
    (6, 4) + (5,) * 14: True,
    (6,) * 12: True,
}


def measurement(result):
    queries = tuple(result["query_lengths"])
    assert queries in BOUNDARIES
    enabled = BOUNDARIES[queries]
    assert result["policy"]["QUERY_CONTIGUOUS_HEADS"] is enabled
    assert result["policy"]["QUERY_GROUP"] == (6 if enabled else 2)
    assert result["history"] == 131072 and result["batch"] == result["request_capacity"] == len(queries)
    assert result["history_lengths"] == [131072] * len(queries)
    assert result["token_capacity"] == result["actual_tokens"] == sum(queries)
    assert not result.get("constant_indexer_history") and not result.get("wait_policy_experiment")
    assert not result.get("phase_clock") and not result["comparison_reference"].get("diagnostic_phase_clock")
    audit = result["bsh_wait_policy_audit"]
    assert audit["passed"] and audit["ticks"] == 150
    sources = audit["actual_ccec_sources"]
    assert len(sources) >= 4 and all(s["passed"] and s["native_lowering"] and not s["legacy_rewrite"] for s in sources)
    timing = result["l1"]
    assert timing["passed"] and not timing["shared_workspace"]
    assert all(timing["protocol"].get(side) for side in ("spmd_tnd", "spmd_original"))
    windows = timing["windows"]
    assert len(windows) >= 10 and len(windows) % 2 == 0
    for index, window in enumerate(windows):
        order = ("spmd_tnd", "spmd_original") if (index // 2) % 2 == 0 else ("spmd_original", "spmd_tnd")
        assert window["side"] == order[index % 2]
        assert window["core_records"] == window["cores"] == 72 and window["task_count"] == 1
        assert window["clock_freq_hz"] == 50000000
        assert math.isfinite(window["incore_span_us"]) and window["incore_span_us"] > 0
    samples = {
        side: [w["incore_span_us"] for w in windows if w["side"] == side] for side in ("spmd_original", "spmd_tnd")
    }
    baseline, candidate = (median(samples[side]) for side in ("spmd_original", "spmd_tnd"))
    return dict(
        enabled=enabled,
        classification="active boundary" if enabled else "disabled control",
        history=131072,
        batch=len(queries),
        query_lengths=list(queries),
        token_capacity=sum(queries),
        query_group=result["policy"]["QUERY_GROUP"],
        reference_v122_us=baseline,
        candidate_us=candidate,
        reference_samples_us=samples["spmd_original"],
        candidate_samples_us=samples["spmd_tnd"],
        latency_change_pct=(candidate / baseline - 1) * 100,
        paired_faster=sum(a < b for a, b in zip(samples["spmd_tnd"], samples["spmd_original"])),
        native_precision_exact=result["precision_exact"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and len(args.task_ids) == len(set(args.task_ids))
    here = Path(__file__).resolve().parent
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    candidate = json.loads((here / "query_head_guard.json").read_text())["files"]
    jobs_path = here.parent / "csa_spmd_tnd_20261020/jobs.json"
    jobs = json.loads(jobs_path.read_text())
    by_id = {job["task_id"]: job for job in jobs}
    rows = []
    for task in args.task_ids:
        job = by_id[task]
        status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
        assert status == "completed (exit=0)", (task, status)
        job["last_observed_status"] = status
        output = Path(job["output"])
        snapshot = output.with_name(output.name + "_source")
        path = output / "result.json"
        result = json.loads(path.read_text())
        assert result["stage"] == "complete" and result["passed"] and result["functional_passed"]
        assert result["device_execution"] and result["comparison_reference"]["kind"] == REFERENCE_KIND
        assert result["comparison_reference"]["files"] == expected
        assert source_hashes(snapshot, "original_spmd/") == expected
        assert source_hashes(snapshot, "spmd/") == candidate
        sdk = shared_toolchain_fingerprint(snapshot, result)
        assert sdk is not None
        row = measurement(result)
        roots = {}
        for label in ("spmd", "original_spmd"):
            pto = [
                p
                for p in (output / "build_output").glob("*/ptoas/dynamic_csa_full.pto")
                if str(snapshot / label) + "/" in p.read_text()
            ]
            assert len(pto) == 1
            roots[label] = pto[0].parent.parent
        actual_pto = roots["spmd"] / "ptoas/dynamic_csa_full.pto"
        assert ("q_head_input_" in actual_pto.read_text()) == row["enabled"]
        if not row["enabled"]:
            equal = {
                rel: normalized(roots["spmd"] / rel) == normalized(roots["original_spmd"] / rel)
                for rel in (
                    "ptoas/dynamic_csa_full.cpp",
                    "kernels/aic/cube_schedule.cpp",
                    "kernels/aiv/vector_schedule.cpp",
                    "orchestration/decode_csa_spmd.cpp",
                    "compiled_meta.json",
                )
            }
            assert all(equal.values()), equal
            row["disabled_generated_code_equal"] = equal
        row.update(
            task_id=task,
            output=str(output),
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            passed=True,
            shared_toolchain=sdk,
            actual_kernel_roots={k: str(v) for k, v in roots.items()},
        )
        rows.append(row)
    report = dict(
        reference_kind=REFERENCE_KIND,
        formal_gate_eligible=False,
        formal_nine_case_gate_passed=False,
        scope="Explicit non-cohort boundaries; disabled-control timings are not algorithm changes",
        cases=rows,
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    jobs_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n")
    for row in rows:
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "task_id",
                        "classification",
                        "reference_v122_us",
                        "candidate_us",
                        "latency_change_pct",
                        "paired_faster",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
