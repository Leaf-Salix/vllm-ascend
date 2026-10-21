# SPDX-License-Identifier: Apache-2.0
"""Evaluate eight explicitly selected paired S6 runs against the frozen BSH."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

EXPECTED_CASES = frozenset(
    [(131072, batch) for batch in (1, 4, 8, 16, 24, 28)] + [(8192, batch) for batch in (16, 32)]
)


def source_hashes(snapshot, prefix):
    manifest = json.loads((snapshot / "source_manifest.json").read_text())
    for relative, expected in manifest.items():
        path = snapshot / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {path}")
    return {key.removeprefix(prefix): value for key, value in manifest.items() if key.startswith(prefix)}


def apply_codegen_certificate(fingerprint, task_id, snapshot, result, certificate):
    """Accept source differences only with the recorded exact executable-C++ proof."""
    if certificate is None or task_id not in certificate["reused"]:
        return fingerprint
    item = certificate["reused"][task_id]
    if not certificate["passed"] or certificate["normalization"] != (
        "Only full // pto: debug-name comment lines removed"
    ):
        raise ValueError("Unrecognized or failed code equivalence certificate")
    if (result["history"], result["batch"]) != (item["history"], item["batch"]):
        raise ValueError(f"{task_id}: code equivalence case mismatch")
    if hashlib.sha256((snapshot / "source_manifest.json").read_bytes()).hexdigest() != item["source_manifest_sha256"]:
        raise ValueError(f"{task_id}: source manifest changed after code equivalence proof")
    if fingerprint["native_toolchain"] != certificate["native_toolchain_sha256"]:
        raise ValueError(f"{task_id}: code equivalence used another toolchain")
    if {proof["side"] for proof in item["proofs"]} != {"candidate", "baseline"}:
        raise ValueError(f"{task_id}: both sides require code equivalence")
    for proof in item["proofs"]:
        normalized = []
        for side in ("old", "new"):
            path = Path(proof[f"{side}_cpp"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != proof[f"{side}_cpp_sha256"]:
                raise ValueError(f"{task_id}: generated C++ changed after proof")
            normalized.append("\n".join(
                line for line in path.read_text().splitlines() if not line.lstrip().startswith("// pto:")
            ))
        if normalized[0] != normalized[1]:
            raise ValueError(f"{task_id}: generated executable code differs")
        if proof.get("orchestration"):
            orchestration = proof["orchestration"]
            contents = []
            for side in ("old", "new"):
                path = Path(orchestration[f"{side}_path"])
                value = path.read_bytes()
                if hashlib.sha256(value).hexdigest() != orchestration[f"{side}_sha256"]:
                    raise ValueError(f"{task_id}: host orchestration changed after proof")
                contents.append(value)
            if contents[0] != contents[1]:
                raise ValueError(f"{task_id}: host argument binding differs")
    canonical = certificate["canonical_fields"]
    if set(canonical) != {"candidate", "baseline", "harness"}:
        raise ValueError("Certificate cannot override runtime or measurement fields")
    return dict(fingerprint, **canonical)


def evaluate(jobs, selected, certificate=None):
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError("Select exactly eight distinct task IDs from one frozen candidate")
    records = {job["task_id"]: job for job in jobs}
    rows = []
    fingerprints = []
    for task_id in selected:
        job = records[task_id]
        directory = Path(job["output"])
        result = json.loads((directory / "result.json").read_text())
        snapshot = directory.with_name(directory.name + "_source")
        if result.get("comparison_reference") or (snapshot / "comparison_reference.json").exists():
            raise ValueError(f"{task_id}: incremental TND comparison cannot replace the BSH baseline")
        if not (
            job["last_observed_status"] == "completed (exit=0)"
            and result["stage"] == "complete"
            and result["passed"]
            and result.get("bsh_wait_policy_audit", {}).get("passed")
            and result["bsh_wait_policy_audit"]["ticks"] == 150
        ):
            raise ValueError(f"{task_id}: missing completed BSH150 correctness/codegen evidence")
        batch, history = result["batch"], result["history"]
        if not (
            result["query_lengths"] == [6] * batch
            and result["history_lengths"] == [history] * batch
            and result["token_capacity"] == 6 * batch
            and result["actual_tokens"] == 6 * batch
            and result["request_capacity"] == batch
            and not result.get("constant_indexer_history", False)
            and not result.get("wait_policy_experiment")
        ):
            raise ValueError(f"{task_id}: not the required uniform S6 case and capacity")
        timing = result["l1"]
        if not timing["passed"] or timing.get("shared_workspace"):
            raise ValueError(f"{task_id}: invalid or diagnostic-only L1 measurement")
        values = {}
        for side in ("spmd_original", "spmd_tnd"):
            windows = [row for row in timing["windows"] if row["side"] == side]
            if len(windows) < 5 or any(
                row["core_records"] != 72 or row["cores"] != 72 or row["task_count"] != 1 for row in windows
            ):
                raise ValueError(f"{task_id}: incomplete full-core L1 windows")
            samples = [row["incore_span_us"] for row in windows]
            if any(value <= 0 or not math.isfinite(value) for value in samples):
                raise ValueError(f"{task_id}: invalid L1 duration")
            values[side] = {"median_us": median(samples), "samples_us": samples}
        baseline = values["spmd_original"]["median_us"]
        candidate = values["spmd_tnd"]["median_us"]
        rows.append(
            dict(
                history=history,
                batch=batch,
                task_id=task_id,
                bsh_us=baseline,
                tnd_us=candidate,
                ratio=candidate / baseline,
                latency_change_pct=(candidate / baseline - 1) * 100,
                windows=values,
            )
        )
        snapshot = directory.with_name(directory.name + "_source")
        native_manifest = snapshot / "native_toolchain_manifest.json"
        all_sources = json.loads((snapshot / "source_manifest.json").read_text())
        fingerprint = {
            "candidate": source_hashes(snapshot, "spmd/"),
            "baseline": source_hashes(snapshot, "original_spmd/"),
            "native_toolchain": (
                hashlib.sha256(native_manifest.read_bytes()).hexdigest() if native_manifest.exists() else None
            ),
            "runtime_artifacts": result["runtime_artifacts"],
            "harness": {
                key: value
                for key, value in all_sources.items()
                if not key.startswith(("spmd/", "original_spmd/"))
            },
        }
        if not fingerprint["candidate"] or not fingerprint["baseline"]:
            raise ValueError(f"{task_id}: missing source fingerprint")
        if native_manifest.exists() and not result.get("toolchain_post_run_audit", {}).get("passed"):
            raise ValueError(f"{task_id}: missing native toolchain post-run audit")
        fingerprints.append(apply_codegen_certificate(fingerprint, task_id, snapshot, result, certificate))
    if {(row["history"], row["batch"]) for row in rows} != EXPECTED_CASES:
        raise ValueError("The selected runs do not cover the required eight cases")
    if any(item != fingerprints[0] for item in fingerprints[1:]):
        raise ValueError("Candidate, baseline or loaded toolchain differs across the selected runs")
    geometric_mean = math.exp(sum(math.log(row["ratio"]) for row in rows) / len(rows))
    return {
        "scope": "Performance gate only; the separate TND/service/state acceptance still applies",
        "metric": "Equal-weight geometric mean of TND/BSH full-chain L1 median latency ratios",
        "threshold": 0.90,
        "geometric_mean_ratio": geometric_mean,
        "passed": geometric_mean <= 0.90,
        "cases": sorted(rows, key=lambda row: (-row["history"], row["batch"])),
        "fingerprint": fingerprints[0],
        "reused_via_exact_codegen_proof": (
            sorted(set(selected) & set(certificate["reused"])) if certificate else []
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codegen-reuse-certificate", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    certificate = (
        json.loads(args.codegen_reuse_certificate.read_text()) if args.codegen_reuse_certificate else None
    )
    result = evaluate(json.loads((here / "jobs.json").read_text()), args.task_ids, certificate)
    if args.codegen_reuse_certificate:
        result["codegen_reuse_certificate_sha256"] = hashlib.sha256(
            args.codegen_reuse_certificate.read_bytes()
        ).hexdigest()
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "geometric_mean_ratio", "threshold")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
