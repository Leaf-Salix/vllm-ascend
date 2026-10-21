# SPDX-License-Identifier: Apache-2.0
"""Certify reuse of six measurements after an inactive-branch type correction."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_s6_cohort import source_hashes
from verify_short_typefix_codegen import normalized


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(snapshot):
    entries = json.loads((snapshot / "source_manifest.json").read_text())
    assert all(digest(snapshot / name) == expected for name, expected in entries.items())
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--canonical-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    proof = json.loads(args.proof.read_text())
    assert proof["passed"] and len(proof["rows"]) == 12
    canonical = args.canonical_snapshot.resolve()
    new_entries = manifest(canonical)
    canonical_fields = {
        "candidate": source_hashes(canonical, "spmd/"),
        "baseline": source_hashes(canonical, "original_spmd/"),
        "harness": {key: value for key, value in new_entries.items()
                    if not key.startswith(("spmd/", "original_spmd/"))},
    }
    native_hash = digest(canonical / "native_toolchain_manifest.json")
    assert native_hash == proof["native_toolchain_sha256"]
    jobs = {r["task_id"]: r for r in json.loads((here / "jobs.json").read_text())}
    reused = {}
    for row in proof["rows"]:
        assert row["passed"] and row["returncode"] == 0
        old_cpp, new_cpp = Path(row["old_cpp"]), Path(row["new_cpp"])
        assert digest(old_cpp) == row["old_cpp_sha256"] and digest(new_cpp) == row["new_cpp_sha256"]
        assert normalized(old_cpp) == normalized(new_cpp)
        task = jobs[row["task_id"]]
        directory = Path(task["output"])
        snapshot = directory.with_name(directory.name + "_source")
        old_entries = manifest(snapshot)
        assert set(old_entries) == set(new_entries)
        assert digest(snapshot / "native_toolchain_manifest.json") == native_hash
        changed = {name for name in old_entries if old_entries[name] != new_entries[name]}
        expected = {"spmd/indexer_score_stages.py", "original_spmd/indexer_score_stages.py",
                    "launcher.sh", "bsh_baseline_manifest.json"}
        assert changed == expected, changed
        for prefix in ("spmd", "original_spmd"):
            old = (snapshot / prefix / "indexer_score_stages.py").read_text()
            marker = "[TOPK_CANDIDATES_PER_LEAF // (2 * score_tile), score_tile]"
            replacement = "[TOPK_CANDIDATES_PER_LEAF // (2 * BUFFERED_LONG_SCORE_TILE), BUFFERED_LONG_SCORE_TILE]"
            assert old.count(marker) == 2
            assert old.replace(marker, replacement) == (canonical / prefix / "indexer_score_stages.py").read_text()
        launcher = (snapshot / "launcher.sh").read_text()
        assert launcher.count("bsh_baseline_a241fa61_manifest.json") == 1
        corrected_launcher = launcher.replace(
            "bsh_baseline_a241fa61_manifest.json", "bsh_baseline_a241fa61_short_typefix_manifest.json"
        )
        assert corrected_launcher == (canonical / "launcher.sh").read_text()
        old_baseline = json.loads((snapshot / "bsh_baseline_manifest.json").read_text())
        new_baseline = json.loads((canonical / "bsh_baseline_manifest.json").read_text())
        assert old_baseline.keys() == new_baseline.keys()
        assert all(old_baseline[key] == new_baseline[key] for key in old_baseline if key != "bsh")
        old_bsh, new_bsh = old_baseline["bsh"], new_baseline["bsh"]
        changed_fields = {key for key in old_bsh.keys() | new_bsh.keys() if old_bsh.get(key) != new_bsh.get(key)}
        assert changed_fields == {"snapshot", "files", "local_compile_fix"}
        assert old_bsh["files"].keys() == new_bsh["files"].keys()
        assert {key for key in old_bsh["files"] if old_bsh["files"][key] != new_bsh["files"][key]} == {
            "deepseek_v4_flash_csa_spmd/indexer_score_stages.py"
        }
        item = reused.setdefault(row["task_id"], {"batch": row["batch"], "history": 131072,
                                 "source_manifest_sha256": digest(snapshot / "source_manifest.json"), "proofs": []})
        item["proofs"].append(row)
    assert len(reused) == 6 and all({r["side"] for r in item["proofs"]} == {"candidate", "baseline"}
                                   for item in reused.values())
    result = dict(passed=True, scope="Code equivalence only; no performance or short-case correctness claim",
                  proof_sha256=digest(args.proof), canonical_snapshot=str(canonical),
                  canonical_fields=canonical_fields, native_toolchain_sha256=native_hash, reused=reused,
                  normalization="Only full // pto: debug-name comment lines removed")
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "reused_long_cases": len(reused), "executable_cpp_matches": 12}))


if __name__ == "__main__":
    main()
