# SPDX-License-Identifier: Apache-2.0
"""Reuse six unchanged paired runs after complete candidate-code comparisons."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from audit_borrowed_score_stream import normalized_cpp
from certify_output_balance_reuse import checked_manifest, verify_post_l1_harness
from evaluate_s6_cohort import source_hashes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def algorithm(path):
    tree = ast.parse(path.read_text())
    if (isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        tree.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def check_optional_guard(before, after):
    """Only the disabled HC-consumer poison check may differ in run_probe."""
    trees = [ast.parse(path.read_text()) for path in (before, after)]
    counts = []
    for tree in trees:
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "workspace_padding_checks")
        extras = [
            n for n in function.body
            if isinstance(n, ast.If) and ast.unparse(n.test) == "entry.policy.hc_consumer_reduce"
        ]
        counts.append(len(extras))
        for node in extras:
            function.body.remove(node)
    assert counts == [0, 1]
    assert ast.dump(trees[0], include_attributes=False) == ast.dump(trees[1], include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    canonical = args.canonical_snapshot.resolve()
    current = checked_manifest(canonical)
    native = canonical / "native_toolchain_manifest.json"
    assert all(digest(Path(p)) == value for p, value in json.loads(native.read_text())["files"].items())
    cleanup = json.loads((here / "borrowed_combined_cleanup_v106.json").read_text())
    plan = json.loads((here / "borrowed_combined_compile_plan_v105.json").read_text())
    audit = json.loads((here / "borrowed_combined_codegen_audit_v105.json").read_text())
    assert audit["passed"]
    for name, value in cleanup["source_manifest"].items():
        assert digest(canonical / "spmd" / name) == value
        assert algorithm(canonical / "spmd" / name) == algorithm(Path(plan["source"]) / name)
    for name, value in plan["source_manifest"].items():
        assert digest(Path(plan["source"]) / name) == value
    jobs = {r["task_id"]: r for r in json.loads((here / "jobs.json").read_text())}
    cohort = json.loads((here / "cohort_output_balance_v4.json").read_text())
    reused = {}
    for case in cohort["cases"]:
        key = case["history"], case["batch"]
        if key in ((131072, 8), (8192, 16)):
            continue
        job = jobs[case["task_id"]]
        assert job["last_observed_status"] == "completed (exit=0)"
        directory = Path(job["output"])
        result = json.loads((directory / "result.json").read_text())
        assert result["passed"] and result["l1"]["passed"]
        assert not result["policy"]["HC_CONSUMER_REDUCE"]
        old = directory.with_name(directory.name + "_source")
        old_manifest = checked_manifest(old)
        assert source_hashes(old, "original_spmd/") == source_hashes(canonical, "original_spmd/")
        assert digest(old / "native_toolchain_manifest.json") == digest(native)
        old_harness = {k: v for k, v in old_manifest.items() if not k.startswith(("spmd/", "original_spmd/"))}
        new_harness = {k: v for k, v in current.items() if not k.startswith(("spmd/", "original_spmd/"))}
        assert old_harness.keys() == new_harness.keys()
        changes = {k for k in old_harness if old_harness[k] != new_harness[k]}
        assert changes <= {"run_probe.py", "metadata_matrix.py"}
        if "run_probe.py" in changes:
            check_optional_guard(old / "run_probe.py", canonical / "run_probe.py")
        if "metadata_matrix.py" in changes:
            verify_post_l1_harness(
                (old / "metadata_matrix.py").read_text(), (canonical / "metadata_matrix.py").read_text()
            )
        proof = next(p for p in audit["proofs"] if (p["history"], p["batch"]) == key)
        assert not proof["topk_prefix"] and not proof["short_half_ub"]
        assert proof["whole_cpp_equal"] and proof["host_byte_equal"]
        for side in ("old", "new"):
            assert digest(Path(proof[f"{side}_cpp"])) == proof[f"{side}_cpp_sha256"]
        assert normalized_cpp(Path(proof["old_cpp"])) == normalized_cpp(Path(proof["new_cpp"]))
        candidate = dict(proof, side="candidate")
        baseline_files = [p for p in (directory / "build_output").glob("*/ptoas/dynamic_csa_full.cpp")
                          if "PTOAS__WAIT_GE_BACKOFF<150>" not in p.read_text()]
        assert len(baseline_files) == 1
        baseline_cpp = baseline_files[0]
        host = baseline_cpp.parents[1] / "orchestration/decode_csa_spmd.cpp"
        # Byte-identical frozen baseline source/toolchain: use its existing
        # compiled executable, without recompiling an unchanged baseline.
        baseline = dict(side="baseline", old_cpp=str(baseline_cpp), new_cpp=str(baseline_cpp),
                        old_cpp_sha256=digest(baseline_cpp), new_cpp_sha256=digest(baseline_cpp),
                        orchestration=dict(old_path=str(host), new_path=str(host),
                                           old_sha256=digest(host), new_sha256=digest(host)),
                        unchanged_frozen_source=True)
        reused[case["task_id"]] = dict(
            history=key[0], batch=key[1], source_manifest_sha256=digest(old / "source_manifest.json"),
            proofs=[candidate, baseline], harness_changes=sorted(changes),
            harness_changes_outside_timing_and_disabled=True,
        )
    assert len(reused) == 6
    report = dict(
        passed=True, canonical_snapshot=str(canonical), reused=reused,
        normalization="Only full // pto: debug-name comment lines removed",
        native_toolchain_sha256=digest(native),
        canonical_fields=dict(candidate=source_hashes(canonical, "spmd/"),
                              baseline=source_hashes(canonical, "original_spmd/"),
                              harness=new_harness),
        source_cleanup_ast_equal=True,
        scope=("Six unchanged executable/Host paths; complete paired samples retained, "
               "no fastest-window selection"),
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "reused_cases": len(reused)}))


if __name__ == "__main__":
    main()
