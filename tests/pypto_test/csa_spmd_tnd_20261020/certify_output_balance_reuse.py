# SPDX-License-Identifier: Apache-2.0
"""Freshly compile both sides before reusing an unchanged output-balance case.

The only accepted source differences are the tested host policy guard and the
post-L1 metadata fixture capacity guard. Kernel mathematics, the timing helper,
loaded tools, and every other snapshotted file must remain unchanged.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_s6_cohort import source_hashes
from prepare_output_work_balance_guard import load_policy
from verify_short_typefix_codegen import normalized


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_manifest(snapshot):
    entries = json.loads((snapshot / "source_manifest.json").read_text())
    assert all(digest(snapshot / name) == value for name, value in entries.items())
    return entries


def verify_post_l1_harness(before, after):
    old = '''    if not b1_only and (capacity != 41 or requests != 12 or fixture["query_lengths"] != [6] * 6):
        raise ValueError("本矩阵入口要求6个S6活动请求、T容量41、请求槽12")
'''
    new = '''    required_requests = 6 + (capacity - 6 + 5) // 6
    if not b1_only and (
        not 41 <= capacity <= 72 or requests != required_requests or fixture["query_lengths"] != [6] * 6
    ):
        raise ValueError("本矩阵需6个S6请求及足够槽位：6 + ceil((T容量 - 6) / 6)，覆盖S1的dummy尾部")
'''
    assert before.count(old) == 1
    expected = before.replace(old, new).replace(
        "# The candidate continues to receive the full T41/B12 capacity.",
        "# The candidate retains the original token/request capacity.",
    )
    assert expected == after, "Unexpected metadata-matrix edits"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("task_ids", nargs="+")
    args = parser.parse_args()
    assert len(set(args.task_ids)) == len(args.task_ids)
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    repo, workspace = here.parents[2], here.parents[2].parent.parent
    canonical = args.canonical_snapshot.resolve()
    current_manifest = checked_manifest(canonical)
    native_path = canonical / "native_toolchain_manifest.json"
    native = json.loads(native_path.read_text())
    assert all(Path(path).is_file() and digest(Path(path)) == value for path, value in native["files"].items())
    current_policy = load_policy(canonical / "spmd/policy.py", "canonical_balance_reuse_policy")
    cleanup = json.loads((here / "output_balance_constant_cleanup.json").read_text())
    jobs = {row["task_id"]: row for row in json.loads((here / "jobs.json").read_text())}
    common = [
        sys.executable, str(repo / "tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py"),
        "--workspace-root", str(workspace),
        "--vllm-root", str(workspace / ".cache/migration-v0.25.1rc1/vllm"),
        "--pypto-root", native["repositories"]["pypto"]["root"],
        "--simpler-root", str(workspace / ".worktrees/simpler-csa-core-args-256-20261016"),
        "--ptoas-root", str(Path(native["repositories"]["ptoas"]["root"]) / "build"),
        str(here / "check_capacity_compile.py"),
    ]
    work, reused = [], {}
    for task_id in args.task_ids:
        job = jobs[task_id]
        assert job["last_observed_status"] == "completed (exit=0)"
        directory = Path(job["output"])
        result = json.loads((directory / "result.json").read_text())
        assert result["passed"] and result["stage"] == "complete"
        assert result["history"] == 131072 and result["batch"] in (1, 16)
        assert result["query_lengths"] == [6] * result["batch"]
        assert result["actual_tokens"] == result["token_capacity"] == 6 * result["batch"]
        assert result["request_capacity"] == result["batch"]
        old_snapshot = directory.with_name(directory.name + "_source")
        old_manifest = checked_manifest(old_snapshot)
        assert old_manifest.keys() == current_manifest.keys()
        changed = {key for key in old_manifest if old_manifest[key] != current_manifest[key]}
        assert changed == {"spmd/policy.py", "spmd/cube_schedule.py", "metadata_matrix.py"}, changed
        cube_after = (canonical / "spmd/cube_schedule.py").read_text()
        assert cube_after.count(cleanup["constants"] + cleanup["anchor"]) == 1
        assert cube_after.count(cleanup["after_block"]) == 1
        cube_restored = cube_after.replace(cleanup["constants"] + cleanup["anchor"], cleanup["anchor"])
        cube_restored = cube_restored.replace(cleanup["after_block"], cleanup["before_block"])
        assert cube_restored == (old_snapshot / "spmd/cube_schedule.py").read_text()
        verify_post_l1_harness((old_snapshot / "metadata_matrix.py").read_text(),
                              (canonical / "metadata_matrix.py").read_text())
        run_probe = (canonical / "run_probe.py").read_text()
        assert run_probe.index("report[\"l1\"]") < run_probe.index("from metadata_matrix import validate")
        assert digest(old_snapshot / "native_toolchain_manifest.json") == digest(native_path)
        kwargs = dict(tokens=result["token_capacity"], requests=result["request_capacity"],
                      max_query_tokens=result["dspark_spec_tokens"] + 1,
                      max_seq_len=result["history"] + result["query_tokens"])
        assert current_policy(**kwargs).constexprs() == result["policy"]
        description = json.loads((directory / "buffer_descriptors.json").read_text())
        assert description["policy"] == result["policy"]
        existing = list((directory / "build_output").glob("decode_csa_spmd_*/ptoas/dynamic_csa_full.cpp"))
        assert len(existing) == 2
        reused[task_id] = dict(batch=result["batch"], history=result["history"],
                               source_manifest_sha256=digest(old_snapshot / "source_manifest.json"), proofs=[])
        for side, package in (("candidate", "spmd"), ("baseline", "original_spmd")):
            old_cpp = [p for p in existing if ("PTOAS__WAIT_GE_BACKOFF<150>" in p.read_text()) == (side == "candidate")]
            assert len(old_cpp) == 1
            descriptor = json.loads(json.dumps(description))
            if side == "baseline":
                del descriptor["policy"]["OUTPUT_WORK_BALANCE"]
            descriptor_path = args.output / f"b{result['batch']}_{side}_descriptors.json"
            descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n")
            work.append(dict(task_id=task_id, batch=result["batch"], side=side,
                             old_cpp=old_cpp[0], descriptor=descriptor_path, source=canonical / package))

    def compile_case(item):
        label = f"b{item['batch']}_{item['side']}"
        output = args.output / label
        command = common + ["--source", str(item["source"]), "--descriptors", str(item["descriptor"]),
                            "--output", str(output)]
        with (args.output / f"{label}.log").open("w") as log:
            process = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        new_cpp = output / "kernel/ptoas/dynamic_csa_full.cpp"
        passed = process.returncode == 0 and new_cpp.exists() and normalized(item["old_cpp"]) == normalized(new_cpp)
        row = dict(task_id=item["task_id"], batch=item["batch"], side=item["side"],
                   returncode=process.returncode, passed=passed, command=command,
                   old_cpp=str(item["old_cpp"]), new_cpp=str(new_cpp),
                   old_cpp_sha256=digest(item["old_cpp"]),
                   new_cpp_sha256=digest(new_cpp) if new_cpp.exists() else None)
        if passed:
            old_host = item["old_cpp"].parent.parent / "orchestration/decode_csa_spmd.cpp"
            new_host = new_cpp.parent.parent / "orchestration/decode_csa_spmd.cpp"
            row["orchestration"] = dict(old_path=str(old_host), new_path=str(new_host),
                                        old_sha256=digest(old_host), new_sha256=digest(new_host))
            row["passed"] = old_host.read_bytes() == new_host.read_bytes()
        (args.output / f"{label}_comparison.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in ("batch", "side", "returncode", "passed")}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(compile_case, work))
    assert all(row["passed"] for row in rows), "Fresh code generation differs; do not reuse measurements"
    for row in rows:
        reused[row["task_id"]]["proofs"].append(row)
    certificate = dict(
        passed=True, scope="Exact executable-code and post-L1-only fixture guard equivalence; no new performance claim",
        canonical_snapshot=str(canonical), native_toolchain_sha256=digest(native_path), reused=reused,
        normalization="Only full // pto: debug-name comment lines removed",
        canonical_fields={
            "candidate": source_hashes(canonical, "spmd/"),
            "baseline": source_hashes(canonical, "original_spmd/"),
            "harness": {key: value for key, value in current_manifest.items()
                        if not key.startswith(("spmd/", "original_spmd/"))},
        },
        policy_outputs_equal=True, only_post_l1_fixture_guard_and_comment_changed=True,
        cube_changes_only_named_constants=True,
    )
    (args.output / "certificate.json").write_text(json.dumps(certificate, indent=2) + "\n")
    print(json.dumps({"passed": True, "reused_cases": len(reused), "fresh_cpp_matches": len(rows)}))


if __name__ == "__main__":
    main()
