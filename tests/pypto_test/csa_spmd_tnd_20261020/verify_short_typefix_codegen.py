# SPDX-License-Identifier: Apache-2.0
"""Prove the short-branch type fix leaves completed long-case kernels unchanged.

Only full // pto: debug-name comment lines are removed from generated C++.
No SSA names, operations, constants, addresses or synchronization are removed.
Two CPU workers compile fresh meta-tensor artifacts without allocating an NPU.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(path):
    return "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("// pto:"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    workspace = repo.parent.parent
    assert (workspace / "env-dsv4-0251rc1.sh").exists()
    cohort = json.loads((here / "cohort_prefix_a241_v41.json").read_text())
    native = json.loads((here / "native_toolchain_manifest.json").read_text())
    pypto_root = native["repositories"]["pypto"]["root"]
    ptoas_root = str(Path(native["repositories"]["ptoas"]["root"]) / "build")
    sources = {
        "candidate": here / "runs/prefix_policy_short_typefix_native_candidate_v3",
        "baseline": here / "bsh_baseline_a241fa61_short_typefix/deepseek_v4_flash_csa_spmd",
    }
    common = [
        sys.executable, str(repo / "tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py"),
        "--workspace-root", str(workspace),
        "--vllm-root", str(workspace / ".cache/migration-v0.25.1rc1/vllm"),
        "--pypto-root", pypto_root,
        "--simpler-root", str(workspace / ".worktrees/simpler-csa-core-args-256-20261016"),
        "--ptoas-root", ptoas_root, str(here / "check_capacity_compile.py"),
    ]
    work = []
    for task in cohort["tasks"]:
        arguments = task["arguments"]
        history = int(arguments[arguments.index("--history") + 1])
        if history != 131072:
            continue
        batch = int(arguments[arguments.index("--batch") + 1])
        directory = Path(task["output"])
        files = list((directory / "build_output").glob("decode_csa_spmd_*/ptoas/dynamic_csa_full.cpp"))
        assert len(files) == 2, files
        for side, source in sources.items():
            matches = [f for f in files if ("PTOAS__WAIT_GE_BACKOFF<150>" in f.read_text()) == (side == "candidate")]
            assert len(matches) == 1, (batch, side)
            work.append(dict(task_id=task["task_id"], batch=batch, side=side, source=source,
                             old_cpp=matches[0], descriptors=directory / "buffer_descriptors.json"))
    assert len(work) == 12

    def run(item):
        label = f"b{item['batch']}_{item['side']}"
        output = args.output / label
        command = common + ["--source", str(item["source"]), "--descriptors", str(item["descriptors"]),
                            "--output", str(output)]
        with (args.output / f"{label}.log").open("w") as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        row = {key: item[key] for key in ("task_id", "batch", "side")}
        row.update(command=command, returncode=result.returncode, old_cpp=str(item["old_cpp"]),
                   descriptors_sha256=digest(item["descriptors"]))
        new_cpp = output / "kernel/ptoas/dynamic_csa_full.cpp"
        row["new_cpp"] = str(new_cpp)
        row["passed"] = False
        if result.returncode == 0 and new_cpp.exists():
            old, new = normalized(item["old_cpp"]), normalized(new_cpp)
            row.update(passed=old == new, old_cpp_sha256=digest(item["old_cpp"]), new_cpp_sha256=digest(new_cpp),
                       normalized_old_sha256=hashlib.sha256(old.encode()).hexdigest(),
                       normalized_new_sha256=hashlib.sha256(new.encode()).hexdigest())
        (args.output / f"{label}_comparison.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in ("batch", "side", "passed", "returncode")}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run, work))
    report = dict(passed=all(row["passed"] for row in rows), device_execution=False,
                  normalization="Only full // pto: debug-name comment lines removed", rows=rows,
                  native_toolchain_sha256=digest(here / "native_toolchain_manifest.json"))
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
