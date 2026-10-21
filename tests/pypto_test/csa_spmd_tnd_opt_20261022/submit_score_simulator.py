# SPDX-License-Identifier: Apache-2.0
"""Freeze the diagnostic sources/tools before submitting one simulator job."""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def submit(job, here):
    manifest = json.loads((job / "job_manifest.json").read_text())
    for relative, expected in manifest["immutable_files"].items():
        assert hashlib.sha256((job / relative).read_bytes()).hexdigest() == expected, relative
    path = here / "diagnostic_jobs.json"
    jobs = json.loads(path.read_text()) if path.exists() else []
    assert not any(j["output"] == str(job) for j in jobs), "This frozen job already has a task handle"
    command = shlex.join(["bash", str(job / "launch.sh")])
    task = subprocess.check_output(
        ["task-submit", "--device", "auto", "--max-time", "1800", command], text=True
    ).strip()
    assert task.startswith("task_") and len(task.splitlines()) == 1
    jobs.append(
        dict(task_id=task, label=manifest["label"], output=str(job), command=command, last_observed_status="submitted")
    )
    path.write_text(json.dumps(jobs, indent=2) + "\n")
    print(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("v122", "n256"), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--submit-prepared", action="store_true")
    parser.add_argument("--name")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    if args.prepare_only and args.submit_prepared:
        parser.error("Choose prepare-only or submit-prepared")
    name = args.name or f"score_sim_{args.label}"
    assert Path(name).name == name and name not in (".", "..")
    job = here / "runs" / name
    if args.submit_prepared:
        assert json.loads((job / "job_manifest.json").read_text())["label"] == args.label
        submit(job, here)
        return
    workspace = here
    while not (workspace / "env-dsv4-0251rc1.sh").exists():
        assert workspace != workspace.parent
        workspace = workspace.parent
    compile_dir = here / "runs" / f"compile_producer_diagnostic_{args.label}_v2"
    assert json.loads((compile_dir / "result.json").read_text())["full_codegen_passed"]
    job.mkdir(exist_ok=False)
    for relative in (
        "ptoas/score_producer.cpp",
        "ptoas/score_producer.pto",
        "kernels/aic/score_producer.cpp",
        "compiled_meta.json",
    ):
        target = job / "input" / relative
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(compile_dir / "kernel" / relative, target)
    (job / "scripts/tools").mkdir(parents=True)
    for name in ("generate_score_sim_case.py", "run_score_simulator.py", "prepare_score_producer_diagnostic.py"):
        shutil.copyfile(here / name, job / "scripts" / name)
    skill = workspace / "pypto-lib/.claude/skills/incore-profiling"
    for name in ("incore_profile.py", "gen_profiling_case.py"):
        shutil.copyfile(skill / name, job / "scripts/tools" / name)
    launch = job / "launch.sh"
    shared_dir = workspace / "vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_shared_toolchain"
    launch.write_text(
        "#!/usr/bin/env bash\nset -eo pipefail\n"
        + "source "
        + shlex.quote(str(workspace / "env-dsv4-0251rc1.sh"))
        + "\n"
        + "source "
        + shlex.quote(str(shared_dir / "env.sh"))
        + "\n"
        + "exec python "
        + shlex.quote(str(job / "scripts/run_score_simulator.py"))
        + " --job "
        + shlex.quote(str(job))
        + ' "$@"\n'
    )
    shared = json.loads((shared_dir / "manifest.json").read_text())
    manifest = dict(
        scope="synthetic unthrottled single-AIC Score producer",
        formal_gate_eligible=False,
        label=args.label,
        workspace=str(workspace),
        compile_dir=str(compile_dir),
        shared_toolchain=shared,
        immutable_files={
            str(p.relative_to(job)): hashlib.sha256(p.read_bytes()).hexdigest() for p in job.rglob("*") if p.is_file()
        },
    )
    (job / "job_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(job)
        return
    submit(job, here)


if __name__ == "__main__":
    main()
