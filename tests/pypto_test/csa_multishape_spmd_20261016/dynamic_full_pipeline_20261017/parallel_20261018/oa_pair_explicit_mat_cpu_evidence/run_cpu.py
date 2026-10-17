# SPDX-License-Identifier: Apache-2.0
"""记录完整命令并运行既有隔离 CPU 编译入口；不创建设备 runtime。"""

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    workspace = Path("/data/pyptouser/qinchuanyu/pto-eager")
    repo = workspace / "vllm-ascend-dsv4-pto-0251rc1"
    frozen = repo / "tests/pypto_test/csa_static_spmd_feasibility_20261015"
    source = (here / args.source).resolve()
    output = (here / args.output).resolve()
    if source.parent != here or output.parent != here or output.exists():
        raise ValueError("仅允许独占目录内的新输出，禁止覆盖历史编译")
    output.mkdir()
    executable = [
        str(workspace / ".venv-dsv4-0251rc1/bin/python"),
        str(frozen / "bootstrap_isolated.py"),
        str(frozen / "compile_jit_entry.py"),
        "--source", str(source), "--output", str(output), "--memory-planner", "PYPTO", "--kernel-mode",
    ]
    assignments = {
        "CSA_STATIC_PYPTO_ROOT": workspace / ".worktrees/pypto-csa-static-fixes-20261016",
        "HCA_SIMPLER_ROOT": workspace / ".worktrees/simpler-csa-core-args-256-20261016",
    }
    script = "source " + shlex.quote(str(workspace / "env-dsv4-0251rc1.sh")) + "\n"
    script += "\n".join("export " + key + "=" + shlex.quote(str(value)) for key, value in assignments.items())
    extension_site = assignments["HCA_SIMPLER_ROOT"] / ".venv/lib/python3.10/site-packages"
    script += "\nexport PYTHONPATH=" + shlex.quote(str(extension_site)) + ":${PYTHONPATH:-}\n"
    script += "exec " + shlex.join(executable) + "\n"
    command = {"argv": ["bash", "-lc", script], "cwd": str(repo), "device_execution": False}
    (output / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n")
    started = time.time()
    with (output / "compile.log").open("w") as stream:
        completed = subprocess.run(command["argv"], cwd=repo, stdout=stream, stderr=subprocess.STDOUT, check=False)
    command.update(returncode=completed.returncode, elapsed_seconds=time.time() - started)
    (output / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(command, ensure_ascii=False, indent=2))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
