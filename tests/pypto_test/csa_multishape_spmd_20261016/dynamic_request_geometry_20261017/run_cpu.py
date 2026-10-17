# SPDX-License-Identifier: Apache-2.0
"""复用已有隔离工具链入口，只记录本探针的 CPU 编译命令与日志。"""

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    workspace = repo.parent
    output = args.output.resolve()
    if output.parent != here or output.exists():
        raise ValueError("编译输出须为本测试目录下的新目录")
    output.mkdir()
    frozen = here.parent.parent / "csa_static_spmd_feasibility_20261015"
    executable = [str(workspace / ".venv-dsv4-0251rc1/bin/python"), str(frozen / "bootstrap_isolated.py"),
                  str(here / "compile_probe.py"), str(output)]
    roots = {
        "CSA_STATIC_PYPTO_ROOT": workspace / ".worktrees/pypto-csa-static-fixes-20261016",
        "HCA_SIMPLER_ROOT": workspace / ".worktrees/simpler-csa-core-args-256-20261016",
    }
    script = "source " + shlex.quote(str(workspace / "env-dsv4-0251rc1.sh")) + "\n"
    script += "\n".join("export " + key + "=" + shlex.quote(str(value)) for key, value in roots.items())
    site = roots["HCA_SIMPLER_ROOT"] / ".venv/lib/python3.10/site-packages"
    script += "\nexport PYTHONPATH=" + shlex.quote(str(site)) + ":${PYTHONPATH:-}\n"
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
