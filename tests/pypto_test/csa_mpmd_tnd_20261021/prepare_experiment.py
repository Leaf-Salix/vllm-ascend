# SPDX-License-Identifier: Apache-2.0
"""一次冻结一个候选及其测试代码；多档共用，不复制 SDK 或大权重。"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_reference import prepare_reference


def main():
    prepare_reference()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    destination = here / "experiments" / sys.argv[1]
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd",
        destination / "operator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("run_case.py", "scenarios.py", "cases.json", "baseline.json", "launch.sh", "state_replay.py"):
        shutil.copyfile(here / name, destination / name)
    helpers = destination / "helpers"
    helpers.mkdir()
    pending = {"dsv4_csa_env", "dsv4_csa_native_case", "dsv4_csa_single_layer", "dsv4_csa_validation"}
    done = set()
    while pending:
        name = pending.pop()
        if name in done:
            continue
        relative = f"tests/pypto_test/{name}.py"
        text = subprocess.check_output(["git", "-C", str(repo), "show", f"84751e34:{relative}"], text=True)
        (helpers / (name + ".py")).write_text(text)
        done.add(name)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dsv4_"):
                pending.add(node.module)
            elif isinstance(node, ast.Import):
                pending.update(alias.name for alias in node.names if alias.name.startswith("dsv4_"))
    print(destination)


if __name__ == "__main__":
    main()
