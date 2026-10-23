# SPDX-License-Identifier: Apache-2.0
"""冻结本轮算子与已验证测试入口；SDK始终由公共bootstrap选择。"""

import argparse
import ast
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREVIOUS = HERE.parent / "csa_mpmd_tnd_20261021"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("实验名只能是一段目录名")
    destination = HERE / "experiments" / args.name
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd",
        destination / "operator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in (
        "run_case.py",
        "scenarios.py",
        "cases.json",
        "baseline.json",
        "launch.sh",
        "state_replay.py",
        "service_validation.py",
    ):
        shutil.copyfile(PREVIOUS / name, destination / name)
    runner = destination / "run_case.py"
    text = runner.read_text()
    # 完整性能对照每侧10个窗口；单独状态验证仍由原入口控制，不增加重复运行。
    assert text.count("range(5)") == 1
    runner.write_text(text.replace("range(5)", "range(10)"))
    helpers = destination / "helpers"
    helpers.mkdir()
    pending = {"dsv4_csa_env", "dsv4_csa_native_case", "dsv4_csa_single_layer", "dsv4_csa_validation"}
    done = set()
    while pending:
        name = pending.pop()
        if name in done:
            continue
        source = subprocess.check_output(
            ["git", "-C", str(REPO), "show", f"84751e34:tests/pypto_test/{name}.py"], text=True
        )
        (helpers / (name + ".py")).write_text(source)
        done.add(name)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dsv4_"):
                pending.add(node.module)
            elif isinstance(node, ast.Import):
                pending.update(alias.name for alias in node.names if alias.name.startswith("dsv4_"))
    manifest = {
        "source_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "candidate_includes_worktree_changes": True,
        "reference_commit": "84751e34cb1f27eff9e4d3fdf03f8495982f1cdf",
        "shared_toolchain": json.loads((HERE.parent / "csa_shared_toolchain/manifest.json").read_text()),
        "windows_per_side": 10,
        "scope": "legacy fixed cases; pass --cases for the separate new performance spec",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
