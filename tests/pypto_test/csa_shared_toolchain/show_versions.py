# SPDX-License-Identifier: Apache-2.0
"""核对公共SDK在当前进程和子进程中的实际加载来源；不初始化NPU。"""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

from selection import selected_toolchain


def identity():
    data, roots, _ = selected_toolchain()
    # torch/torch_npu先提供桥接DSO所需的共享库，只加载模块，不创建NPU张量。
    importlib.import_module("torch")
    importlib.import_module("torch_npu")
    expected = {
        "pypto": roots["pypto"] / "python",
        "pypto.pypto_core": roots["pypto"] / "python",
        "pypto._torch_npu": roots["pypto"] / "python",
        "simpler": roots["simpler"] / "python",
        "_task_interface": roots["simpler"] / ".venv",
    }
    paths = {}
    for name, root in expected.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"{name}来自错误路径：{path}，预期{root}")
        paths[name] = str(path)
    return {"release": data["release"], "status": data["status"], "modules": paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = identity()
    marker = "CSA_TOOLCHAIN_IDENTITY "
    if args.child:
        print(marker + json.dumps(report), flush=True)
        return
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child"], text=True, capture_output=True, check=True
    )
    lines = [line[len(marker) :] for line in child.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"子进程未返回唯一SDK来源记录：{child.stdout}\n{child.stderr}")
    child_report = json.loads(lines[0])
    if report != child_report:
        raise RuntimeError(f"父子进程工具链不一致：{report} / {child_report}")
    _, roots, _ = selected_toolchain()
    binary = roots["ptoas"] / "bin/ptoas"
    version = subprocess.check_output([str(binary), "--version"], text=True)
    report.update(passed=True, child_matches=True, ptoas={"path": str(binary), "version": version.strip()})
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
