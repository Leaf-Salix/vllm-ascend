# SPDX-License-Identifier: Apache-2.0
"""同进程各对照侧固定实际导入/编译路径；只记录一次Git版本，不做文件hash。"""

import importlib
import json
import os
import subprocess
from pathlib import Path


def snapshot():
    from pypto.backend._ptoas_locate import find_ptoas_binary
    from pypto.runtime.kernel_compiler import KernelCompiler

    roots = {
        "pypto": Path(os.environ["CSA_SDMA_PYPTO_ROOT"]).resolve(),
        "ptoas": Path(os.environ["PTOAS_ROOT"]).resolve(),
        "simpler": Path(os.environ["HCA_SIMPLER_ROOT"]).resolve(),
    }
    loaded = {}
    for name, owner in (
        ("pypto", "pypto"),
        ("pypto.pypto_core", "pypto"),
        ("simpler", "simpler"),
        ("simpler_setup", "simpler"),
        ("_task_interface", "simpler"),
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if roots[owner] not in path.parents:
            raise RuntimeError(f"{name}实际加载路径不属于冻结工具链：{path}")
        loaded[name] = str(path)
    ptoas = Path(find_ptoas_binary()).resolve()
    if ptoas != (roots["ptoas"] / "bin/ptoas").resolve():
        raise RuntimeError(f"PTOAS实际可执行文件不一致：{ptoas}")
    sdk_root = Path(KernelCompiler(platform="a2a3").project_root).resolve()
    if sdk_root != roots["simpler"]:
        raise RuntimeError(f"编译器实际Simpler SDK根目录不一致：{sdk_root}")
    return dict(
        roots={k: str(v) for k, v in roots.items()},
        modules=loaded,
        ptoas_executable=str(ptoas),
        simpler_sdk_root=str(sdk_root),
    )


def record():
    identity = snapshot()
    identity["commits"] = {
        name: subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        for name, root in identity["roots"].items()
    }
    return identity


def check_same(expected):
    actual = snapshot()
    if actual != {k: v for k, v in expected.items() if k != "commits"}:
        raise RuntimeError("对照侧之间工具链路径发生变化，不能继续进行性能对照")


def check_ptoas_invocations(path, expected):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    actual = {str(Path(row["compiler"]).resolve()) for row in records}
    if actual != {expected["ptoas_executable"]}:
        raise RuntimeError(f"实际PTOAS调用混用不同编译器：{sorted(actual)}")
    return dict(calls=len(records), executables=sorted(actual))
