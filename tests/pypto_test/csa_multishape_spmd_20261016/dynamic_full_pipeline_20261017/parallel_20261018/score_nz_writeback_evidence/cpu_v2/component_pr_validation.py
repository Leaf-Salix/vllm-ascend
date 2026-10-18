# SPDX-License-Identifier: Apache-2.0
"""在当前干净组件 worktree 执行指定 CPU 回归，记录实际核心与编译器来源。

测试代码来自当前 worktree，不复制其它目录的二进制；只移除共享环境的 PyPTO
editable-import 定位器，避免它覆盖显式源码路径。结果放入用户指定的新目录。
此脚本不触碰设备、不改 Git 状态、不代表完整系统测试或整层性能验收。
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("PYPTO_SKIP_RUNTIME_PIN_CHECK", "1")
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    sys.meta_path[:] = [f for f in sys.meta_path
                       if "pypto.pypto_core" not in getattr(f, "known_wheel_files", {})]
    sys.path.insert(0, str(root / "python"))
    import pypto
    from pypto import pypto_core
    from pypto.backend._ptoas_locate import find_ptoas_binary
    import pytest

    core = Path(pypto_core.__file__).resolve()
    assert root.resolve() in core.parents, core
    compiler = find_ptoas_binary()
    provenance = dict(root=str(root), core=str(core), python=pypto.__file__,
                      commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      ptoas=str(compiler), ptoas_version=subprocess.check_output(
                          [str(compiler), "--version"], text=True).strip(), tests=args.tests)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    options = ["-n", "0", "-vv", "--tb=short", "--junitxml=" + str(args.output / "junit.xml"),
               "--basetemp=" + str(args.output / "tmp"), *args.tests]
    code = pytest.main(options)
    (args.output / "exit.json").write_text(json.dumps(dict(exit_code=int(code))) + "\n")
    raise SystemExit(int(code))


if __name__ == "__main__":
    main()
