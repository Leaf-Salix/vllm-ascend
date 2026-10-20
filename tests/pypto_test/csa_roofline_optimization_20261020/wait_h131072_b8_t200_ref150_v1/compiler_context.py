# SPDX-License-Identifier: Apache-2.0
"""同进程编译器绑定与记录：本轮三方均使用修正后的冻结v3。

每次候选调用（包括graph capture时）使用同一选择，避免工具链cache身份
改变导致捕获内重新编译。只改测试进程的既有PTOAS_ROOT/PATH并立即恢复，
不改变设备输入、数学源码或同步策略；L1计时仍来自设备。
"""
import contextlib
import functools
import json
import os
from pathlib import Path


@contextlib.contextmanager
def select_compiler(root):
    root = Path(root).resolve()
    previous = {key: os.environ.get(key) for key in ("PTOAS_ROOT", "PATH")}
    os.environ["PTOAS_ROOT"] = str(root)
    os.environ["PATH"] = str(root / "bin") + os.pathsep + (previous["PATH"] or "")
    try:
        from pypto.backend._ptoas_locate import find_ptoas_binary
        assert Path(find_ptoas_binary()).resolve() == root / "bin/ptoas"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def bind_compiler(base, root):
    class CompilerBoundPrepared(base):
        def __call__(self, *, config=None):
            with select_compiler(root):
                return super().__call__(config=config)
    return CompilerBoundPrepared


def record_invocations(path):
    from pypto.backend import pto_backend
    from pypto.backend._ptoas_locate import find_ptoas_binary
    original = pto_backend._run_ptoas

    @functools.wraps(original)
    def logged(*args, **kwargs):
        with Path(path).open("a") as output:
            output.write(json.dumps({"compiler": find_ptoas_binary(),
                                     "pto": args[0] if args else kwargs.get("pto_path")},
                                    ensure_ascii=False) + "\n")
        return original(*args, **kwargs)
    pto_backend._run_ptoas = logged
