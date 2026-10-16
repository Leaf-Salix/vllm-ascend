# SPDX-License-Identifier: Apache-2.0
"""仅在当前测试进程选择独立 PyPTO 源码及其自行构建的扩展。"""

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dsv4_csa_env

original_activate = dsv4_csa_env.activate
pypto_root = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]).resolve()


def activate():
    loaded = sys.modules.get("pypto")
    if loaded is not None:
        assert pypto_root in Path(loaded.__file__).resolve().parents
        return Path(dsv4_csa_env.__file__).resolve().parents[2]
    result = original_activate()
    original = Path(os.environ["PTO_EAGER_ROOT"]) / "pypto"
    finders = [finder for finder in sys.meta_path
               if "pypto.pypto_core" in getattr(finder, "known_wheel_files", {})]
    if len(finders) != 1:
        raise RuntimeError("无法唯一定位本进程 PyPTO editable hook")
    finder = finders[0]
    if os.environ.get("CSA_STATIC_PYPTO_IMPORT_MODE") == "filesystem":
        # 上游版本可能新增模块；旧 editable 安装的模块清单不能代表新 worktree。
        # 只在当前进程移除 PyPTO hook，由目标目录加载源码及其自行构建的扩展。
        extensions = list((pypto_root / "python/pypto").glob("pypto_core*.so"))
        if len(extensions) != 1:
            raise RuntimeError(f"独立 worktree 的本地构建扩展不唯一：{extensions}")
        sys.meta_path.remove(finder)
        sys.path.insert(0, str(pypto_root / "python"))
        return result
    for name, path in list(finder.known_source_files.items()):
        if name == "pypto" or name.startswith("pypto."):
            finder.known_source_files[name] = str(pypto_root / Path(path).relative_to(original))
    extensions = list((pypto_root / "python/pypto").glob("pypto_core*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"独立 worktree 的本地构建扩展不唯一：{extensions}")
    finder.known_wheel_files["pypto.pypto_core"] = str(extensions[0])
    adapter = list((pypto_root / "python/pypto").glob("_torch_npu.cpython-*.so"))
    if len(adapter) == 1:
        finder.known_wheel_files["pypto._torch_npu"] = str(adapter[0])
    else:
        # 不回退到全局安装的旧 ABI；缺失时由正式加载器明确报错。
        finder.known_wheel_files.pop("pypto._torch_npu", None)
    return result


dsv4_csa_env.activate = activate
activate()
target = sys.argv.pop(1)
if target == "-m":
    module = sys.argv.pop(1)
    sys.argv[0] = module
    runpy.run_module(module, run_name="__main__")
else:
    sys.path.insert(0, str(Path(target).resolve().parent))
    sys.argv[0] = target
    runpy.run_path(target, run_name="__main__")
