"""只供本轮独立进程使用：阻止历史 editable finder 抢先绑定旧工具链。

由测试脚本显式加入 PYTHONPATH，子进程继承相同绑定；不修改共享环境。
只替换指定顶层包的查找位置，不改变算子、运行时或精度比较逻辑。
"""

import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys


roots = {}
if os.environ.get("CSA_SDMA_SIMPLER_ROOT"):
    root = Path(os.environ["CSA_SDMA_SIMPLER_ROOT"]).resolve()
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    roots.update(simpler=root / "python", simpler_setup=root,
                 _task_interface=root / ".venv/lib" / version / "site-packages")
if os.environ.get("CSA_SDMA_PYPTO_ROOT"):
    roots["pypto"] = Path(os.environ["CSA_SDMA_PYPTO_ROOT"]).resolve() / "python"


class SelectedToolchainFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.partition(".")[0]
        if top not in roots:
            return None
        spec = importlib.machinery.PathFinder.find_spec(
            fullname, path if "." in fullname else [str(roots[top])]
        )
        if spec is None:
            raise ImportError(f"指定工具链中缺少 {fullname}: {roots[top]}")
        return spec


sys.meta_path.insert(0, SelectedToolchainFinder())
