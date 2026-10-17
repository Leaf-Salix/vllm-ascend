# SPDX-License-Identifier: Apache-2.0
"""仅供历史复现的静态 shape 模块加载器，禁止作为正式动态实现入口。

旧范围包装会按描述符实值化 shape 并展开阶段图。目录迁移不改变历史
结果，但这些结果不能作为目标动态单 SPMD 的功能或性能验收。
"""

import importlib
import importlib.util
import sys
from pathlib import Path

ARCHIVE_ROOT = Path(__file__).resolve().parent / "static_shape_archive_20261017"
KERNEL_SOURCE = ARCHIVE_ROOT / "kernels.py"


def load_historical_module(name):
    if name not in {"policy", "signal_codegen", "compiler", "arguments", "binding", "catalog"}:
        raise ValueError("未登记的历史静态 CSA 模块")
    package_name = "_csa_spmd_static_history"
    path = ARCHIVE_ROOT / "__init__.py"
    if package_name in sys.modules:
        if Path(sys.modules[package_name].__file__).resolve() != path.resolve():
            raise RuntimeError("历史模块已从其他归档加载")
    else:
        spec = importlib.util.spec_from_file_location(
            package_name, path, submodule_search_locations=[str(ARCHIVE_ROOT)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载历史静态 CSA 包：{path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for key in list(sys.modules):
                if key == package_name or key.startswith(package_name + "."):
                    sys.modules.pop(key)
            raise
    return importlib.import_module(f"{package_name}.{name}")
