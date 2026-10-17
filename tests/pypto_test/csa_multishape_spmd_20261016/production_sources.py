# SPDX-License-Identifier: Apache-2.0
"""CPU 验证按文件加载正式模块，避免仅检查策略时初始化 vLLM。

这只是测试侧加载器。正式算子包不导入 tests，也不读取历史实验目录。
"""

import importlib
import importlib.util
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
OPERATOR_ROOT = REPOSITORY / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
SIGNAL_SOURCE = OPERATOR_ROOT / "signals.py"


def load_operator_module(name):
    if name not in {"query_source", "signals", "wait_codegen"}:
        raise ValueError("静态 shape 生成链已移至历史目录；正式动态入口不回退到历史实现")
    package_name = "_csa_spmd_production"
    path = OPERATOR_ROOT / "__init__.py"
    if package_name in sys.modules:
        if Path(sys.modules[package_name].__file__).resolve() != path.resolve():
            raise RuntimeError("正式模块已从其他工作区加载")
    else:
        spec = importlib.util.spec_from_file_location(
            package_name, path, submodule_search_locations=[str(OPERATOR_ROOT)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载正式 CSA SPMD 包：{path}")
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
