# SPDX-License-Identifier: Apache-2.0
"""测试进程中的动态CSA源码隔离，不切换共享安装或主工作区。

每个候选目录作为独立Python包导入，包内相对导入保留，编译时读取自己的
entry/helper。旧SPMD与候选可以同进程绑定同一批Native根Tensor，分别拥有
工作区、信号和JIT对象。该工具只供验证，不能作为生产路径的源码热切换接口。
"""

import importlib
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace


def load_package(source, label):
    if source is None:
        name = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd"
        package = importlib.import_module(name)
    else:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", label):
            raise ValueError("隔离包标签只允许小写字母、数字和下划线")
        source = Path(source).resolve()
        init = source / "__init__.py"
        if not init.is_file() or not (source / "entry.py").is_file():
            raise ValueError(f"不是完整动态CSA包目录：{source}")
        importlib.import_module("vllm_ascend.ops.pypto")
        name = "vllm_ascend.ops.pypto._csa_validation_" + label
        if name in sys.modules:
            raise RuntimeError(f"隔离包已导入，禁止中途换源码：{name}")
        spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(source)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        try:
            spec.loader.exec_module(package)
        except BaseException:
            del sys.modules[name]
            raise
    return SimpleNamespace(name=name, source=Path(package.__file__).resolve().parent)


def load_binding(source, label):
    package = load_package(source, label)
    binding = importlib.import_module(package.name + ".binding")
    policy = importlib.import_module(package.name + ".policy")
    abi = importlib.import_module(package.name + ".runtime_abi")
    return SimpleNamespace(
        **vars(package), prepared=binding.PreparedDynamicCSA,
        select_policy=policy.select_policy, phase_count=abi.PHASE_COUNT,
    )
