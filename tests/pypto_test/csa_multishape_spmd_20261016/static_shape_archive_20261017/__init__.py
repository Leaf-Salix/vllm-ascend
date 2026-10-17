# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 Flash 的 CSA 单 SPMD 实现。

policy 仅读取已有 Host 描述信息；kernels 中的计算继续读取 device metadata。
build_csa 从本包源码生成完整入口；独占工作区绑定在 prepare_arguments 中准备。
当前 S=6，生产 JIT 的等待回调接入及新调度设备验收仍未完成。
导入本包不初始化 NPU，也不加载 PyPTO 编译器或 torch。
"""

from .compiler import GeneratedCSA, build_csa, descriptors_from_tensors
from .policy import RangePolicy, policy_from_descriptors, select_policy


def prepare_arguments(generated, native_args, *, workspace_guard_bytes=0):
    """在捕获前按生成合同绑定 Native Tensor，不复制权重或读取设备元素。"""
    from .arguments import PreparedCSAArguments

    return PreparedCSAArguments(generated, native_args, workspace_guard_bytes=workspace_guard_bytes)


__all__ = [
    "GeneratedCSA",
    "RangePolicy",
    "build_csa",
    "descriptors_from_tensors",
    "policy_from_descriptors",
    "prepare_arguments",
    "select_policy",
]
