# SPDX-License-Identifier: Apache-2.0
"""CSA 单 SPMD 的形状编译入口。

只读取调用方已有的 shape/stride/dtype 和长度上界；数学源码、效果合同、
分核与工作区全部来自本包。构建不启动设备，不读取历史 trace 或测试目录。
当前数学仍限每请求 S=6；其它 S 的实现与区间性能验收尚未完成。
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from .binding import bind
from .catalog import catalog
from .codegen import generate, jit_source
from .graph import build_source_schedule
from .layout import prepare
from .policy import QUERY_TOKENS, policy_from_descriptors
from .qk_indices import cache_qk_indices
from .row_dependencies import refine
from .signal_codegen import use_local_signals
from .worker_loops import loop_rebound_blocks


@dataclass
class GeneratedCSA:
    """编译产物及其绑定合同；生成成功不代表设备验证或性能达标。"""

    source: str
    schedule: dict
    layout: dict
    bindings: dict
    preparation: dict

    def save(self, directory):
        """保存到调用方独占的 JIT 缓存目录，拒绝覆盖已有产物。"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "entry_jit.py").write_text(self.source)
        for name, value in (
            ("schedule.json", self.schedule),
            ("layout.json", self.layout),
            ("entry.bindings.json", self.bindings),
            ("preparation.json", self.preparation),
        ):
            (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        return directory


def descriptors_from_tensors(arguments):
    """仅访问 Tensor 的 Host 描述符，不读取任何设备元素。"""
    return {
        name: {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype)}
        for name, tensor in arguments.items()
    }


def build_csa(
    descriptors, *, max_seq_len, workspace_directory, core_tensor_capacity, query_tokens=QUERY_TOKENS, member_costs=None
):
    """按实际规格生成一个完整 SPMD，可由调用方按规格缓存不同二进制。

    max_seq_len 必须是已有的 Host 长度上界，包含本轮 query。容量由部署方
    按实际 Simpler ABI 显式提供，不能将本地扩容后的能力视为共享运行库能力。
    member_costs 是显式调度权重，不是性能结论；缺省值同样不代表设备实测。
    workspace_directory 只存编译所需的数学特化源码，不是设备 GM 工作区。
    """
    if type(core_tensor_capacity) is not int or core_tensor_capacity <= 0:
        raise ValueError("必须显式提供实际运行库的正整数 Tensor 参数容量")
    policy = policy_from_descriptors(descriptors, max_seq_len=max_seq_len, query_tokens=query_tokens)
    inputs = copy.deepcopy(descriptors)
    for descriptor in inputs.values():
        # 旧验证器可能携带样本数值；生产编排只允许已有 Host 上界。
        descriptor.pop("metadata_values", None)
    bodies = catalog()
    bound = bind(Path(bodies["logical_source_ir"]), inputs, max_seq_len=max_seq_len)
    schedule, graph_audit = build_source_schedule(
        bound, bodies, policy=policy.schedule_policy, member_costs=member_costs
    )

    qk_audit = None
    if policy.qk_vector_indices:
        body_source, qk_audit = cache_qk_indices(schedule, bodies)
        path = Path(workspace_directory) / "qk_indices.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        # 源码写入自己的缓存目录，避免不同形状的并发构建相互覆盖。
        with path.open("x") as handle:
            handle.write(
                "# Copyright (c) PyPTO Contributors.\n"
                "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n" + body_source
            )
        bodies = dict(bodies, body_bundle=str(path))
    row_audit = refine(schedule, bodies) if policy.row_dependencies else None
    layout = prepare(schedule, bodies)
    program, bindings = generate(schedule, layout, bodies)
    source, layout, bindings, signal_audit = use_local_signals(jit_source(program), layout, bindings)
    source, looped_batches = loop_rebound_blocks(source, layout)

    contract = bindings["_entry_contract"]
    leaf_count = contract["leaf_tensor_count"]
    if leaf_count > core_tensor_capacity:
        raise ValueError(f"CSA 需要 {leaf_count} 个核 Tensor 参数，实际容量仅 {core_tensor_capacity}")
    contract.update(
        current_core_limit=core_tensor_capacity,
        runtime_supported=True,
        runtime_scope="调用方显式声明的运行库容量；部署前仍须核对实际 ABI",
    )
    layout["abi"].update(
        runtime_core_tensor_limit=core_tensor_capacity, fits_current_runtime=True, source="调用方显式容量合同"
    )
    preparation = {
        "batch": policy.batch,
        "tokens": policy.tokens,
        "query_tokens_per_request": policy.query_tokens,
        "host_max_seq_len_for_strategy": max_seq_len,
        "range_policy": policy.report(),
        "source_schedule_audit": graph_audit,
        "qk_index_cache": qk_audit,
        "row_dependencies": row_audit,
        "signal_audit": signal_audit,
        "looped_batches": looped_batches,
        "selected_control_path": bound["selected_control_path"],
        "per_step_device_metadata_preserved": True,
        "historical_artifacts_required": False,
        "wait_codegen_required": "dcci_volatile_load_backoff_100",
        "device_validated": False,
        "performance_validated": False,
    }
    return GeneratedCSA(source, schedule, layout, bindings, preparation)
