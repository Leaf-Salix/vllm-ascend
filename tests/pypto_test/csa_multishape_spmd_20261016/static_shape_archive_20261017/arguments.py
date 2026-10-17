# SPDX-License-Identifier: Apache-2.0
"""绑定原 Native Tensor 与独占 GM 工作区；不注册算子、不复制权重。

仅在 warmup/capture 前构造。同一对象不可被并发调用者共享，图重放不重置
进度代际。实际 JIT 编译/执行入口与等待回调集成尚须由部署层接入。
"""

import torch

DTYPES = {
    "BF16": torch.bfloat16,
    "FP16": torch.float16,
    "FP32": torch.float32,
    "INT8": torch.int8,
    "INT32": torch.int32,
    "INT64": torch.int64,
}
BASE_NPU_FORMATS = frozenset({0, 2})  # 与已有 CSA Native 适配合同一致：NCHW、ND。
NZ_NPU_FORMAT = 29


def validate_root_aliases(native_args, root_descriptors, contract):
    """源码 DAG 把各 root 视为不同 allocation，拒绝未建模的写侧重叠。

    检查只用 Host 存储地址、偏移和跨度；同一 root 的内部视图照常保留。
    对非连续布局采用保守包围区间，NZ 则采用完整物理 storage 区间。
    两个只读 root 可以共享存储，但任何一侧可写时必须证明区间不相交。
    """
    writable = {
        item["reference"]["allocation"].removeprefix("argument:")
        for item in contract["ordered_inputs"]
        if item["mutable"] and item["reference"].get("allocation", "").startswith("argument:")
    }
    regions = []
    for name, tensor in native_args.items():
        if tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        # meta 没有实际地址，仅用于 CPU 合同验证；真实设备使用完整地址范围。
        key = (tensor.device, storage._cdata if tensor.device.type == "meta" else None)
        base = 0 if tensor.device.type == "meta" else storage.data_ptr()
        if root_descriptors[name]["layout"] == "NZ":
            begin, end = base, base + storage.nbytes()
        else:
            strides = tuple(tensor.stride())
            if any(stride < 0 for stride in strides):
                raise ValueError("CSA 入口不支持负 stride")
            begin = base + tensor.storage_offset() * tensor.element_size()
            span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, strides))
            end = begin + span * tensor.element_size()
        for other_key, first, last, other_name in regions:
            if key == other_key and begin < last and first < end and (name in writable or other_name in writable):
                raise ValueError(f"不同 Native root 存在未建模的写侧别名：{other_name}/{name}")
        regions.append((key, begin, end, name))


class PreparedCSAArguments:
    def __init__(self, generated, native_args, *, workspace_guard_bytes=0):
        contract = generated.bindings["_entry_contract"]
        if contract["workspace_binding"] != "external":
            raise ValueError("CSA 入口要求 external GM 视图绑定")
        layout = generated.layout
        schedule = generated.schedule
        self.native_args = {name: native_args[name] for name in schedule["root_arguments"]}
        self.contract = contract
        self.selected_control_path = schedule["selected_control_path"]
        devices = {tensor.device for tensor in self.native_args.values()}
        if len(devices) != 1:
            raise ValueError("同一 CSA 的 Native 参数必须在同一设备")
        device = next(iter(devices))
        for name, expected in schedule["root_arguments"].items():
            tensor = self.native_args[name]
            if list(tensor.shape) != expected["shape"] or tensor.dtype != DTYPES[expected["dtype"]]:
                raise ValueError(f"Native shape/dtype 与模板不符：{name}")
            if list(tensor.stride()) != expected["strides"]:
                raise ValueError(f"Native stride 与模板不符：{name}")
            if device.type == "npu":
                import torch_npu

                actual_format = int(torch_npu.get_npu_format(tensor))
                allowed = {NZ_NPU_FORMAT} if expected["layout"] == "NZ" else BASE_NPU_FORMATS
                if actual_format not in allowed:
                    raise ValueError(f"Native 物理格式与 {expected['layout']} 合同不符：{name}/{actual_format}")

        validate_root_aliases(self.native_args, schedule["root_arguments"], contract)
        workspace = layout["workspace"]
        if workspace_guard_bytes < 0 or workspace_guard_bytes % workspace["alignment"]:
            raise ValueError("工作区保护区必须非负并保持原 arena 对齐")
        # 仅设备验证启用首尾保护区；默认准备路径及所有内部 offset 均保持不变。
        self.workspace_guard_bytes = workspace_guard_bytes
        self.workspace_allocation = torch.empty(
            workspace["bytes"] + 2 * workspace_guard_bytes, dtype=torch.int8, device=device
        )
        self.workspace = self.workspace_allocation.narrow(0, workspace_guard_bytes, workspace["bytes"])
        if device.type != "meta" and self.workspace.data_ptr() % workspace["alignment"]:
            raise ValueError("实际工作区地址未满足协议要求的缓存行对齐")
        if workspace_guard_bytes:
            self.workspace_allocation[:workspace_guard_bytes].fill_(37)
            self.workspace_allocation[-workspace_guard_bytes:].fill_(37)

        def arena_view(ref):
            # narrow/view 都是别名操作；不复制临时 Tensor，也不转换权重。
            return (
                self.workspace.narrow(0, ref["offset_bytes"], ref["bytes"])
                .view(DTYPES[ref["dtype"]])
                .view(ref["shape"])
            )

        self.sections = {name: arena_view(ref) for name, ref in workspace["sections"].items()}
        self.temporaries = {name: arena_view(ref) for name, ref in workspace["temporaries"].items()}
        self.sections["progress"].zero_()
        for name in ("records", "waits"):
            source = torch.tensor(workspace[name], dtype=torch.int64)
            if source.numel() != self.sections[name].numel():
                raise ValueError(f"编译期 {name} 表与工作区长度不符")
            self.sections[name].copy_(source)

        arguments = []
        for item in contract["ordered_inputs"]:
            ref = item["reference"]
            if "allocation" not in ref:
                tensor = self.sections[item["name"]]
            else:
                allocation = ref["allocation"]
                if allocation.startswith("argument:"):
                    base = self.native_args[allocation.removeprefix("argument:")]
                else:
                    base = self.temporaries[allocation]
                same_view = (
                    list(base.shape) == ref["shape"] and list(base.stride()) == ref["strides"] and ref["offset"] == 0
                )
                if ref["layout"] == "NZ":
                    if not same_view:
                        raise ValueError("禁止对原 NZ 权重猜测新视图")
                    tensor = base
                elif same_view:
                    tensor = base
                else:
                    last = ref["offset"] + sum((n - 1) * s for n, s in zip(ref["shape"], ref["strides"]))
                    if last >= base.numel():
                        raise ValueError(f"视图越过原 allocation：{item['name']}")
                    tensor = torch.as_strided(base, ref["shape"], ref["strides"], base.storage_offset() + ref["offset"])
            arguments.append(tensor)
        self.param_names = tuple(item["name"] for item in contract["ordered_inputs"])
        self.core_args = tuple(arguments)
        if len(self.core_args) != contract["root_tensor_count"]:
            raise ValueError("实际绑定数量与已记录核 ABI 不符")

    def require_core_capacity(self, capacity):
        """部署前按实际运行库能力检查；不允许悄悄截断参数。"""
        leaf_count = self.contract["leaf_tensor_count"]
        if leaf_count > capacity:
            raise RuntimeError(f"CSA 单 SPMD 需要 {leaf_count} 个核 Tensor 参数，当前仅支持 {capacity}")
