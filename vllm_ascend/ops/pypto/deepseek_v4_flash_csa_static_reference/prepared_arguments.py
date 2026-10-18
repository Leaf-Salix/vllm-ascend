# SPDX-License-Identifier: Apache-2.0
"""静态 CSA 的独立参数准备；不注册算子、不启动设备计算。

构造发生在 warmup/capture 前。Native 输入及四份 NZ 权重保持原存储；
只有临时工作区的 byte arena 建立 dtype/shape 视图。每个并发调用者必须
独占此对象。普通重放不能重复初始化进度标记。
"""

import json
from pathlib import Path

import torch


DTYPES = {"BF16": torch.bfloat16, "FP16": torch.float16, "FP32": torch.float32, "INT8": torch.int8,
          "INT32": torch.int32, "INT64": torch.int64}


class PreparedStaticArguments:
    def __init__(self, case_directory, native_args, *, workspace_guard_bytes=0):
        directory = Path(case_directory)
        contract = json.loads((directory / "entry.bindings.json").read_text())["_entry_contract"]
        if contract["workspace_binding"] != "external":
            raise ValueError("此适配器要求 external GM 视图入口")
        layout = json.loads((directory / "layout.json").read_text())
        prepared = directory.parent.parent / "bindings" / (directory.name + "_prepared.json")
        schedule = json.loads(prepared.read_text())
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
            if expected["layout"] == "NZ" and device.type == "npu":
                import torch_npu

                if torch_npu.get_npu_format(tensor) != 29:
                    raise ValueError(f"原 NZ 权重格式不符：{name}")

        workspace = layout["workspace"]
        if workspace_guard_bytes < 0 or workspace_guard_bytes % workspace["alignment"]:
            raise ValueError("工作区保护区必须非负并保持原 arena 对齐")
        # 仅设备验证启用首尾保护区；默认准备路径及所有内部 offset 均保持不变。
        self.workspace_guard_bytes = workspace_guard_bytes
        self.workspace_allocation = torch.empty(
            workspace["bytes"] + 2 * workspace_guard_bytes, dtype=torch.int8, device=device)
        self.workspace = self.workspace_allocation.narrow(0, workspace_guard_bytes, workspace["bytes"])
        if device.type != "meta" and self.workspace.data_ptr() % workspace["alignment"]:
            raise ValueError("实际工作区地址未满足协议要求的缓存行对齐")
        if workspace_guard_bytes:
            self.workspace_allocation[:workspace_guard_bytes].fill_(37)
            self.workspace_allocation[-workspace_guard_bytes:].fill_(37)

        def arena_view(ref):
            # narrow/view 都是别名操作；不复制临时 Tensor，也不转换权重。
            return self.workspace.narrow(0, ref["offset_bytes"], ref["bytes"]).view(
                DTYPES[ref["dtype"]]).view(ref["shape"])

        self.sections = {name: arena_view(ref) for name, ref in workspace["sections"].items()}
        self.temporaries = {name: arena_view(ref) for name, ref in workspace["temporaries"].items()}
        self.sections["progress"].zero_()
        for name in ("records", "waits"):
            source = torch.tensor(workspace[name], dtype=torch.int64)
            if source.numel() != self.sections[name].numel():
                raise ValueError(f"静态 {name} 表与工作区长度不符")
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
                same_view = list(base.shape) == ref["shape"] and list(base.stride()) == ref["strides"] and ref["offset"] == 0
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
                    tensor = torch.as_strided(base, ref["shape"], ref["strides"],
                                              base.storage_offset() + ref["offset"])
            arguments.append(tensor)
        self.param_names = tuple(item["name"] for item in contract["ordered_inputs"])
        self.core_args = tuple(arguments)
        if len(self.core_args) != contract["root_tensor_count"]:
            raise ValueError("实际绑定数量与已记录核 ABI 不符")

    def require_core_capacity(self, capacity):
        """部署前按实际运行库能力检查；不允许悄悄截断参数。"""
        leaf_count = self.contract["leaf_tensor_count"]
        if leaf_count > capacity:
            raise RuntimeError(f"静态 CSA 需要 {leaf_count} 个核 Tensor 参数，当前仅支持 {capacity}")
