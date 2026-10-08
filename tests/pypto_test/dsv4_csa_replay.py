# SPDX-License-Identifier: Apache-2.0
"""CSA 快照与回放：显式物理布局、共享存储和每次调用前的初态。"""

import ast
import ctypes
import inspect
import json
import textwrap
from pathlib import Path

SCHEMA_VERSION = 2


def argument_roles(root_function):
    from pypto.language import constexpr

    definition = ast.parse(textwrap.dedent(inspect.getsource(root_function))).body[0]
    signature = inspect.signature(root_function)
    roles = {}
    for arg in definition.args.args:
        # constexpr parameters specialize source but never appear in the
        # runtime ABI or a device tensor snapshot.
        if signature.parameters[arg.arg].annotation is constexpr:
            continue
        annotation = arg.annotation
        marker = annotation.value if isinstance(annotation, ast.Subscript) else None
        roles[arg.arg] = {"Out": "out", "InOut": "inout"}.get(getattr(marker, "attr", None), "in")
    return roles


def capture_tensors(tensors, roles, weight_layouts, source):
    """同一底层存储只复制一次；保存原始偏移和重建偏移，不把 view 各自 clone。"""
    import torch

    if set(tensors) != set(roles) or any(role not in ("in", "out", "inout") for role in roles.values()):
        raise ValueError("快照角色必须覆盖全部根参数")
    groups = {}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.numel() or any(s < 0 for s in tensor.stride()):
            raise ValueError(f"不支持的根入参：{name}")
        key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        groups.setdefault(key, []).append(name)
    specs, stores, payload = {}, {}, {}
    for index, names in enumerate(groups.values()):
        sid = str(index)
        first = tensors[names[0]]
        source_format = 2
        if first.device.type == "npu":
            import torch_npu

            source_format = int(torch_npu.get_npu_format(first))
        if source_format not in (0, 2, 29):
            raise ValueError(f"快照尚不支持 Native 格式 {source_format}：{names}")
        if source_format == 29:
            if (
                len(names) != 1
                or roles[names[0]] != "in"
                or weight_layouts.get(names[0]) != "NZ"
                or first.storage_offset() != 0
                or not first.is_contiguous()
                or first.untyped_storage().nbytes() != first.numel() * first.element_size()
            ):
                raise ValueError(f"NZ 快照要求完整、只读、无 padding 的独占权重：{names}")
        ranges = []
        for name in names:
            value = tensors[name]
            start = value.storage_offset() * value.element_size()
            stop = start + (1 + sum((n - 1) * s for n, s in zip(value.shape, value.stride()))) * value.element_size()
            ranges.append((start, stop))
        # 保持入参相对 64 字节边界的偏移；不复制整个模型共享池中与本调用无关的两端。
        start = min(left for left, _ in ranges) // 64 * 64
        stop = min((max(right for _, right in ranges) + 63) // 64 * 64, first.untyped_storage().nbytes())
        if source_format == 29:
            # Tensor.cpu()/storage.cpu() 会解码 Native NZ；快照需要保存原始物理字节。
            torch.npu.synchronize(first.device)
            raw = torch.empty(stop - start, dtype=torch.uint8)
            acl = ctypes.CDLL("libascendcl.so")
            copy = acl.aclrtMemcpy
            copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            copy.restype = ctypes.c_int
            code = copy(raw.data_ptr(), raw.numel(), first.data_ptr() + start, raw.numel(), 2)
            if code:
                raise RuntimeError(f"Native NZ 快照 D2H 失败：{code}")
            payload[sid] = raw
        else:
            raw = (
                first.as_strided((1,), (1,), 0)
                .view(torch.uint8)
                .as_strided((first.untyped_storage().nbytes(),), (1,), 0)
            )
            payload[sid] = raw[start:stop].to("cpu", copy=True)
        stores[sid] = {
            "source_byte_offset": start,
            "source_nbytes": first.untyped_storage().nbytes(),
            "nbytes": stop - start,
            "parameters": names,
        }
        for name in names:
            value = tensors[name]
            specs[name] = {
                "storage": sid,
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "role": roles[name],
                "layout": weight_layouts.get(name, "ND"),
                "source_format": source_format,
                "source_storage_offset": value.storage_offset(),
                "storage_offset": value.storage_offset() - start // value.element_size(),
            }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source": dict(source),
        "param_names": list(tensors),
        "tensors": specs,
        "storages": stores,
        "weight_layouts": dict(weight_layouts),
    }
    return meta, payload


def save_snapshot(directory, meta, payload):
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "csa_args_meta.json").exists() or (directory / "csa_args.pt").exists():
        raise FileExistsError(f"快照目录必须是新的：{directory}")
    torch.save(payload, directory / "csa_args.pt")
    (directory / "csa_args_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def materialize(meta, payload, device="cpu"):
    import torch

    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("旧快照缺少布局/初态/别名信息；必须重新采集 schema=2，不能猜测或自动二次打包")
    if meta.get("source", {}).get("state_timing") not in ("before_call", "constructed_initial"):
        raise ValueError("快照未声明可回放的调用前初态")
    backings = {}
    for sid, desc in meta["storages"].items():
        raw = payload[sid]
        if raw.dtype != torch.uint8 or list(raw.shape) != [desc["nbytes"]]:
            raise ValueError(f"快照存储大小或类型错误：{sid}")
        backings[sid] = raw.to(device, copy=True)
    tensors = {}
    for name in meta["param_names"]:
        spec = meta["tensors"][name]
        dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
        raw = backings[spec["storage"]]
        size = torch.empty((), dtype=dtype).element_size()
        tensors[name] = (
            raw[: raw.numel() // size * size]
            .view(dtype)
            .as_strided(spec["shape"], spec["stride"], spec["storage_offset"])
        )
    return tensors, backings


def convert_weight_layouts(tensors, meta, target_layouts, target_shapes=None):
    """来源布局/形状相同则不动字节；只转换独占的只读权重。"""
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.native_adapter import repack_weights

    converted = []
    for name, target in target_layouts.items():
        source = meta["weight_layouts"].get(name)
        if source not in ("ND", "NZ") or target not in ("ND", "NZ"):
            raise ValueError(f"未知权重布局：{name} {source} -> {target}")
        shape_matches = target_shapes is None or tuple(tensors[name].shape) == tuple(target_shapes[name])
        if source == target and shape_matches:
            continue
        spec = meta["tensors"][name]
        if spec["role"] != "in":
            raise ValueError(f"只有只读权重允许转换布局：{name}")
        if len(meta["storages"][spec["storage"]]["parameters"]) != 1:
            raise ValueError(f"布局转换会破坏共享存储：{name}，需先明确整组别名的转换方式")
        converted.append(name)
    return repack_weights(tensors, meta["weight_layouts"], target_layouts, target_shapes), converted


def restore_mutable_storages(meta, initial, backings):
    mutable = {spec["storage"] for spec in meta["tensors"].values() if spec["role"] != "in"}
    for sid in mutable:
        backings[sid].copy_(initial[sid])


def capture_written_pages(tensors):
    """在线数值诊断只备份声明会写的页；越界与保护区需独立检查，不能由此证明。"""
    import torch

    slots = {
        "kv_cache": "ori_slot_mapping",
        "cmp_kv": "cmp_slot_mapping",
        "idx_kv_cache": "idx_slot_mapping",
        "compress_state": "state_slot_mapping",
        "inner_compress_state": "inner_state_slot_mapping",
    }
    saved = []
    for name, slot_name in slots.items():
        value = tensors[name]
        page_ids = tensors[slot_name].detach().cpu()[:, 0].to(torch.int64)
        page_ids = page_ids[page_ids >= 0].unique()
        if page_ids.numel() and int(page_ids.max()) >= value.shape[0]:
            raise ValueError(f"{slot_name} 包含越界页号")
        page_ids = page_ids.to(value.device)
        saved.append((value, page_ids, value.index_select(0, page_ids)))
    for name in ("idx_topk_scores", "idx_topk", "x_out"):
        value = tensors[name]
        saved.append((value, None, value.detach().clone()))
    return saved


def restore_written_pages(saved):
    for value, page_ids, initial in saved:
        if page_ids is None:
            value.copy_(initial)
        else:
            value.index_copy_(0, page_ids, initial)
