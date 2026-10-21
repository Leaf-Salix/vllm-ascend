# SPDX-License-Identifier: Apache-2.0
"""当前 release 的单卡进程组、Native cache owner 与带保护区的物理存储。"""

import tempfile
from contextlib import contextmanager


@contextmanager
def native_session(config, device_index: int):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_ascend.ascend_config import init_ascend_config

    torch.npu.set_device(device_index)
    with tempfile.TemporaryDirectory(prefix="csa_tp1_") as temporary, set_current_vllm_config(config):
        init_ascend_config(config)
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=device_index,
                distributed_init_method=f"file://{temporary}/distributed",
                backend="hccl",
            )
            initialize_model_parallel(backend="hccl")
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def make_cache_groups(config, device, attention):
    """五组缓存各自使用 Native owner、spec 和 builder。"""
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

    owners = {
        "swa": attention.dsa_attn.swa_cache_layer,
        "compressed": attention.dsa_attn.dsa_attn,
        "state": attention.compressor.state_cache,
    }
    if attention.compress_ratio == 4:
        owners.update(indexer=attention.indexer.k_cache, indexer_state=attention.indexer.compressor.state_cache)
    groups = {}
    for kind, owner in owners.items():
        prefix = owner.layer_name if kind == "compressed" else owner.prefix
        spec = owner.get_kv_cache_spec(config)
        builder = AscendDSAMetadataBuilder(spec, [prefix], config, device)
        groups[kind] = {"owner": owner, "prefix": prefix, "spec": spec, "builder": builder}
    return groups


def allocate_native_cache(group, pages, device):
    """在带保护区的分配上调用 Native runner 的页布局函数。"""
    import torch

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    spec = group["spec"]
    backend = group["owner"].get_attn_backend()
    page_size = spec.block_size
    shape = backend.get_kv_cache_shape(pages, page_size, spec.num_kv_heads, spec.head_size)
    shapes, dtypes = [shape], [spec.dtype]
    if getattr(spec, "scale_dim", 0):
        shapes.append(backend.get_kv_cache_shape(pages, page_size, spec.num_kv_heads, spec.scale_dim))
        dtypes.append(spec.scale_dtype)
    guard_bytes = 128
    allocation = torch.full((pages * spec.page_size_bytes + 2 * guard_bytes,), 37, dtype=torch.int8, device=device)
    raw = allocation[guard_bytes:-guard_bytes]
    views = NPUModelRunner._adjust_kv_layout(None, raw, shapes, dtypes, spec.page_size_bytes)
    group.update(allocation=allocation, raw=raw, views=views, page_size=page_size, pages=pages)
    return {
        "spec": repr(spec),
        "logical_block_size": spec.block_size,
        "storage_block_size": page_size,
        "page_bytes": spec.page_size_bytes,
        "raw_offset": raw.storage_offset(),
        "views": [
            {
                "shape": list(view.shape),
                "dtype": str(view.dtype),
                "stride": list(view.stride()),
                "storage_offset_elements": view.storage_offset(),
                "storage_offset_bytes": view.storage_offset() * view.element_size(),
                "contiguous": view.is_contiguous(),
                "storage_ptr": view.untyped_storage().data_ptr(),
            }
            for view in views
        ],
    }
