# SPDX-License-Identifier: Apache-2.0
"""实验副本的图外纯DSL常量准备；保留原HF32打包函数，不改计算边界。"""

import pypto.language as pl

from .hc_native_projection_stages import hc_pack_weight


@pl.jit
def prepare_hc_weight(
    weight: pl.Tensor[[24, 16384], pl.FP32],
    packed: pl.InOut[pl.Tensor[[32768, 32], pl.BF16]],
):
    for worker in pl.spmd(48, name_hint="hc_constant_weight_pack", allow_early_resolve=True):
        packed = hc_pack_weight(weight, packed, worker)
    return packed


def precompile_static_weights():
    from pypto.runtime.kernel.abi import KernelConfig

    artifact, _, _ = prepare_hc_weight._resolve_compiled(
        (),
        {},
        allow_signature_mode=True,
        _kernel=True,
        _kernel_config=KernelConfig("a2a3", "tensormap_and_ringbuffer", 0, ring_task_window=2048),
    )
    artifact.load()
    return artifact.identity().hex()


def weight_descriptor(weight):
    return (
        weight.data_ptr(),
        weight.storage_offset(),
        tuple(weight.shape),
        tuple(weight.stride()),
        weight.dtype,
        weight.device,
    )


def prepare_static_weights(binding):
    import torch
    import torch_npu

    if torch_npu.npu.is_current_stream_capturing():
        raise RuntimeError("HC常量必须在FULL capture前准备")
    weight = binding.buffers["hc_fn"]
    version = weight._version  # 不可追踪版本的inference tensor须由调用方在图外显式克隆。
    descriptor = weight_descriptor(weight)
    packed = torch.empty((32768, 32), dtype=torch.bfloat16, device=weight.device)
    prepare_hc_weight(weight, packed)
    # 同流写后读；全局同步只发生于图外绑定准备，绝不进入解码计时。
    torch.npu.synchronize()
    if weight._version != version or weight_descriptor(weight) != descriptor:
        raise RuntimeError("准备期间HC权重变化")
    # 图外逐字节验证整个2MiB布局；参考原pack的整数变换/FP32低位相减/BF16边界。
    source = weight.cpu().transpose(0, 1).contiguous()
    padded = torch.zeros((16384, 32), dtype=torch.float32)
    padded[:, :24].copy_(source)
    converted_bits = (padded.view(torch.int32) + 2048) & -4096
    high = (converted_bits & -65536).view(torch.float32)
    low = converted_bits.view(torch.float32) - high
    reference = torch.cat(
        (high.to(torch.bfloat16).view(-1, 8, 32), low.to(torch.bfloat16).view(-1, 8, 32)), dim=1
    ).view(32768, 32)
    assert torch.equal(packed.cpu().view(torch.uint8), reference.view(torch.uint8)), "HC常量pack首差"
    binding.buffers["hc_prepacked_weight"] = packed
    binding._hc_weight_owner = weight
    binding._hc_weight_version = version
    binding._hc_weight_descriptor = descriptor


def check_static_weights(binding, native=None):
    weight = binding._hc_weight_owner
    if weight._version != binding._hc_weight_version or weight_descriptor(weight) != binding._hc_weight_descriptor:
        raise RuntimeError("HC权重已修改，必须重建绑定和FULL graph")
    if native is not None and native["hc_attn_fn"] is not weight:
        raise RuntimeError("HC权重已替换，必须重建绑定和FULL graph")
