# SPDX-License-Identifier: Apache-2.0
"""Native loaded-weight binding for the complete BSH attention half.

Derived from nalinaly/vllm-ascend 71153bb3 native_adapter.py. No dequantization
or per-forward preparation; the owning decoder layer retains these tensors.
"""

import torch

_ACL_FORMAT_NCHW = 0
_ACL_FORMAT_ND = 2


def _base_weight_format(value: torch.Tensor) -> torch.Tensor:
    """仅为 ND 参数或离线打包承载张量在原设备上归一化基础格式。"""
    if value.device.type == "npu":
        import torch_npu

        if torch_npu.get_npu_format(value) not in (0, 2):
            return torch_npu.npu_format_cast(value, 2)
    return value


def prepare_weights(attention, hadamard: torch.Tensor | None, layer=None, *, root_function) -> dict[str, torch.Tensor]:
    """Prepare the TP1 ABI from already-loaded Native parameters exactly once."""
    import torch_npu

    from .pto_kernels.dspark_layer.nz_mode import root_weight_layouts

    if attention.compress_ratio != 4 or attention.n_local_heads != 64 or attention.n_local_groups != 8:
        raise ValueError("CSA specialization requires C4 and TP1 with 64 heads / 8 output groups")

    layouts = root_weight_layouts(root_function)

    def root_weight(name, shape, dtype):
        # 根矩阵方向与 Native 相同。格式已匹配时借用原存储，禁止解包、转置或重新打包。
        value = getattr(attention, name).weight.detach()
        if tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
            raise ValueError(f"Unexpected Native {name} weight: {value.shape}/{value.dtype}")
        current = int(torch_npu.get_npu_format(value))
        if layouts[name] == "NZ":
            return value if current == 29 else torch_npu.npu_format_cast(value, 29)
        return value if current in (_ACL_FORMAT_NCHW, _ACL_FORMAT_ND) else torch_npu.npu_format_cast(value, 2)

    def weight(module, shape, dtype, transpose=False):
        value = module.weight.detach()
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"Unexpected loaded weight: {value.shape}/{value.dtype}; expected {shape}/{dtype}")
        if torch_npu.get_npu_format(value) not in (_ACL_FORMAT_NCHW, _ACL_FORMAT_ND):
            # 这些非目标权重仍由根签名声明 ND，按其数学方向在加载期准备一次。
            # 四张目标权重由 root_weight 独立绑定，已有 NZ 存储直接复用。
            value = torch_npu.npu_format_cast(value, _ACL_FORMAT_ND)
        if transpose:
            value = value.transpose(-1, -2)
        return _base_weight_format(value.contiguous())

    def scale(module, width):
        result = module.weight_scale.detach().reshape(-1)
        if result.numel() != width:
            raise ValueError("Unexpected quantized channel-scale count")
        offset = getattr(module, "weight_offset", None)
        if offset is not None and bool(torch.count_nonzero(offset).cpu()):
            raise ValueError("The reference CSA chain requires symmetric INT8 weights")
        return result.float().contiguous()

    bf16, int8 = torch.bfloat16, torch.int8
    main, indexer = attention.compressor, attention.indexer
    inner = indexer.compressor
    # mHC 的门控权重与 attention 的 input_layernorm 挂在 DeepseekV4DecoderLayer 上。
    hc = {}
    if layer is not None:
        hc = {
            "hc_attn_fn": layer.hc_attn_fn.detach().float().contiguous(),
            "hc_attn_scale": layer.hc_attn_scale.detach().float().contiguous(),
            "hc_attn_base": layer.hc_attn_base.detach().float().contiguous(),
            "attn_norm_w": layer.input_layernorm.weight.detach().to(bf16).contiguous(),
        }
    weights = {
        **hc,
        "wq_a": root_weight("wq_a", (1024, 4096), bf16),
        "wq_b": root_weight("wq_b", (1024, 32768), int8),
        "wq_b_scale": scale(attention.wq_b, 32768),
        "wkv": weight(attention.wkv, (512, 4096), bf16, True),
        "gamma_cq": weight(attention.q_norm, (1024,), bf16),
        "gamma_ckv": weight(attention.kv_norm, (512,), bf16),
        "cmp_wkv": weight(main.wkv, (1024, 4096), bf16),
        "cmp_wgate": weight(main.wgate, (1024, 4096), bf16),
        "cmp_ape": main.ape.detach().float().contiguous(),
        # Match Native A3 storage; the RMS task widens loaded BF16 tiles.
        "cmp_norm_w": weight(main.norm, (512,), bf16),
        "idx_wq_b": weight(indexer.wq_b, (1024, 8192), int8),
        "idx_wq_b_scale": scale(indexer.wq_b, 8192),
        "weights_proj": weight(indexer.weights_proj, (64, 4096), bf16, True),
        **({"hadamard_idx": hadamard.detach().T.to(bf16).contiguous()} if hadamard is not None else {}),
        "inner_wkv": weight(inner.wkv, (256, 4096), bf16),
        "inner_wgate": weight(inner.wgate, (256, 4096), bf16),
        "inner_ape": inner.ape.detach().float().contiguous(),
        "inner_norm_w": weight(inner.norm, (128,), bf16),
        "attn_sink": attention.attn_sink.detach().contiguous(),
        "wo_a": root_weight("wo_a", (8, 4096, 1024), bf16),
        "wo_b": root_weight("wo_b", (8192, 4096), int8),
        "wo_b_scale": scale(attention.wo_b, 4096),
    }
    return weights
