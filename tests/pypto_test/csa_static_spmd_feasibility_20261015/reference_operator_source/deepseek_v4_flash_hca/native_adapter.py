# SPDX-License-Identifier: Apache-2.0
"""HCA 的 Native 零拷贝描述符与一次整层调用。"""

import math
from dataclasses import dataclass
from typing import Any

import torch

from ..deepseek_v4_flash_csa.native_adapter import prepare_weights as _prepare_weights
from ..deepseek_v4_flash_csa.native_storage import physical_pages, table_storage
from ..variant import csa_runtime
from .decode_hca import _decode_hca_tp1_layer, decode_hca_tp1_layer_hbg, decode_hca_tp1_layer_test


@dataclass(frozen=True)
class HCAOperators:
    attention: Any

    @property
    def is_hbg(self):
        return self.attention._schema.name == "dsv4_hca::attention_hbg"

    @classmethod
    def register(cls):
        import pypto.torch

        from ..deepseek_v4_flash_csa.reduction import ATOMIC_ADD, validate_reduction_mode

        validate_reduction_mode()
        if ATOMIC_ADD:
            raise ValueError("HCA 首版要求 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0，请在导入算子前设置")
        if csa_runtime() == "host_build_graph":
            return cls(pypto.torch.register(decode_hca_tp1_layer_hbg, "dsv4_hca::attention_hbg"))
        return cls(pypto.torch.register(decode_hca_tp1_layer_test, "dsv4_hca::attention"))


def prepare_weights(attention, layer, *, host_scalars=False):
    if attention.compress_ratio != 128:
        raise ValueError("HCA 只接管 C128 层")
    weights = _prepare_weights(attention, None, layer, root_function=_decode_hca_tp1_layer)
    if host_scalars:
        # 初始化阶段仅复制这三个 FP32 常量；调用和图重放期间不读取设备值。
        # 权重更新后必须重新 prepare_weights 并重新捕获图。
        scale = weights["hc_attn_scale"]
        if scale.dtype != torch.float32 or scale.shape != (3,):
            raise ValueError("HCA HBG 要求 hc_attn_scale 为 FP32 [3]")
        values = scale.cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise ValueError("HCA HBG 的 hc_attn_scale 必须为有限值")
        weights.update({f"host_hc_scale{index}": value for index, value in enumerate(values)})
    return weights


class NativeHCACall:
    def __init__(self, operators, weights, hidden, positions, groups, *, layer_name, compact_metadata, output):
        tokens = hidden.shape[0]
        if (hidden.dtype != torch.bfloat16 or hidden.ndim != 3 or hidden.shape[1:] != (4, 4096)
                or not hidden.is_contiguous() or tokens % 6 or not 1 <= tokens // 6 <= 64):
            raise ValueError("HCA 要求连续 BF16 [B*6,4,4096] 输入，B 在 1 到 64 之间")
        if positions.dtype != torch.int64 or positions.shape != (tokens,):
            raise ValueError("HCA positions 必须是 Native INT64 token 向量")
        if output.shape != hidden.shape or output.dtype != hidden.dtype or not output.is_contiguous():
            raise ValueError("HCA 输出必须与层间残差流同形同类型")
        req, views = {}, {}
        for name, (metadata, cache_views) in groups.items():
            value = metadata.decode
            if (metadata.num_prefills or value is None or metadata.num_decodes != tokens // 6
                    or metadata.num_actual_tokens > tokens or metadata.num_actual_tokens % 6
                    or value.seq_lens.numel() != tokens // 6 or value.query_start_loc.numel() != tokens // 6 + 1):
                raise ValueError(f"{name} 不是完整 S6 的 Native decode metadata")
            req[name], views[name] = value, cache_views[0]
        state = views["state"]
        if state.dtype != torch.float32 or state.shape[1:] != (8, 1, 1024):
            raise ValueError("HCA state 必须是 Native FP32 [pages,8,1,1024]")
        for name in ("swa", "compressed"):
            cache = views[name]
            if cache.dtype != torch.bfloat16 or cache.shape[1:] != (32, 1, 512) or not cache.is_contiguous():
                raise ValueError(f"{name} 必须是 Native BF16 [pages,32,1,512]")
        main = req["compressed"]
        cos, sin = main.cos[layer_name], main.sin[layer_name]
        if cos.shape[0] < tokens or sin.shape[0] < tokens:
            raise ValueError("HCA Native RoPE 视图必须覆盖本次输入行")
        cmp_cos, cmp_sin, cmp_slots = compact_metadata
        self.args = dict(weights)
        self.args.update(
            x_hc=hidden, positions=positions, x_out=output,
            freqs_cos=cos[:tokens].view(tokens, 64), freqs_sin=sin[:tokens].view(tokens, 64),
            state=physical_pages(state), state_table=table_storage(req["state"].block_table),
            state_slots=req["state"].slot_mapping, query_bounds=main.query_start_loc, seq_lens=main.seq_lens,
            cmp_cos=cmp_cos.view(-1, 64), cmp_sin=cmp_sin.view(-1, 64), cmp_slots=cmp_slots,
            cmp_cache=views["compressed"], cmp_table=table_storage(main.block_table),
            ori_cache=views["swa"], ori_table=table_storage(req["swa"].block_table),
            ori_slots=req["swa"].slot_mapping,
        )
        self.operators = operators
        # constexpr 不属于运行时 ABI；HBG 则在同一组 Tensor 后显式增加三个 float。
        self.core_args = tuple(self.args[arg.name] for arg in operators.attention._schema.arguments)

    def __call__(self):
        self.operators.attention(*self.core_args)
