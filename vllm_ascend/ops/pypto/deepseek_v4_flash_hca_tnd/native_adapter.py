# SPDX-License-Identifier: Apache-2.0
"""TND HCA 的描述符绑定：只检查 shape/dtype，Q/L 数值留在设备。"""

from dataclasses import dataclass
from typing import Any

import torch

from ..deepseek_v4_flash_csa.native_adapter import prepare_weights as _prepare_weights
from ..deepseek_v4_flash_csa.native_storage import physical_pages, table_storage
from ..deepseek_v4_flash_csa.nz_mode import QUANT_WEIGHT_NZ
from .entry import (
    _decode_hca_tp1_layer,
    decode_hca_tp1_layer_test,
    decode_hca_tp1_layer_tall_test,
    decode_hca_tp1_layer_long_test,
)

# 64K实测有回退，当前只在至少128K的长历史区间采用；使用Native已有Host字段。
# 数学正确性不依赖这个阈值，它只选择Q_B/O_B供数策略；缺少Host字段则保持普通入口。
LONG_HISTORY_MIN = 131072
TALL_PROJECTION_MIN_TOKENS = 129
TALL_PROJECTION_MAX_TOKENS = 192
LONG_ATTENTION_MIN_TOKENS = 33


def use_tall_projection(tokens, max_seq_lens):
    """只检查已知shape和Host整数，不对设备Tensor执行item、int或bool。"""
    return (QUANT_WEIGHT_NZ and TALL_PROJECTION_MIN_TOKENS <= tokens <= TALL_PROJECTION_MAX_TOKENS
            and isinstance(max_seq_lens, int) and max_seq_lens >= LONG_HISTORY_MIN)


def use_long_attention(tokens, max_seq_lens):
    """K512用于长历史且T>32；短历史/小T保留KV驻留L1的K128流水。

    B8/B16/B28的长历史实测支持此区间；B4及8K/B32反例要求保留原路径。
    这里读的是Native既有Host max_seq_lens和输入shape；每个请求的实际Q/L
    仍在设备端裁剪，混长请求不会被补成同长度。
    """
    return (QUANT_WEIGHT_NZ and tokens >= LONG_ATTENTION_MIN_TOKENS
            and isinstance(max_seq_lens, int) and max_seq_lens >= LONG_HISTORY_MIN)


def token_capacity_view(value, tokens, width):
    """借用 Native 短视图背后的容量，不分配、补齐或读取设备数据。

    Native RoPE/slots 的存储按输入容量分配，decode 描述符可能只暴露有效行。
    与 CSA TND 接口相同，先核实既有存储足够，再建立同地址的容量描述符。
    失活 token 的业务访问仍由设备 Q/L 与 slots 保护。
    """
    if value.ndim < 2 or value.numel() != value.shape[0] * width or not value.is_contiguous():
        raise ValueError("Native token 视图必须连续，且行宽固定")
    needed = (value.storage_offset() + tokens * width) * value.element_size()
    if needed > value.untyped_storage().nbytes():
        raise ValueError("Native 既有存储不足以覆盖 token 容量")
    return value.as_strided((tokens, width), (width, 1), value.storage_offset())


@dataclass(frozen=True)
class HCAOperators:
    attention: Any
    attention_tall: Any = None
    attention_long: Any = None

    @classmethod
    def register(cls, *, config=None):
        import pypto.torch

        from ..deepseek_v4_flash_csa.reduction import ATOMIC_ADD, validate_reduction_mode

        validate_reduction_mode()
        if ATOMIC_ADD:
            raise ValueError("HCA 首版要求 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0，请在导入算子前设置")
        return cls(
            pypto.torch.register(decode_hca_tp1_layer_test, "dsv4_hca_tnd::attention", config=config),
            pypto.torch.register(decode_hca_tp1_layer_tall_test, "dsv4_hca_tnd::attention_tall", config=config)
            if QUANT_WEIGHT_NZ else None,
            pypto.torch.register(decode_hca_tp1_layer_long_test, "dsv4_hca_tnd::attention_long", config=config)
            if QUANT_WEIGHT_NZ else None,
        )


def prepare_weights(attention, layer):
    """沿用 Native 已加载的权重类型与 NZ 布局，不从设备提取 Host 标量。"""
    if attention.compress_ratio != 128:
        raise ValueError("HCA 只接管 C128 层")
    return _prepare_weights(attention, None, layer, root_function=_decode_hca_tp1_layer)


class NativeHCACall:
    def __init__(self, operators, weights, hidden, positions, groups, *, layer_name, compact_metadata, output):
        tokens = hidden.shape[0]
        if (hidden.dtype != torch.bfloat16 or hidden.ndim != 3 or hidden.shape[1:] != (4, 4096)
                or not hidden.is_contiguous() or not 1 <= tokens <= 384):
            raise ValueError("TND HCA 要求连续 BF16 [T,4,4096] 输入，1≤T≤384")
        if positions.dtype != torch.int64 or positions.shape != (tokens,):
            raise ValueError("HCA positions 必须是 Native INT64 token 向量")
        if output.shape != hidden.shape or output.dtype != hidden.dtype or not output.is_contiguous():
            raise ValueError("HCA 输出必须与层间残差流同形同类型")
        req, views = {}, {}
        for name, (metadata, cache_views) in groups.items():
            value = metadata.decode
            if (metadata.num_prefills or value is None or metadata.num_actual_tokens > tokens
                    or value.seq_lens.ndim != 1 or not 1 <= value.seq_lens.numel() <= 64
                    or value.query_start_loc.numel() != value.seq_lens.numel() + 1):
                raise ValueError(f"{name} 不满足 TND decode metadata 描述符合同")
            # Native 的 max_seqlen_q 可能包含一个吸收大段 padding 的失活请求，
            # 优先使用已有Host字段max_seqlen_q_actual检查真实请求上限；不回读设备Q/L。
            # 设备 L=0 分支屏蔽dummy，缺少actual字段时按旧max保守判定。
            query_bound = getattr(value, "max_seqlen_q_actual", None)
            if query_bound is None:
                query_bound = value.max_seqlen_q
            if not 1 <= query_bound <= 6:
                raise ValueError(f"{name} 仅支持 Dspark=5 下真实 query 长度 1～6")
            if value.seq_lens.dtype != torch.int32 or value.query_start_loc.dtype != torch.int32:
                raise ValueError(f"{name} 的 Native Q/L 必须是 INT32 device Tensor")
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
        cmp_cos, cmp_sin, cmp_slots = compact_metadata
        self.args = dict(weights)
        self.args.update(
            x_hc=hidden, positions=positions, x_out=output,
            freqs_cos=token_capacity_view(cos, tokens, 64), freqs_sin=token_capacity_view(sin, tokens, 64),
            state=physical_pages(state), state_table=table_storage(req["state"].block_table),
            state_slots=token_capacity_view(req["state"].slot_mapping, tokens, 2),
            query_bounds=main.query_start_loc, seq_lens=main.seq_lens,
            cmp_cos=cmp_cos.view(-1, 64), cmp_sin=cmp_sin.view(-1, 64), cmp_slots=cmp_slots,
            cmp_cache=views["compressed"], cmp_table=table_storage(main.block_table),
            ori_cache=views["swa"], ori_table=table_storage(req["swa"].block_table),
            ori_slots=token_capacity_view(req["swa"].slot_mapping, tokens, 2),
        )
        self.operators = operators
        selected_tall = operators.attention_tall is not None and use_tall_projection(
            tokens, getattr(main, "max_seq_lens", None),
        )
        self.policy = "tall_projection" if selected_tall else "default"
        self.kernel = operators.attention_tall if selected_tall else operators.attention
        if not selected_tall and operators.attention_long is not None and use_long_attention(
            tokens, getattr(main, "max_seq_lens", None),
        ):
            self.policy = "long_attention"
            self.kernel = operators.attention_long
        # 根ABI全部为Native device Tensor；每次调用消费本轮设备内容。
        self.core_args = tuple(self.args[arg.name] for arg in self.kernel._schema.arguments)

    def __call__(self):
        self.kernel(*self.core_args)
