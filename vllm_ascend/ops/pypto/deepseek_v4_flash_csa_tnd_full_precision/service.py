# SPDX-License-Identifier: Apache-2.0
"""完整PTO TND的单根服务：Host只绑定原生metadata和权重，不启动分阶段算子。"""

import torch
import torch_npu

from ..deepseek_v4_flash_csa_tnd_precision.native_adapter import NativeCSACall
from ..deepseek_v4_flash_csa_tnd_precision.service import CSAServiceRuntime as BaselineRuntime
from .indexer_order_data import INDEXER_K_ORDERS, INDEXER_TILE_ORDERS, INDEXER_TOKEN_MODES
from .oproj_order_data import OPROJ_TILE_STARTS, OPROJ_TOKEN_MODES
from .wkv_order_data import WKV_COLUMN_ORDERS, WKV_K_ORDERS, WKV_TOKEN_MODES
from .wkv_profile import WKVProfile


class CSAServiceRuntime(BaselineRuntime):
    """原混合基准保留；本路径用PTO发布各Native计算和舍入边界。"""

    restored_modules = ("hc_pre", "input_norm", "wkv", "indexer_weights", "o_proj", "hc_post")
    # Native mDim changes above 256 rows; the current K32/22 partition is for mDim1.
    max_hc_pre_tokens = 256
    single_pto_entry = True

    def eligible(self, context, hidden, positions):
        if hidden.shape[0] > self.max_hc_pre_tokens or not super().eligible(context, hidden, positions):
            return False
        if self.wkv_profile.metadata_sha256 is None:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("PTO O-proj capture requires verified static qualification and PTO warmup")
            return False
        return True

    def __init__(self, attention, operators, max_num_seqs, layer=None):
        super().__init__(attention, operators, max_num_seqs, layer)
        from vllm_ascend.ascend_config import get_ascend_config

        self.wkv_profile = WKVProfile()
        self.wkv_static_requested = get_ascend_config().ascend_compilation_config.enable_static_kernel

        for parameter, shape in (
            (layer.hc_attn_fn, (24, 16384)),
            (layer.hc_attn_scale, (3,)),
            (layer.hc_attn_base, (24,)),
        ):
            if parameter.dtype != torch.float32 or tuple(parameter.shape) != shape or not parameter.is_contiguous():
                raise ValueError("PTO HC-pre requires original contiguous FP32 Native HC parameters")
            if torch_npu.get_npu_format(parameter) not in (0, 2):
                raise ValueError("PTO HC-pre requires original ND Native HC parameters")
        device = attention.wq_a.weight.device
        norm = layer.input_layernorm
        if norm.variance_epsilon != 1e-6 or norm.bias_loaded:
            raise ValueError("PTO input Norm requires Native epsilon=1e-6 without a loaded bias")
        norm_weight = norm.weight
        if norm_weight.dtype != torch.bfloat16 or tuple(norm_weight.shape) != (4096,):
            raise ValueError("PTO input Norm requires original Native BF16 [4096] weight")
        if not norm_weight.is_contiguous() or torch_npu.get_npu_format(norm_weight) not in (0, 2):
            raise ValueError("PTO input Norm requires original contiguous ND weight")
        wkv = layer.self_attn.wkv
        weight = wkv.weight.detach()
        if wkv.bias is not None or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (512, 4096):
            raise ValueError("PTO WKV requires the original unbiased BF16 [512, 4096] Native weight")
        # 仅加载期整理布局；不改dtype、不反量化，也不改变权重数值。
        if torch_npu.get_npu_format(weight) not in (0, 2):
            weight = torch_npu.npu_format_cast(weight, 2)
        self.wkv_weight = weight.contiguous()
        self.wkv_orders = torch.tensor(
            [[WKV_K_ORDERS[index] for index in mode] for mode in WKV_COLUMN_ORDERS],
            dtype=torch.int32,
            device=device,
        )
        projection = self.wrapper.dsa_attn.impl.weights_proj
        weight = projection.weight.detach()
        if projection.bias is not None or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (64, 4096):
            raise ValueError("PTO Indexer projection requires original unbiased BF16 [64, 4096] weight")
        if type(projection.quant_method).__name__ != "AscendUnquantizedLinearMethod" or projection.return_bias:
            raise ValueError("PTO Indexer projection requires Native unquantized tensor output")
        impl = self.wrapper.dsa_attn.impl
        if impl.indexer_heads != 64 or impl.indexer_softmax_scale * impl.indexer_heads**-0.5 != 0.011048543456039806:
            raise ValueError("PTO Indexer projection requires the calibrated Native scale")
        # 保留原权重dtype与值；加载期只把Native NZ布局整理为ND。
        if torch_npu.get_npu_format(weight) not in (0, 2):
            weight = torch_npu.npu_format_cast(weight, 2)
        self.indexer_projection_weight = weight.contiguous()
        self.indexer_orders = torch.tensor(
            [[[INDEXER_K_ORDERS[index] for index in row] for row in mode] for mode in INDEXER_TILE_ORDERS],
            dtype=torch.int32,
            device=device,
        )
        from vllm_ascend.utils import olora_tp_enable

        if get_ascend_config().weight_nz_mode != 2 or olora_tp_enable():
            raise ValueError("PTO O-proj currently requires the calibrated A3 NZ2 grouped TP1 path")
        if impl.wo_b.bias is not None:
            raise ValueError("PTO O-proj requires the original unbiased W8A8 WO-B")
        self.oproj_starts = torch.tensor(OPROJ_TILE_STARTS, dtype=torch.int32, device=device)
        del self.native_heads, self.native_attention_out

    def qualify_static_wkv(self, *, metadata_sha256, reselected):
        """加载者须先验证静态包/SDK/SOC/确定性并完成实际 reselection，再通知每层。"""
        self.wkv_profile.qualify_static(metadata_sha256=metadata_sha256, reselected=reselected)

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector,
            notify_kv_cache_written,
            wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.memcache_comm_fence import record_attention_compute_start

        if self.wkv_profile.metadata_sha256 is None:
            raise RuntimeError("PTO O-proj requires explicit verified static qualification before execution")
        metadata = {name: context.attn_metadata[prefix] for name, prefix in self.prefixes.items()}
        hadamard = metadata["indexer"].hadamard
        if hadamard is None:
            raise ValueError("Native indexer metadata did not provide the Hadamard matrix")
        if self._hadamard is not hadamard:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("PTO CSA requires Native metadata warmup before graph capture")
            self.weights["hadamard_idx"] = hadamard.detach().T.to(torch.bfloat16).contiguous()
            self._hadamard = hadamard

        wait_for_kv_layer_from_connector(self.layer_name)
        compact = {name: self._compact_metadata(context, metadata[name].decode) for name in ("compressed", "indexer")}
        compressed, swa, state, indexer_state, indexer_key, indexer_scale = kv_cache
        groups = {
            "swa": (metadata["swa"], (swa,)),
            "compressed": (metadata["compressed"], (compressed,)),
            "state": (metadata["state"], (state,)),
            "indexer": (metadata["indexer"], (indexer_key, indexer_scale)),
            "indexer_state": (metadata["indexer_state"], (indexer_state,)),
        }
        tokens = hidden.shape[0]
        if not 1 <= tokens <= self.max_hc_pre_tokens:
            raise ValueError("PTO HC-pre currently supports the Native mDim1 range, 1..256 token rows")
        mode = self.wkv_profile.select(
            tokens,
            WKV_TOKEN_MODES,
            capturing=torch.npu.is_current_stream_capturing(),
            static_requested=self.wkv_static_requested,
        )
        call_weights = dict(
            self.weights,
            legacy_wkv=self.weights["wkv"],
            legacy_weights_proj=self.weights["weights_proj"],
            wkv=self.wkv_weight,
            weights_proj=self.indexer_projection_weight,
            wkv_order=self.wkv_orders[mode],
            indexer_order=self.indexer_orders[INDEXER_TOKEN_MODES[tokens - 1]],
            oproj_start=self.oproj_starts[OPROJ_TOKEN_MODES[tokens - 1]],
        )
        call = NativeCSACall(
            self.operators,
            call_weights,
            hidden,
            positions,
            groups,
            layer_name=self.layer_name,
            compact_metadata=compact,
            buffers={"idx_topk_scores": self.scores[:tokens], "idx_topk": self.topk[:tokens], "x_out": output},
        )
        record_attention_compute_start()
        self.operators.attention(*call.core_args)
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
