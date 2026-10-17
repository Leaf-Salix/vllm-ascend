# SPDX-License-Identifier: Apache-2.0
"""完整PTO TND的分阶段候选：HC-pre至HC-post均由PTO执行，保留真实TND边界。"""

import torch
import torch_npu

from ..deepseek_v4_flash_csa_tnd_precision.native_adapter import NativeCSACall
from ..deepseek_v4_flash_csa_tnd_precision.service import CSAServiceRuntime as BaselineRuntime
from .hc_post import hc_post_test
from .hc_pre import hc_pre_test
from .indexer_order_data import INDEXER_K_ORDERS, INDEXER_TILE_ORDERS, INDEXER_TOKEN_MODES
from .indexer_projection import indexer_projection
from .input_norm import input_norm_test
from .o_proj import o_proj_test
from .oproj_order_data import OPROJ_TILE_STARTS, OPROJ_TOKEN_MODES
from .wkv import wkv_test
from .wkv_order_data import WKV_COLUMN_ORDERS, WKV_K_ORDERS, WKV_TOKEN_MODES
from .wkv_profile import WKVProfile


class CSAServiceRuntime(BaselineRuntime):
    """原混合基准保留；本路径用PTO发布各Native计算和舍入边界。"""

    restored_modules = ("hc_pre", "input_norm", "wkv", "indexer_weights", "o_proj", "hc_post")
    # Native mDim changes above 256 rows; the current K32/22 partition is for mDim1.
    max_hc_pre_tokens = 256

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
        import pypto.torch

        from vllm_ascend.ascend_config import get_ascend_config

        self.wkv_profile = WKVProfile()
        self.wkv_static_requested = get_ascend_config().ascend_compilation_config.enable_static_kernel

        self.pto_hc_pre = pypto.torch.register(hc_pre_test, "dsv4_csa_tnd_full_precision::hc_pre")
        for parameter, shape in (
            (layer.hc_attn_fn, (24, 16384)),
            (layer.hc_attn_scale, (3,)),
            (layer.hc_attn_base, (24,)),
        ):
            if parameter.dtype != torch.float32 or tuple(parameter.shape) != shape or not parameter.is_contiguous():
                raise ValueError("PTO HC-pre requires original contiguous FP32 Native HC parameters")
            if torch_npu.get_npu_format(parameter) not in (0, 2):
                raise ValueError("PTO HC-pre requires original ND Native HC parameters")
        tokens = self.native_attention_out.shape[0]
        device = self.native_attention_out.device
        norm = layer.input_layernorm
        if norm.variance_epsilon != 1e-6 or norm.bias_loaded:
            raise ValueError("PTO input Norm requires Native epsilon=1e-6 without a loaded bias")
        norm_weight = norm.weight
        if norm_weight.dtype != torch.bfloat16 or tuple(norm_weight.shape) != (4096,):
            raise ValueError("PTO input Norm requires original Native BF16 [4096] weight")
        if not norm_weight.is_contiguous() or torch_npu.get_npu_format(norm_weight) not in (0, 2):
            raise ValueError("PTO input Norm requires original contiguous ND weight")
        self.pto_input_norm = pypto.torch.register(input_norm_test, "dsv4_csa_tnd_full_precision::input_norm")
        self.normed = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=device)
        self.norm_rstd = torch.empty((tokens, 8), dtype=torch.float32, device=device)
        wkv = layer.self_attn.wkv
        weight = wkv.weight.detach()
        if wkv.bias is not None or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (512, 4096):
            raise ValueError("PTO WKV requires the original unbiased BF16 [512, 4096] Native weight")
        # 仅加载期整理布局；不改dtype、不反量化，也不改变权重数值。
        if torch_npu.get_npu_format(weight) not in (0, 2):
            weight = torch_npu.npu_format_cast(weight, 2)
        self.wkv_weight = weight.contiguous()
        self.wkv_projected = torch.empty((tokens, 512), dtype=torch.bfloat16, device=device)
        self.wkv_orders = torch.tensor(
            [[WKV_K_ORDERS[index] for index in mode] for mode in WKV_COLUMN_ORDERS],
            dtype=torch.int32,
            device=device,
        )
        self.pto_wkv = pypto.torch.register(wkv_test, "dsv4_csa_tnd_full_precision::wkv")
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
        self.indexer_projection_raw = torch.empty((tokens, 64), dtype=torch.bfloat16, device=device)
        self.indexer_projection_scaled = torch.empty((tokens, 64), dtype=torch.bfloat16, device=device)
        self.indexer_weights = torch.empty((tokens, 64), dtype=torch.float16, device=device)
        self.indexer_orders = torch.tensor(
            [[[INDEXER_K_ORDERS[index] for index in row] for row in mode] for mode in INDEXER_TILE_ORDERS],
            dtype=torch.int32,
            device=device,
        )
        self.pto_indexer_projection = pypto.torch.register(
            indexer_projection, "dsv4_csa_tnd_full_precision::indexer_projection"
        )
        from vllm_ascend.utils import olora_tp_enable

        if get_ascend_config().weight_nz_mode != 2 or olora_tp_enable():
            raise ValueError("PTO O-proj currently requires the calibrated A3 NZ2 grouped TP1 path")
        if impl.wo_b.bias is not None:
            raise ValueError("PTO O-proj requires the original unbiased W8A8 WO-B")
        self.oproj_starts = torch.tensor(OPROJ_TILE_STARTS, dtype=torch.int32, device=device)
        self.oproj_projected = torch.empty((tokens, 8192), dtype=torch.bfloat16, device=device)
        self.oproj_quantized = torch.empty((tokens, 8192), dtype=torch.int8, device=device)
        self.oproj_token_scale = torch.empty((tokens, 1), dtype=torch.float32, device=device)
        self.pto_o_proj = pypto.torch.register(o_proj_test, "dsv4_csa_tnd_full_precision::o_proj")
        self.hc_mixed = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=device)
        self.hc_post_gates = torch.empty((tokens, 4), dtype=torch.float32, device=device)
        self.hc_comb_gates = torch.empty((tokens, 16), dtype=torch.float32, device=device)
        self.pto_hc_post = pypto.torch.register(hc_post_test, "dsv4_csa_tnd_full_precision::hc_post")

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
        mixed, post, comb = self.hc_mixed[:tokens], self.hc_post_gates[:tokens], self.hc_comb_gates[:tokens]
        self.pto_hc_pre(
            hidden, self.layer.hc_attn_fn, self.layer.hc_attn_scale, self.layer.hc_attn_base, mixed, post, comb
        )
        normed = self.normed[:tokens]
        self.pto_input_norm(mixed, self.layer.input_layernorm.weight, normed, self.norm_rstd[:tokens])
        kv_projected = self.wkv_projected[:tokens]
        mode = self.wkv_profile.select(
            tokens,
            WKV_TOKEN_MODES,
            capturing=torch.npu.is_current_stream_capturing(),
            static_requested=self.wkv_static_requested,
        )
        order = self.wkv_orders[mode]
        self.pto_wkv(normed, self.wkv_weight, order, kv_projected)
        if kv_projected.dtype != torch.bfloat16 or tuple(kv_projected.shape) != (tokens, 512):
            raise ValueError("PTO WKV must publish BF16 [tokens, 512]")
        if not kv_projected.is_contiguous() or torch_npu.get_npu_format(kv_projected) not in (0, 2):
            raise ValueError("PTO WKV must publish contiguous ND storage")
        indexer_weights = self.indexer_weights[:tokens]
        indexer_order = self.indexer_orders[INDEXER_TOKEN_MODES[tokens - 1]]
        self.pto_indexer_projection(
            normed,
            self.indexer_projection_weight,
            indexer_order,
            self.indexer_projection_raw[:tokens],
            self.indexer_projection_scaled[:tokens],
            indexer_weights,
        )
        if indexer_weights.dtype != torch.float16 or tuple(indexer_weights.shape) != (tokens, 64):
            raise ValueError("PTO Indexer weights must publish FP16 [tokens, 64]")
        if not indexer_weights.is_contiguous() or torch_npu.get_npu_format(indexer_weights) not in (0, 2):
            raise ValueError("PTO Indexer weights must publish contiguous ND storage")
        call_weights = dict(
            self.weights,
            native_normed=normed,
            # 保留CSA内部的历史边界名称，实际producer已改为PTO。
            native_kv_projected=kv_projected,
            native_indexer_weights=indexer_weights,
            native_heads=self.native_heads[:tokens],
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
        call()
        self.pto_o_proj(
            self.native_heads[:tokens],
            self.weights["wo_a"],
            self.weights["wo_b"],
            self.weights["wo_b_scale"],
            self.oproj_starts[OPROJ_TOKEN_MODES[tokens - 1]],
            self.oproj_projected[:tokens],
            self.oproj_quantized[:tokens],
            self.oproj_token_scale[:tokens],
            self.native_attention_out[:tokens],
        )
        # Directly publish into the caller's output; no Native HC-post call remains.
        self.pto_hc_post(self.native_attention_out[:tokens], hidden, post, comb.view(tokens, 16), output)
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
