# SPDX-License-Identifier: Apache-2.0
"""完整PTO TND的分阶段候选：当前仅将HC-post恢复为PTO。"""

import torch
import torch_npu

from vllm_ascend.device.device_op import DeviceOperator

from ..deepseek_v4_flash_csa_tnd_precision.native_adapter import NativeCSACall
from ..deepseek_v4_flash_csa_tnd_precision.service import CSAServiceRuntime as BaselineRuntime
from .hc_post import hc_post_test


class CSAServiceRuntime(BaselineRuntime):
    """原基准保持可运行；剩余Native边界须后续逐项对拍再恢复。"""

    restored_modules = ("hc_post",)

    def __init__(self, attention, operators, max_num_seqs, layer=None):
        super().__init__(attention, operators, max_num_seqs, layer)
        import pypto.torch

        self.pto_hc_post = pypto.torch.register(hc_post_test, "dsv4_csa_tnd_full_precision::hc_post")

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector,
            notify_kv_cache_written,
            wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.memcache_comm_fence import record_attention_compute_start

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
        mixed, post, comb = self.layer.hc_pre(
            hidden, self.layer.hc_attn_fn, self.layer.hc_attn_scale, self.layer.hc_attn_base
        )
        normed = self.layer.input_layernorm(mixed)
        native_kv_projected = self.layer.self_attn.wkv(normed)
        if native_kv_projected.dtype != torch.bfloat16 or tuple(native_kv_projected.shape) != (tokens, 512):
            raise ValueError("Native WKV must publish BF16 [tokens, 512]")
        if not native_kv_projected.is_contiguous() or torch_npu.get_npu_format(native_kv_projected) not in (0, 2):
            raise ValueError("Native WKV must publish contiguous ND storage")
        impl = self.wrapper.dsa_attn.impl
        native_indexer_weights = DeviceOperator.prepare_dsa_indexer_weights(
            impl.weights_proj(normed) * (impl.indexer_softmax_scale * impl.indexer_heads**-0.5)
        )
        if native_indexer_weights.dtype != torch.float16 or tuple(native_indexer_weights.shape) != (tokens, 64):
            raise ValueError("Native Indexer weights must publish FP16 [tokens, 64]")
        if not native_indexer_weights.is_contiguous() or torch_npu.get_npu_format(native_indexer_weights) not in (0, 2):
            raise ValueError("Native Indexer weights must publish contiguous ND storage")
        call_weights = dict(
            self.weights,
            native_normed=normed,
            native_kv_projected=native_kv_projected,
            native_indexer_weights=native_indexer_weights,
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
        impl._forward_o_proj(self.native_heads[:tokens].view(tokens, 64, 512), self.native_attention_out[:tokens])
        # Directly publish into the caller's output; no Native HC-post call remains.
        self.pto_hc_post(self.native_attention_out[:tokens], hidden, post, comb.view(tokens, 16), output)
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
