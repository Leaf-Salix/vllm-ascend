# SPDX-License-Identifier: Apache-2.0
"""Full precision metadata lifecycle with a model-owned SPMD alias workspace."""

import torch

from ..deepseek_v4_flash_csa_tnd_full_precision.indexer_order_data import INDEXER_TOKEN_MODES
from ..deepseek_v4_flash_csa_tnd_full_precision.oproj_order_data import OPROJ_TOKEN_MODES
from ..deepseek_v4_flash_csa_tnd_full_precision.service import CSAServiceRuntime as FullPrecisionRuntime
from ..deepseek_v4_flash_csa_tnd_full_precision.wkv_order_data import WKV_TOKEN_MODES
from ..deepseek_v4_flash_csa_tnd_full_precision.wkv_profile import STATIC_WKV_MODE, STATIC_WKV_TOKENS
from ..deepseek_v4_flash_csa_tnd_precision.native_adapter import NativeCSACall
from .workspace import pack_norm_weights

STATIC_INDEXER_TOKENS = 6
STATIC_INDEXER_MODE = 1


def select_indexer_mode(tokens, *, static_qualified):
    """Qualified Native static T6 uses the existing T16 reduction profile."""
    if not 1 <= tokens <= len(INDEXER_TOKEN_MODES):
        raise ValueError("Indexer token count is outside the calibrated range")
    if static_qualified and tokens == STATIC_INDEXER_TOKENS:
        return STATIC_INDEXER_MODE
    return INDEXER_TOKEN_MODES[tokens - 1]


class CSAServiceRuntime(FullPrecisionRuntime):
    """Preserve all Native qualification; roots and captured graphs run serially."""

    def __init__(self, attention, operators, max_num_seqs, layer=None):
        super().__init__(attention, operators, max_num_seqs, layer)
        operators.workspace.prepare(attention.wq_a.weight.device)
        self.norm_packet = pack_norm_weights(self.weights)
        keys = {
            (
                STATIC_WKV_MODE if t == STATIC_WKV_TOKENS else WKV_TOKEN_MODES[t - 1],
                INDEXER_TOKEN_MODES[t - 1],
                OPROJ_TOKEN_MODES[t - 1],
            )
            for t in range(1, self.max_hc_pre_tokens + 1)
        }
        operators.workspace.prepare_orders(self.wkv_orders, self.indexer_orders, self.oproj_starts, keys)
        torch.npu.current_stream(attention.wq_a.weight.device).synchronize()

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
        indexer_mode = select_indexer_mode(tokens, static_qualified=self.wkv_profile.metadata_sha256 is not None)
        call_weights = dict(
            self.weights,
            legacy_wkv=self.weights["wkv"],
            legacy_weights_proj=self.weights["weights_proj"],
            wkv=self.wkv_weight,
            weights_proj=self.indexer_projection_weight,
            wkv_order=self.wkv_orders[mode],
            indexer_order=self.indexer_orders[indexer_mode],
            oproj_start=self.oproj_starts[OPROJ_TOKEN_MODES[tokens - 1]],
        )
        call_weights.update(
            self.operators.workspace.arguments(hidden),
            norm_packet=self.norm_packet,
            order_packet=self.operators.workspace.order_packets[(mode, indexer_mode, OPROJ_TOKEN_MODES[tokens - 1])],
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
        # Warmup may switch streams before the first capture. Finish the whole
        # root before returning; captured replay retains ordinary stream order.
        if not torch.npu.is_current_stream_capturing():
            torch.npu.current_stream(hidden.device).synchronize()
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
