# SPDX-License-Identifier: Apache-2.0
"""HCA 的整层服务调用，复用 CSA 的入口判定与按步缓存 metadata 策略。"""

from ..deepseek_v4_flash_csa.service import AttentionServiceBase
from ..deepseek_v4_flash_csa.service_config import MAX_BATCH_SIZE
from .native_adapter import NativeHCACall, prepare_weights


class HCAServiceRuntime(AttentionServiceBase):
    def __init__(self, attention, operators, max_num_seqs, layer):
        self.wrapper = attention.dsa_attn
        self.layer_name = self.wrapper.dsa_attn.layer_name
        self.operators = operators
        self.weights = prepare_weights(attention, layer, host_scalars=operators.is_hbg)
        self.prefixes = {
            "swa": self.wrapper.swa_cache_layer.prefix,
            "compressed": self.layer_name,
            "state": attention.compressor.state_cache.prefix,
        }
        self.batch_capacity = min(max_num_seqs, MAX_BATCH_SIZE)

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector,
            notify_kv_cache_written,
            wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.memcache_comm_fence import record_attention_compute_start

        metadata = {name: context.attn_metadata[prefix] for name, prefix in self.prefixes.items()}
        wait_for_kv_layer_from_connector(self.layer_name)
        compact = self._compact_metadata(context, metadata["compressed"].decode)
        compressed, swa, state = kv_cache[:3]
        groups = {
            "compressed": (metadata["compressed"], (compressed,)),
            "swa": (metadata["swa"], (swa,)),
            "state": (metadata["state"], (state,)),
        }
        call = NativeHCACall(
            self.operators, self.weights, hidden, positions, groups,
            layer_name=self.layer_name, compact_metadata=compact, output=output,
        )
        record_attention_compute_start()
        call()
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
