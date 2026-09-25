# SPDX-License-Identifier: Apache-2.0
"""Per-layer PTO resources and binding of the current Native service buffers."""

import torch

from .native_adapter import NativeCSACall, prepare_weights
from .service_config import MAX_BATCH_SIZE, QUERY_TOKENS

# 同一个 decode step 里，compressor_metadata 的入参只跟 KV cache group 有关、与层无关：
# vLLM 让同一个 attn group 的所有层共用同一个 metadata 对象（`vllm_ascend/worker/
# model_runner_v1.py` 里 `attn_metadata_dict[layer_name] = attn_metadata_i`），所以各层
# 拿到的 cos / sin / slot_mapping 完全相同，这个算子被按层数重算了同样多遍。
# Native 路径每算一次就紧跟一个 compressor 把它消费掉，开销摊在 70us 量级的算子里；
# PTO 把 compressor 融进了自己的 kernel，这两次调用就裸露成 kernel 正前方的串行开销
# （16 卡 eager 实测每层约 30us，21 层合计 600us 级）。按 decode metadata 的对象身份
# 缓存，一个 step 只算一次。
_COMPACT_METADATA_CACHE = "pto_csa_compact_compressor_metadata"


class CSAServiceRuntime:
    def __init__(self, attention, operators, max_num_seqs):
        self.wrapper = attention.dsa_attn
        self.layer_name = self.wrapper.dsa_attn.layer_name
        self.operators = operators
        self.weights = prepare_weights(attention, None)
        self.prefixes = {
            "swa": self.wrapper.swa_cache_layer.prefix,
            "compressed": self.layer_name,
            "state": attention.compressor.state_cache.prefix,
            "indexer": attention.indexer.k_cache.prefix,
            "indexer_state": attention.indexer.compressor.state_cache.prefix,
        }
        self.batch_capacity = min(max_num_seqs, MAX_BATCH_SIZE)
        tokens = self.batch_capacity * QUERY_TOKENS
        device = attention.wq_a.weight.device
        # Allocate once after weight loading; these buffers are private to one
        # layer. Output belongs to the Native forward call / captured graph.
        self.scores = torch.empty((tokens, 512), dtype=torch.float32, device=device)
        self.topk = torch.empty((tokens, 512), dtype=torch.int32, device=device)
        self._hadamard = None

    def eligible(self, context, hidden, positions):
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return False
        if hidden.ndim != 2 or hidden.shape[1] != 4096 or hidden.dtype != torch.bfloat16:
            return False
        tokens = hidden.shape[0]
        if tokens % QUERY_TOKENS or not 1 <= tokens // QUERY_TOKENS <= self.batch_capacity:
            return False
        if not hidden.is_contiguous() or positions.dtype != torch.int64 or positions.shape != (tokens,):
            return False
        batch = tokens // QUERY_TOKENS
        for prefix in self.prefixes.values():
            item = metadata.get(prefix)
            # 补位档位下这两个计数不再相等：num_actual_tokens 是**实际** token 数，
            # num_decodes 是**补齐后**的请求数（实测 18 与 4，整档 24）。补位请求
            # 由 kernel 内的 seq_lens 判据屏蔽，这里只要求真实部分是完整六行请求。
            if item is None or item.num_prefills or item.num_actual_tokens > tokens:
                return False
            if item.num_actual_tokens % QUERY_TOKENS or item.num_decodes != batch:
                return False
            req = item.decode
            if req is None or req.seq_lens.numel() != batch or req.query_start_loc.numel() != batch + 1:
                return False
            actual = item.num_actual_tokens // QUERY_TOKENS
            if req.max_seqlen_q != QUERY_TOKENS or req.num_reqs_actual not in (None, actual):
                return False
            if req.ori_win_right not in (None, 0) or req.dspark_swa_indices is not None:
                return False
        main = metadata[self.prefixes["compressed"]]
        if main.num_actual_tokens < tokens and main.decode.cos[self.layer_name].shape[0] < tokens:
            # Native 的 decode RoPE 视图按实际 token 切片。图捕获发生在无补位的满档
            # dummy 上，捕获到的就是整档视图，重放不受影响；只有 eager 下真出现补位
            # 才会切短。那种情况下整档算不出来，让本层回退 Native，
            # 而不是另建一份冗余缓冲去凑。
            return False
        swa = metadata[self.prefixes["swa"]].decode
        return swa.ori_win_left in (None, 127)

    def _compact_metadata(self, context, req):
        """取这一步该 KV cache group 的 (cos, sin, slot_mapping)，同 step 内跨层复用。

        缓存挂在 forward context 的 additional_kwargs 上：它每次前向都由
        `vllm_ascend/platform.py` 的 set_additional_forward_context 新建，
        所以不会跨 step 残留。aclgraph 下被捕获进图的同样只有第一层那一次调用，
        重放时后面各层读的是同一块固定地址的输出，语义与逐层重算完全一致。
        """
        cache = context.additional_kwargs.setdefault(_COMPACT_METADATA_CACHE, {})
        # id() 在对象回收后可能被复用，所以连同对象本身一起存下来比对。
        cached = cache.get(id(req))
        if cached is not None and cached[0] is req:
            return cached[1]
        value = self.wrapper.dsa_attn.impl._compute_compressor_metadata(req)
        cache[id(req)] = (req, value)
        return value

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector, notify_kv_cache_written, wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.memcache_comm_fence import record_attention_compute_start

        metadata = {name: context.attn_metadata[prefix] for name, prefix in self.prefixes.items()}
        # Native creates this Hadamard table when it builds attention metadata.
        # Transform once in an ordinary warmup call, never during capture/replay.
        hadamard = metadata["indexer"].hadamard
        if hadamard is None:
            raise ValueError("Native indexer metadata did not provide the Hadamard matrix")
        if self._hadamard is not hadamard:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("PTO CSA requires Native metadata warmup before graph capture")
            self.weights["hadamard_idx"] = hadamard.detach().T.to(torch.bfloat16).contiguous()
            self._hadamard = hadamard

        wait_for_kv_layer_from_connector(self.layer_name)
        # Release Native creates compact rows at the consumer on this stream.
        # Pass those exact device tensors to CSA, without expanding them or
        # introducing the main-branch DeviceMetadataExecutor API.
        compact = {
            name: self._compact_metadata(context, metadata[name].decode)
            for name in ("compressed", "indexer")
        }
        compressed, swa, state, indexer_state, indexer_key, indexer_scale = kv_cache
        groups = {
            "swa": (metadata["swa"], (swa,)),
            "compressed": (metadata["compressed"], (compressed,)),
            "state": (metadata["state"], (state,)),
            "indexer": (metadata["indexer"], (indexer_key, indexer_scale)),
            "indexer_state": (metadata["indexer_state"], (indexer_state,)),
        }
        tokens = hidden.shape[0]
        call = NativeCSACall(
            self.operators, self.weights, hidden, positions, groups, layer_name=self.layer_name,
            compact_metadata=compact,
            buffers={"idx_topk_scores": self.scores[:tokens], "idx_topk": self.topk[:tokens], "attn_out": output},
        )
        # A single fused call publishes all KV writes. Notify the connector on
        # the same stream after that call; no global synchronization is needed.
        record_attention_compute_start()
        call()
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
