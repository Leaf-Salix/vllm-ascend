# SPDX-License-Identifier: Apache-2.0
"""HCA TND整模型服务：固定图容量、保留设备Q/L，并校验重放的地址与策略。

总体流程：初始化仍绑定Native权重；每步沿用Native KV生命周期，按图容量调用
Native C128 compact producer，再调用未经修改的完整HCA MPMD算子。根Tensor
内容可随decode变化，Python只检查已有Host计数、shape/stride、dtype与地址。
捕获时记录实际PTO提交合同；重放前复用框架已有的TND合同槽位逐层核对。
Native/空路径的捕获不被当成PTO图；超出容量、地址变化或分档变化交回正常派发。
单算子性能基准继续使用原service，本模块只由显式TND整模型选择。
"""

from copy import copy

import torch

from ..deepseek_v4_flash_csa_spmd_tnd.graph_contract import TNDGraphContract, tensor_descriptor
from .native_adapter import NativeHCACall, token_capacity_view, use_long_attention, use_tall_projection
from .service import HCATNDServiceRuntime, MAX_HISTORY, MAX_QUERY_TOKENS

COMPRESS_RATIO = 128
_COMPACT_CACHE = "pto_hca_tnd_graph_compact_metadata"
_CONTRACTS = "pto_csa_tnd_graph_contracts"  # 复用ACLGraphWrapper的通用合同，按实际层名区分。


def compact_capacity(tokens, requests):
    """与Native C128 builder一致的容量上界，不是本步真实关闭的压缩组数量。"""
    return min(tokens, tokens // COMPRESS_RATIO + requests)


def selected_policy(tokens, maximum):
    if use_tall_projection(tokens, maximum):
        return "tall_projection"
    if use_long_attention(tokens, maximum):
        return "long_attention"
    return "default"


class HCATNDGraphServiceRuntime(HCATNDServiceRuntime):
    def __init__(self, attention, operators, max_num_seqs, layer, max_num_batched_tokens=384):
        super().__init__(attention, operators, max_num_seqs, layer, max_num_batched_tokens)
        self.submissions = 0
        self.empty_calls = 0
        self.last_policy = None

    def record_graph_route(self, context, hidden, positions):
        if getattr(context, "capturing", False):
            context.additional_kwargs.setdefault(_CONTRACTS, {})[self.layer_name] = TNDGraphContract(
                self, hidden.shape[0], None
            )

    def handle_empty_call(self, context, hidden, positions, output):
        metadata = context.attn_metadata
        empty = hidden.numel() == 0
        if not empty and metadata and not getattr(context, "is_draft_model", False):
            items = [metadata.get(prefix) for prefix in self.prefixes.values()]
            empty = all(item is not None and not item.num_prefills and item.num_actual_tokens == 0 for item in items)
        if empty:
            output.copy_(hidden)
            self.empty_calls += 1
        return empty

    def _compact_metadata(self, context, req, tokens):
        """图内只用设备L屏蔽dummy，不冻结本步实际请求数或增加Host回读。

        C128与CSA C4的压缩比不同，容量必须按128计算。Native producer对
        start_pos<0的请求生成零行；where作为图内操作，每次重放读取本步L。
        原Native metadata及原start_pos不修改，输出仍由原producer生成。
        """
        cache = context.additional_kwargs.setdefault(_COMPACT_CACHE, {})
        key = (id(req), tokens)
        cached = cache.get(key)
        if cached is not None and cached[0] is req:
            return cached[1]
        requests = req.seq_lens.numel()
        adapted = copy(req)
        adapted.num_reqs_actual = requests
        adapted.num_compressed_tokens = compact_capacity(tokens, requests)
        adapted.start_pos = torch.where(req.seq_lens > 0, req.start_pos, -1)
        value = self.wrapper.dsa_attn.impl._compute_compressor_metadata(adapted)
        cache[key] = (req, value)
        return value

    def graph_signature(self, context, tokens):
        """验证捕获的设备描述符及策略；Tensor数值不进入签名。"""
        from vllm_ascend.ops.dsa import _build_kv_cache

        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False) or not 1 <= tokens <= self.token_capacity:
            return None
        descriptors, counts, requests = [], None, {}
        for name, prefix in self.prefixes.items():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.decode is None:
                return None
            req = item.decode
            batch = req.seq_lens.numel()
            bound = getattr(req, "max_seqlen_q_actual", None)
            bound = req.max_seqlen_q if bound is None else bound
            if (not 1 <= batch <= self.batch_capacity or not 1 <= item.num_actual_tokens <= tokens
                    or not 1 <= bound <= MAX_QUERY_TOKENS or item.num_decodes != batch
                    or req.seq_lens.shape != (batch,) or req.query_start_loc.shape != (batch + 1,)
                    or req.seq_lens.dtype != torch.int32 or req.query_start_loc.dtype != torch.int32
                    or req.start_pos is None or req.start_pos.shape != (batch,) or req.start_pos.dtype != torch.int32
                    or req.max_seq_lens is None or not 1 <= req.max_seq_lens <= MAX_HISTORY
                    or req.ori_win_right not in (None, 0) or req.dspark_swa_indices is not None):
                return None
            if req.num_reqs_actual is not None and not 1 <= req.num_reqs_actual <= batch:
                return None
            current = (batch, item.num_actual_tokens, bound, req.max_seq_lens)
            if counts is not None and counts != current:
                return None
            counts, requests[name] = current, req
            descriptors.append((name, tuple((field, tensor_descriptor(getattr(req, field)))
                                           for field in ("seq_lens", "query_start_loc", "block_table", "start_pos"))))
        if requests["swa"].ori_win_left not in (None, 127):
            return None
        main = requests["compressed"]
        try:
            for name in ("swa", "state"):
                descriptors.append((name + ".slots", tensor_descriptor(
                    token_capacity_view(requests[name].slot_mapping, tokens, 2))))
            for field in ("cos", "sin"):
                descriptors.append((field, tensor_descriptor(
                    token_capacity_view(getattr(main, field)[self.layer_name], tokens, 64))))
        except (ValueError, KeyError, TypeError):
            return None
        descriptors.append(("compact_producer", main.seq_lens.numel(),
                            compact_capacity(tokens, main.seq_lens.numel()), main.block_size,
                            tensor_descriptor(main.full_compress_cos), tensor_descriptor(main.full_compress_sin)))
        descriptors.append(("kv_cache", tuple(tensor_descriptor(value) for value in _build_kv_cache(self.wrapper, context))))
        return tuple(descriptors), selected_policy(tokens, main.max_seq_lens)

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector, notify_kv_cache_written, wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.memcache_comm_fence import record_attention_compute_start

        metadata = {name: context.attn_metadata[prefix] for name, prefix in self.prefixes.items()}
        wait_for_kv_layer_from_connector(self.layer_name)
        compact = self._compact_metadata(context, metadata["compressed"].decode, hidden.shape[0])
        compressed, swa, state = kv_cache[:3]
        groups = {name: (metadata[name], (value,)) for name, value in
                  (("compressed", compressed), ("swa", swa), ("state", state))}
        call = NativeHCACall(self.operators, self.weights, hidden, positions, groups,
                             layer_name=self.layer_name, compact_metadata=compact, output=output)
        if torch.npu.is_current_stream_capturing():
            signature = self.graph_signature(context, hidden.shape[0])
            if signature is None:
                raise RuntimeError("HCA TND捕获输入未满足可重放合同")
            context.additional_kwargs.setdefault(_CONTRACTS, {})[self.layer_name] = TNDGraphContract(
                self, hidden.shape[0], signature
            )
        record_attention_compute_start()
        call()
        self.submissions += 1
        self.last_policy = call.policy
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
