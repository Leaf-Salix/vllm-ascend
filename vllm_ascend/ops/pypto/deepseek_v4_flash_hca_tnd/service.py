# SPDX-License-Identifier: Apache-2.0
"""HCA TND 的显式服务适配，复用已有 dsv4_hca_forward 调用边界。

初始化后为一个 C128 层持有权重描述符；每步只绑定Native已有设备Tensor。
T是输入容量、B是请求容量，两者独立。Host只检查已有shape/dtype/调度计数，
真实query边界、历史长度、页表、slot由PTO核在设备上读取，不从设备回读值。

沿用Native的KV等待、写完成通知与保存钩子，compact metadata仍由Native生成，
通过原有forward-context缓存按步复用。本模块不会自动替换其他模型或旧HCA入口；
调用bind_layer才显式安装，未支持的prefill/布局继续由已有custom op回退Native。
"""

import torch

from ..deepseek_v4_flash_csa.service import AttentionServiceBase
from .native_adapter import NativeHCACall, prepare_weights, token_capacity_view

MAX_REQUESTS = 64
MAX_TOKENS = 384
MAX_QUERY_TOKENS = 6
MAX_HISTORY = 1048576


class HCATNDServiceRuntime(AttentionServiceBase):
    def __init__(self, attention, operators, max_num_seqs, layer, max_num_batched_tokens=MAX_TOKENS):
        self.wrapper = attention.dsa_attn
        self.layer_name = self.wrapper.dsa_attn.layer_name
        self.operators = operators
        self.weights = prepare_weights(attention, layer)
        self.prefixes = {
            "swa": self.wrapper.swa_cache_layer.prefix,
            "compressed": self.layer_name,
            "state": attention.compressor.state_cache.prefix,
        }
        self.batch_capacity = min(max_num_seqs, MAX_REQUESTS)
        self.token_capacity = min(max_num_batched_tokens, MAX_TOKENS)

    def eligible(self, context, hidden, positions):
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return False
        if (hidden.ndim != 3 or hidden.shape[1:] != (4, 4096)
                or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous()):
            return False
        tokens = hidden.shape[0]
        if not 1 <= tokens <= self.token_capacity:
            return False
        if (positions.dtype != torch.int64 or positions.shape != (tokens,)
                or positions.device != hidden.device or not positions.is_contiguous()):
            return False
        counts = None
        requests_by_group = {}
        for name, prefix in self.prefixes.items():
            item = metadata.get(prefix)
            if item is None or item.num_prefills or item.decode is None:
                return False
            req = item.decode
            batch = req.seq_lens.numel()
            bound = getattr(req, "max_seqlen_q_actual", None)
            bound = req.max_seqlen_q if bound is None else bound
            if (not 1 <= batch <= self.batch_capacity or not 1 <= item.num_actual_tokens <= tokens
                    or not 1 <= bound <= MAX_QUERY_TOKENS or item.num_decodes != batch):
                return False
            if (req.seq_lens.shape != (batch,) or req.query_start_loc.shape != (batch + 1,)
                    or req.seq_lens.dtype != torch.int32 or req.query_start_loc.dtype != torch.int32
                    or req.seq_lens.device != hidden.device or req.query_start_loc.device != hidden.device
                    or not req.seq_lens.is_contiguous() or not req.query_start_loc.is_contiguous()):
                return False
            if req.max_seq_lens is None or not 1 <= req.max_seq_lens <= MAX_HISTORY:
                return False
            current = (batch, item.num_actual_tokens, bound, req.max_seq_lens)
            if counts is not None and current != counts:
                return False
            counts = current
            if req.num_reqs_actual is not None and not 1 <= req.num_reqs_actual <= batch:
                return False
            if req.ori_win_right not in (None, 0) or req.dspark_swa_indices is not None:
                return False
            requests_by_group[name] = req
        if requests_by_group["swa"].ori_win_left not in (None, 127):
            return False
        # 与实际绑定使用同一容量检查；只建同地址视图，不补齐或展开设备数据。
        try:
            main = requests_by_group["compressed"]
            token_capacity_view(main.cos[self.layer_name], tokens, 64)
            token_capacity_view(main.sin[self.layer_name], tokens, 64)
            token_capacity_view(requests_by_group["state"].slot_mapping, tokens, 2)
            token_capacity_view(requests_by_group["swa"].slot_mapping, tokens, 2)
        except (ValueError, KeyError, TypeError):
            return False
        return True

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector, notify_kv_cache_written, wait_for_kv_layer_from_connector,
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
            self.operators, self.weights, hidden, positions, groups, layer_name=self.layer_name,
            compact_metadata=compact, output=output,
        )
        record_attention_compute_start()
        call()
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))


def bind_layer(layer, operators, *, max_num_seqs=MAX_REQUESTS, max_num_batched_tokens=MAX_TOKENS):
    """在运行时已初始化、权重已加载后显式接管C128层；复用原服务custom op。"""
    from vllm_ascend.models.pypto_deepseek_v4_hca import install_hca_forward

    runtime = HCATNDServiceRuntime(layer.self_attn, operators, max_num_seqs, layer, max_num_batched_tokens)
    install_hca_forward(layer)
    layer.self_attn.dsa_attn._pto_hca_runtime = runtime
    return runtime
