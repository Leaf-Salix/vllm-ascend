# SPDX-License-Identifier: Apache-2.0
"""Per-layer PTO resources and binding of the current Native service buffers."""

import torch

from vllm_ascend.ops.pypto.variant import csa_runtime

from .host_metadata import CSAHostMetadata
from .native_adapter import HBGNativeCSACall, NativeCSACall, prepare_weights
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


class AttentionServiceBase:
    """CSA性能版与HCA共用的入口判定和metadata缓存；不注册算术实现。"""

    variable_query_lengths = False

    def eligible(self, context, hidden, positions):
        metadata = context.attn_metadata
        if metadata is None or getattr(context, "is_draft_model", False):
            return False
        # 整层入口：进出都是层间的 mHC 残差流 [T, HC_MULT, D]。
        if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (4, 4096) or hidden.dtype != torch.bfloat16:
            return False
        tokens = hidden.shape[0]
        if not self.variable_query_lengths and tokens % QUERY_TOKENS:
            return False
        # TND：tokens 是**档位容量**，不再等于 batch * QUERY_TOKENS。请求数只能从
        # metadata 取，不能由 token 数反推。容量上界仍是满档。
        if not 1 <= tokens <= self.batch_capacity * QUERY_TOKENS:
            return False
        if not hidden.is_contiguous() or positions.dtype != torch.int64 or positions.shape != (tokens,):
            return False
        batch = None
        for prefix in self.prefixes.values():
            item = metadata.get(prefix)
            # 补位档位下这两个计数不再相等：num_actual_tokens 是**实际** token 数，
            # num_decodes 是**补齐后**的请求数（实测 18 与 4，整档 24）。补位请求
            # 由 kernel 内的 seq_lens 判据屏蔽。
            if item is None or item.num_prefills or item.num_actual_tokens > tokens:
                return False
            if not self.variable_query_lengths and (
                item.num_actual_tokens % QUERY_TOKENS or item.num_decodes != tokens // QUERY_TOKENS
            ):
                return False
            if batch is None:
                batch = item.num_decodes
                # 每请求至少一行，档位必须放得下。
                if not 1 <= batch <= self.batch_capacity:
                    return False
            elif item.num_decodes != batch:
                # 各 cache group 必须看到同一批请求，否则 query_start_loc 无法共用。
                return False
            req = item.decode
            if req is None or req.seq_lens.numel() != batch or req.query_start_loc.numel() != batch + 1:
                return False
            # 变长下每请求的 token 数在 [1, QUERY_TOKENS]；num_reqs_actual 是真实请求数，
            # 不能再由 num_actual_tokens // QUERY_TOKENS 推（那是等长才成立的式子）。
            if not self.variable_query_lengths and req.max_seqlen_q != QUERY_TOKENS:
                return False
            if req.max_seqlen_q > QUERY_TOKENS and self.variable_query_lengths:
                bounds_cpu = getattr(req, "query_start_loc_cpu", None)
                actual = req.num_reqs_actual
                if bounds_cpu is None or bounds_cpu.device.type != "cpu" or actual is None:
                    return False
                real_bounds = bounds_cpu[: actual + 1].tolist()
                if len(real_bounds) != actual + 1 or any(
                    not 1 <= end - begin <= QUERY_TOKENS for begin, end in zip(real_bounds, real_bounds[1:])
                ):
                    return False
            elif not 1 <= req.max_seqlen_q <= QUERY_TOKENS:
                return False
            if req.num_reqs_actual is not None and not 0 <= req.num_reqs_actual <= batch:
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


class CSAServiceRuntime(AttentionServiceBase):
    variable_query_lengths = True

    def __init__(self, attention, operators, max_num_seqs, layer=None):
        if layer is None:
            raise ValueError("TND precision requires the Native decoder layer")
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() != AscendDeviceType.A3:
            raise ValueError("TND precision currently requires A3 Native numerical calibration")
        self.layer = layer
        self.wrapper = attention.dsa_attn
        self.layer_name = self.wrapper.dsa_attn.layer_name
        self.operators = operators
        self.is_hbg = csa_runtime() == "host_build_graph"
        if self.is_hbg:
            raise ValueError("TND CSA currently requires PTO_CSA_RUNTIME=tensormap_and_ringbuffer")
        # 精度路径复用本层 Native HC/norm 的计算边界，CSA 主体仍走 PTO TND。
        self.weights = prepare_weights(attention, None, layer)
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
        self.native_heads = torch.empty((tokens, 32768), dtype=torch.bfloat16, device=device)
        self.native_attention_out = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=device)
        self._hadamard = None

    def __call__(self, context, hidden, positions, output, kv_cache):
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector,
            notify_kv_cache_written,
            wait_for_kv_layer_from_connector,
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
        call_weights = dict(self.weights, native_normed=normed, native_heads=self.native_heads[:tokens])
        call_type = HBGNativeCSACall if self.is_hbg else NativeCSACall
        host_kwargs = {}
        if self.is_hbg:
            host = CSAHostMetadata.from_native(metadata["indexer"].decode)
            host_kwargs["host_metadata"] = host
            # An enclosing NPUGraph freezes the Host scalar. Record precisely
            # which Native CPU value its replay guard must check; per-request
            # lengths remain live device inputs inside the captured tasks.
            context.additional_kwargs.setdefault("pto_csa_hbg_graph_metadata", {})[self.prefixes["indexer"]] = host
        call = call_type(
            self.operators,
            call_weights,
            hidden,
            positions,
            groups,
            layer_name=self.layer_name,
            compact_metadata=compact,
            buffers={"idx_topk_scores": self.scores[:tokens], "idx_topk": self.topk[:tokens], "x_out": output},
            **host_kwargs,
        )
        # A single fused call publishes all KV writes. Notify the connector on
        # the same stream after that call; no global synchronization is needed.
        record_attention_compute_start()
        call()
        # Reuse the exact Native projection boundary after PTO TND attention.
        # These persistent buffers retain their addresses across ACLGraph replay.
        self.wrapper.dsa_attn.impl._forward_o_proj(
            self.native_heads[:tokens].view(tokens, 64, 512), self.native_attention_out[:tokens]
        )
        output.copy_(self.layer.hc_post(self.native_attention_out[:tokens], hidden, post, comb))
        notify_kv_cache_written(self.layer_name)
        maybe_save_kv_layer_to_connector(self.layer_name, list(kv_cache))
