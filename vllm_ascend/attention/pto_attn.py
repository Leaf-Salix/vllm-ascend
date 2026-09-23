# SPDX-License-Identifier: Apache-2.0
"""PyPTO implementation of the native AscendDSAImpl forward contract.

Native metadata producers own slots and compact RoPE rows. This implementation
passes their tensors to the kernel; it does not regenerate token masks or cache
addresses. Kernel math and physical-page descriptors are derived from nalinaly
8a4c4e6f, with the surrounding contract defined by v0.25.1rc1 AscendDSAImpl.
"""

from functools import lru_cache

import torch
from vllm.forward_context import get_forward_context

from vllm_ascend.attention.dsa_v1 import AscendDSAImpl, DSAMetadataList
from vllm_ascend.attention.utils import (
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.memcache_comm_fence import record_attention_compute_start
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, olora_tp_enable, oproj_tp_enable


class PyptoDSAImpl(AscendDSAImpl):
    """Use native forward arguments and output ownership for target CSA S6."""

    def _prepare_weights(self, hadamard: torch.Tensor | None) -> dict[str, torch.Tensor] | None:
        """Prepare the TP1 ABI from already-loaded Native parameters exactly once."""
        import torch_npu

        if self.compress_ratio != 4 or self.n_local_heads != 64 or self.n_local_groups != 8:
            raise ValueError("CSA specialization requires C4 and TP1 with 64 heads / 8 output groups")
        # Some checkpoints quantize these projections too. This specialization
        # must leave them native, rather than dequantizing their weights.
        if self.wq_a.weight.dtype != torch.bfloat16 or self.wkv.weight.dtype != torch.bfloat16:
            return None

        def weight(module, shape, dtype, transpose=False):
            value = module.weight.detach()
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"Unexpected loaded weight: {value.shape}/{value.dtype}; expected {shape}/{dtype}")
            if torch_npu.get_npu_format(value) not in (0, 2):
                raise ValueError("CSA weights must already be Native ND after loading")
            if transpose:
                value = value.transpose(-1, -2)
            return value.contiguous()

        def scale(module, width):
            result = module.weight_scale.detach().reshape(-1)
            if result.numel() != width:
                raise ValueError("Unexpected quantized channel-scale count")
            offset = getattr(module, "weight_offset", None)
            if offset is not None and bool(torch.count_nonzero(offset).cpu()):
                raise ValueError("The reference CSA chain requires symmetric INT8 weights")
            return result.float().contiguous()

        bf16, int8 = torch.bfloat16, torch.int8
        main, indexer = self.compressor, self.indexer
        inner = indexer.compressor
        return {
            "wq_a": weight(self.wq_a, (1024, 4096), bf16, True),
            "wq_b": weight(self.wq_b, (1024, 32768), int8),
            "wq_b_scale": scale(self.wq_b, 32768),
            "wkv": weight(self.wkv, (512, 4096), bf16, True),
            "gamma_cq": weight(self.q_norm, (1024,), bf16),
            "gamma_ckv": weight(self.kv_norm, (512,), bf16),
            "cmp_wkv": weight(main.wkv, (1024, 4096), bf16),
            "cmp_wgate": weight(main.wgate, (1024, 4096), bf16),
            "cmp_ape": main.ape.detach().float().contiguous(),
            # Match Native A3 storage; the RMS task widens loaded BF16 tiles.
            "cmp_norm_w": weight(main.norm, (512,), bf16),
            "idx_wq_b": weight(indexer.wq_b, (1024, 8192), int8),
            "idx_wq_b_scale": scale(indexer.wq_b, 8192),
            "weights_proj": weight(indexer.weights_proj, (64, 4096), bf16, True),
            **({"hadamard_idx": hadamard.detach().T.to(bf16).contiguous()} if hadamard is not None else {}),
            "inner_wkv": weight(inner.wkv, (256, 4096), bf16),
            "inner_wgate": weight(inner.wgate, (256, 4096), bf16),
            "inner_ape": inner.ape.detach().float().contiguous(),
            "inner_norm_w": weight(inner.norm, (128,), bf16),
            "attn_sink": self.attn_sink.detach().contiguous(),
            "wo_a": weight(self.wo_a, (8, 4096, 1024), bf16, True),
            "wo_b": weight(self.wo_b, (8192, 4096), int8, True),
            "wo_b_scale": scale(self.wo_b, 4096),
        }

    @staticmethod
    def _physical_pages(view):
        # Only expose allocated storage: a descriptor, never a repacked cache.
        if view.ndim != 4 or view.shape[2] != 1 or view.stride(-1) != 1:
            raise ValueError("Unsupported native physical page view")
        pages, rows, _, width = view.shape
        stride = view.stride(0)
        if view.stride(1) != width or stride < rows * width:
            raise ValueError("Native page rows must be contiguous")
        if (view.storage_offset() + pages * stride) * view.element_size() > view.untyped_storage().nbytes():
            raise ValueError("Native storage does not cover the complete final page")
        return view.as_strided((pages, stride), (stride, 1), view.storage_offset())

    @staticmethod
    def _table_view(table):
        if table.dtype != torch.int32 or table.ndim != 2 or table.stride(1) != 1:
            raise ValueError("Native block table must have contiguous INT32 columns")
        if table.is_contiguous():
            return table
        rows, columns = table.shape
        stride = table.stride(0)
        if stride < columns or (table.storage_offset() + rows * stride) * 4 > table.untyped_storage().nbytes():
            raise ValueError("Native table storage does not cover padded rows")
        return table.as_strided((rows, stride), (stride, 1), table.storage_offset())

    def _supports_configuration(self):
        config = self.vllm_config
        parallel = config.parallel_config
        spec = config.speculative_config
        return not (
            self.compress_ratio != 4
            or self.skip_topk
            or self.use_index_cache
            or get_ascend_device_type() != AscendDeviceType.A3
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or olora_tp_enable()
            or oproj_tp_enable()
            or config.kv_transfer_config is not None
            or config.lora_config is not None
            or spec is None
            or spec.method != "dspark"
            or spec.num_speculative_tokens != 5
            or not config.model_config.enforce_eager
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        for name in ("_pto_weights", "_pto_operator", "_pto_hadamard"):
            if hasattr(self, name):
                delattr(self, name)
        if self._supports_configuration():
            self._pto_weights = self._prepare_weights(None)

    def _eligible(self, hidden, cache, metadata, gather):
        if gather or not self._supports_configuration() or getattr(self, "_pto_weights", None) is None:
            return False
        if not isinstance(metadata, list) or len(metadata) != 5 or cache is None or len(cache) != 6:
            return False
        if getattr(get_forward_context(), "is_draft_model", False):
            return False
        if hidden.ndim != 2 or hidden.shape[1] != 4096 or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            return False
        tokens = hidden.shape[0]
        if tokens % 6 or not 1 <= tokens // 6 <= 64:
            return False
        batch = tokens // 6
        for m in metadata:
            if m.decode is None or m.num_prefills or m.num_actual_tokens != tokens or m.num_decodes != batch:
                return False
            d = m.decode
            offsets = d.query_start_loc_cpu
            if (
                m.num_decode_tokens != tokens
                or offsets is None
                or offsets.device.type != "cpu"
                or offsets.tolist() != list(range(0, tokens + 1, 6))
                or d.seq_lens.numel() != batch
                or d.num_reqs_actual not in (None, batch)
                or d.ori_win_right not in (None, 0)
                or d.dspark_swa_indices is not None
            ):
                return False
        if metadata[-1].decode.ori_win_left not in (None, 127):
            return False
        # This kernel consumes the release's 32-token KV / 2-token state pages.
        # Unsupported native page configurations remain entirely native.
        return all(
            tuple(cache[i].shape[1:]) == shape
            for i, shape in (
                (0, (32, 1, 512)),
                (1, (32, 1, 512)),
                (2, (2, 1, 2048)),
                (3, (2, 1, 512)),
                (4, (32, 1, 128)),
                (5, (32, 1, 1)),
            )
        ) and all(
            cache[i].dtype == dtype
            for i, dtype in enumerate(
                (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32, torch.int8, torch.float16)
            )
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def _register_kernel():
        from pypto.torch import init, register

        from .pto_kernels.dspark.decode_csa import decode_csa_tp1_attention_test

        init()
        return register(decode_csa_tp1_attention_test, "pypto_csa::native_attention_v2")

    def _initialize_kernel(self, hadamard, device):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("PyPTO CSA initialization must precede graph capture")
        if not hasattr(self, "_pto_weights"):
            raise RuntimeError("Native post-load hook must prepare CSA weights before execution")
        self._pto_weights["hadamard_idx"] = hadamard.detach().T.to(torch.bfloat16).contiguous()
        self._pto_hadamard = hadamard
        self._pto_operator = self._register_kernel()
        capacity = min(self.vllm_config.scheduler_config.max_num_seqs, 64) * 6
        self._pto_scores = torch.empty((capacity, 512), dtype=torch.float32, device=device)
        self._pto_topk = torch.empty((capacity, 512), dtype=torch.int32, device=device)
        self._pto_calls = 0

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: DSAMetadataList,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Preserve the native input/output and cache lifecycle contract.

        The ordered metadata groups and cache tuple are exactly those unpacked
        by AscendDSAImpl._forward_decode. All layout checks precede writes;
        exceptions after kernel launch propagate without native re-execution.
        """
        assert output is not None, "Output tensor must be provided."
        if not self._eligible(hidden_states, kv_cache, attn_metadata, need_gather_q_kv):
            return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output)
        main, state, inner, index, swa = (m.decode for m in attn_metadata)
        compressed, raw, main_state, inner_state, key, scale = kv_cache
        if not output.is_contiguous() or output.shape != hidden_states.shape or output.dtype != hidden_states.dtype:
            raise ValueError("Native output must match the contiguous BF16 hidden states")
        if not raw.is_contiguous() or not compressed.is_contiguous():
            raise ValueError("Native KV pages must be contiguous")
        if (
            key.untyped_storage().data_ptr() != scale.untyped_storage().data_ptr()
            or scale.storage_offset() * 2 != key.storage_offset() + 4096
            or key.stride(0) < 4160
            or key.stride(0) % 64
            or scale.stride(0) * 2 != key.stride(0)
        ):
            raise ValueError("Native index key/scale page layout is inconsistent")
        main_pages = self._physical_pages(main_state)
        inner_pages = self._physical_pages(inner_state)
        index_pages = self._physical_pages(key)
        tables = [self._table_view(m.block_table) for m in (main, state, inner, index, swa)]
        hadamard = attn_metadata[3].hadamard
        if hadamard is None:
            raise ValueError("Native indexer metadata did not supply Hadamard")
        if not hasattr(self, "_pto_operator"):
            self._initialize_kernel(hadamard, hidden_states.device)
        elif self._pto_hadamard is not hadamard:
            raise RuntimeError("Native Hadamard changed after CSA initialization; reload weights before reuse")
        w = self._pto_weights
        tokens = hidden_states.shape[0]
        wait_for_kv_layer_from_connector(layer_name)
        # These are the same producers called by the native compressor/indexer.
        cmp_cos, cmp_sin, cmp_slots = self._compute_compressor_metadata(main)
        idx_cos, idx_sin, idx_slots = self._compute_compressor_metadata(index)
        record_attention_compute_start()
        self._pto_operator(
            hidden_states,
            w["wq_a"],
            w["wq_b"],
            w["wq_b_scale"],
            w["wkv"],
            w["gamma_cq"],
            w["gamma_ckv"],
            main.cos[layer_name].view(tokens, 64),
            main.sin[layer_name].view(tokens, 64),
            cmp_cos.view(-1, 64),
            cmp_sin.view(-1, 64),
            idx_cos.view(-1, 64),
            idx_sin.view(-1, 64),
            w["cmp_wkv"],
            w["cmp_wgate"],
            w["cmp_ape"],
            w["cmp_norm_w"],
            main_pages,
            tables[1],
            w["idx_wq_b"],
            w["idx_wq_b_scale"],
            w["weights_proj"],
            w["hadamard_idx"],
            w["inner_wkv"],
            w["inner_wgate"],
            w["inner_ape"],
            w["inner_norm_w"],
            inner_pages,
            tables[2],
            raw,
            compressed,
            tables[0],
            index_pages,
            tables[3],
            swa.slot_mapping,
            tables[4],
            cmp_slots,
            idx_slots,
            state.slot_mapping,
            inner.slot_mapping,
            main.input_positions,
            index.seq_lens,
            main.query_start_loc,
            main.seq_lens,
            index.query_start_loc,
            w["attn_sink"],
            w["wo_a"],
            w["wo_b"],
            w["wo_b_scale"],
            self._pto_scores[:tokens],
            self._pto_topk[:tokens],
            output,
        )
        # This fused call contains both cache writes and attention. Notify after
        # submission; there is no separate prolog/attention overlap boundary.
        notify_kv_cache_written(layer_name)
        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
        self._pto_calls += 1
        if self._pto_calls <= 3 or self._pto_calls % 25 == 0:
            print(f"[pto-native-csa] layer={layer_name} calls={self._pto_calls} tokens={tokens} seq=6", flush=True)
        return output
