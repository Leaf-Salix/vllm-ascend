# SPDX-License-Identifier: Apache-2.0
"""Native DSA forward implementation using the original 8990a7d8e CSA kernel.

The kernel retains its 40-tensor ABI and 128-token physical pages. Native
metadata and cache allocation remain owned by vLLM-Ascend v0.25.1rc1. Binding
inside this implementation preserves the original kernel's numerical path.
"""

import sys

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

_TP = 1
_KCSA = None
_KCFG = None
_OP = None
VLLM_PAGE = 128
VLLM_STATE_PAGE = 8
COMPRESS_RATIO = 4


def _import_kernel():
    argv = sys.argv
    sys.argv = [argv[0], "--tp", str(_TP)]
    try:
        from .pto_kernels.dspark import config as kcfg
        from .pto_kernels.dspark import decode_csa as kcsa
    finally:
        sys.argv = argv
    return kcsa, kcfg


def kernel():
    global _KCSA, _KCFG
    if _KCSA is None:
        _KCSA, _KCFG = _import_kernel()
    return _KCSA, _KCFG


class NativeLayoutError(ValueError):
    """The live vLLM allocation does not satisfy the native CSA ABI."""


def _to_nd(w: torch.Tensor) -> torch.Tensor:
    """Undo the FRACTAL_NZ layout vLLM gives quantized weights.

    ``weight_nz_mode`` defaults to 1, and an NZ-laid-out weight read as ND is
    silently wrong rather than an error, so every INT8 weight goes through this.
    Dense weights are never converted, so they are returned untouched.
    """
    if w.dtype not in (torch.int8, torch.uint8):
        return w
    import torch_npu

    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND

    return torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_ND)


def _quant_int8_per_channel(w: torch.Tensor, kcfg):
    """BF16 -> INT8 plus per-channel scale, matching the kernel's contract."""
    amax = w.float().abs().amax(dim=-1).clamp_min(kcfg.INT8_AMAX_EPS)
    sq = kcfg.INT8_SCALE_MAX / amax
    q = torch.round(w.float() * sq.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q, (1.0 / sq).float()


def _oriented(w: torch.Tensor, want: tuple) -> torch.Tensor:
    """Give the kernel its orientation, whichever one vLLM stored.

    A W8A8 linear comes out of process_weights_after_loading already transposed to
    [in, out], while an unquantized one keeps torch's [out, in]. Deciding by shape
    rather than by quantization keeps both checkpoints working.
    """
    shape = tuple(w.shape)
    if shape == want:
        return w
    if shape == want[::-1]:
        return w.t().contiguous()
    raise ValueError(f"weight is {shape}, expected {want} or its transpose")


def _dense(linear, want: tuple) -> torch.Tensor:
    """A BF16 weight in the kernel's orientation, dequantizing if vLLM quantized it."""
    w = _to_nd(linear.weight.detach())
    scale = getattr(linear, "weight_scale_fp32", None)
    if scale is None:
        scale = getattr(linear, "weight_scale", None)
    if scale is not None and w.dtype in (torch.int8, torch.uint8):
        w = w.float() * scale.detach().float().view(1, -1)
    return _oriented(w.to(torch.bfloat16), want)


def _int8(linear, want: tuple, scale_len: int, kcfg):
    """An INT8 weight plus its per-channel scale, quantizing if vLLM kept it dense.

    The scale's axis is not the same for every slot: wq_b and idx_wq_b carry one
    scale per output column, wo_b one per output row. ``scale_len`` picks which,
    so a mismatch is a shape error here rather than at kernel launch.
    """
    w = _to_nd(linear.weight.detach())
    scale = getattr(linear, "weight_scale_fp32", None)
    if scale is None:
        scale = getattr(linear, "weight_scale", None)
    if w.dtype in (torch.int8, torch.uint8) and scale is not None:
        return _oriented(w, want), scale.detach().float().reshape(-1)

    dense = _oriented(w, want)
    if scale_len == want[1]:
        q, sc = _quant_int8_per_channel(dense.t().contiguous(), kcfg)
        return q.t().contiguous(), sc
    if scale_len == want[0]:
        return _quant_int8_per_channel(dense, kcfg)
    raise ValueError(f"scale length {scale_len} matches neither axis of {want}")


def _hadamard(dim: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    """Sylvester Hadamard, normalized.

    The kernel treats this as a bare right-hand matmul operand with no scaling of
    its own, so the 1/sqrt(dim) that vLLM's rotate_activation applies separately
    has to be folded in here. Mirrors decode_csa.py::init_hadamard_idx.
    """
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / (dim**0.5)).to(dtype).to(device)


def _native_rope_tables(layer: str):
    """Alias the layer's persistent, interleaved FP32 vLLM RoPE tables.

    Do not use the per-step proxy: compressed rows there are compacted by
    boundary. The kernel can address the same static table by absolute position.
    """
    from vllm_ascend.ops.rope_dsv4 import _ROPE_STATE

    try:
        config_key, _ = _ROPE_STATE.layer_info[layer]
        cos, sin = _ROPE_STATE.full_rope_cache[config_key]
    except KeyError as error:
        raise NativeLayoutError(f"no persistent RoPE table registered for {layer}") from error
    for name, table in (("cos", cos), ("sin", sin)):
        if table.dtype != torch.float32 or not table.is_contiguous() or table.shape[-1] != 64:
            raise NativeLayoutError(f"native RoPE {name} must be contiguous FP32 with 64 columns")
    if cos.shape != sin.shape:
        raise NativeLayoutError("native RoPE cos/sin shapes differ")
    return cos.view(-1, 64), sin.view(-1, 64)


def _full_page_view(cache: torch.Tensor, rows: int, row_shape: tuple[int, ...]):
    """Expose a padded physical page as a canonical contiguous tensor view."""
    row_elems = 1
    for extent in row_shape:
        row_elems *= extent
    page_elems = rows * row_elems
    if cache.stride(0) != page_elems:
        raise NativeLayoutError(f"physical page stride is {cache.stride(0)}, expected {page_elems}")
    trailing = 1
    trailing_strides = []
    for extent in reversed(row_shape):
        trailing_strides.append(trailing)
        trailing *= extent
    shape = (cache.shape[0], rows, *row_shape)
    strides = (page_elems, row_elems, *reversed(trailing_strides))
    if (cache.storage_offset() + cache.shape[0] * page_elems) * cache.element_size() > cache.untyped_storage().nbytes():
        raise NativeLayoutError("native allocation does not cover the complete final page")
    view = torch.as_strided(
        cache,
        size=shape,
        stride=strides,
        storage_offset=cache.storage_offset(),
    )
    if not view.is_contiguous():
        raise NativeLayoutError(f"native page view {shape} is not contiguous: {view.stride()}")
    return view


def _registered():
    global _OP
    if _OP is None:
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("CSA kernel registration must precede graph capture")
        from pypto.torch import init, register

        kcsa, _ = kernel()
        init()
        _OP = register(kcsa.decode_csa_attn_tp1_test, "pypto_csa::attention_csa")
    return _OP


ARG_ORDER = (
    "x_normed",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "freqs_cos",
    "freqs_sin",
    "cmp_freqs_cos",
    "cmp_freqs_sin",
    "cmp_wkv",
    "cmp_wgate",
    "cmp_ape",
    "cmp_norm_w",
    "compress_state_pages",
    "kv_cache_pages",
    "cmp_kv_pages",
    "compress_state_block_table",
    "idx_wq_b",
    "idx_wq_b_scale",
    "weights_proj",
    "hadamard_idx",
    "inner_wkv",
    "inner_wgate",
    "inner_ape",
    "inner_norm_w",
    "inner_index_pages",
    "inner_compress_state_block_table",
    "ori_block_table",
    "cmp_block_table",
    "index_block_table",
    "position_ids",
    "token_valid",
    "kv_seq_lens",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "attn_out",
)


class PyptoDSAImpl(AscendDSAImpl):
    """Preserve native arguments, output ownership and cache lifecycle."""

    def _prepare_weights(self):
        """Build and cache the kernel's weight arguments on the implementation."""
        cached = getattr(self, "_pto_attn_weights", None)
        if cached is not None:
            return cached

        kcsa, kcfg = kernel()
        D, QL, HD = kcsa.D, kcsa.Q_LORA, kcsa.HEAD_DIM
        IH, ID = kcsa.IDX_N_HEADS, kcsa.IDX_HEAD_DIM
        wq_b, wq_b_scale = _int8(self.wq_b, (QL, kcsa.H * HD), kcsa.H * HD, kcfg)
        idx_wq_b, idx_wq_b_scale = _int8(self.inderxer_wq_b, (QL, IH * ID), IH * ID, kcfg)
        wo_b, wo_b_scale = _int8(self.wo_b, (D, kcsa.O_GROUPS * kcsa.O_LORA), D, kcfg)

        w = {
            "wq_a": _dense(self.wq_a, (D, QL)),
            "wq_b": wq_b,
            "wq_b_scale": wq_b_scale,
            "wkv": _dense(self.wkv, (D, HD)),
            "gamma_cq": self.q_norm.weight.detach().to(torch.bfloat16),
            "gamma_ckv": self.kv_norm.weight.detach().to(torch.bfloat16),
            "cmp_wkv": _dense(self.compressor_wkv, (kcsa.MAIN_OUT_DIM, D)),
            "cmp_wgate": _dense(self.compressor_wgate, (kcsa.MAIN_OUT_DIM, D)),
            "cmp_ape": self.compressor_ape.detach().float(),
            "cmp_norm_w": self.compressor_norm.weight.detach().to(torch.bfloat16),
            "idx_wq_b": idx_wq_b,
            "idx_wq_b_scale": idx_wq_b_scale,
            "weights_proj": _dense(self.weights_proj, (D, IH)),
            "hadamard_idx": _hadamard(ID, self.wo_b.weight.device),
            "inner_wkv": _dense(self.indexcom_wkv, (kcsa.INNER_OUT_DIM, D)),
            "inner_wgate": _dense(self.indexcom_wgate, (kcsa.INNER_OUT_DIM, D)),
            "inner_ape": self.indexcom_ape.detach().float(),
            "inner_norm_w": self.indexcom_norm.weight.detach().to(torch.bfloat16),
            "attn_sink": self.attn_sink.detach().float(),
            # vLLM keeps [G, O_GROUP_IN, O_LORA]; the kernel wants the transpose.
            "wo_a": self.wo_a.weight.detach().transpose(1, 2).contiguous().to(torch.bfloat16),
            "wo_b": wo_b,
            "wo_b_scale": wo_b_scale,
        }
        self._pto_attn_weights = w
        return w

    def _kernel_args(self, hidden_states, kv_cache, metadata_list, seq: int, layer: str, output):
        """Bind one decode step to the kernel's native-layout arguments.

        ``metadata_list`` is what ``filter_metadata`` returns for a ratio-4 layer:
        five per-cache metadata objects sorted by key -- attn, compressor state,
        indexer-compressor state, indexer k, sliding window.
        """
        if not isinstance(layer, str):
            # RopeDataProxy takes a non-string key as a slice and hands back another
            # proxy, so a wrong name surfaces two frames later as a missing reshape.
            raise NativeLayoutError(f"layer must be the layer's name, got {type(layer).__name__}")
        if len(metadata_list) != 5 or len(kv_cache) != 6:
            raise NativeLayoutError(
                f"ratio-4 CSA requires 5 metadata groups and 6 cache views, "
                f"got {len(metadata_list)} and {len(kv_cache)}"
            )
        if any(m.decode is None for m in metadata_list):
            raise NativeLayoutError("native CSA only accepts decode metadata")
        kcsa, _ = kernel()
        if not 1 <= seq <= kcsa.S:
            raise NativeLayoutError(f"native CSA requires 1..{kcsa.S} tokens per request, got seq={seq}")
        cmp_md, cst_md, inner_state_md, idx_md, swa_md = (m.decode for m in metadata_list)
        cmp_kv_c, swa_kv_c, state_c, inner_state_cache, idx_k_c, idx_s_c = kv_cache

        host_pos = metadata_list[0].decode.input_positions
        if host_pos.dtype != torch.int64 or host_pos.ndim != 1 or not host_pos.is_contiguous():
            raise NativeLayoutError("input_positions must be contiguous INT64 token rows")
        if host_pos.shape[0] % seq:
            raise NativeLayoutError(f"position rows {host_pos.shape[0]} are not divisible by seq={seq}")
        n_real = host_pos.shape[0] // seq  # graph descriptor request rows
        if not 1 <= n_real <= kcsa.B:
            raise NativeLayoutError(f"{n_real} requests exceed the kernel's B={kcsa.B}")
        if hidden_states.shape[0] < host_pos.shape[0]:
            raise NativeLayoutError(
                f"hidden_states has {hidden_states.shape[0]} rows, expected at least {host_pos.shape[0]}"
            )

        b, t = n_real, host_pos.shape[0]
        pos = host_pos
        host_positions = pos.view(b, seq)
        raw_logical_page = torch.div(
            host_positions,
            VLLM_PAGE,
            rounding_mode="floor",
        )
        raw_page_in_range = (raw_logical_page >= 0) & (raw_logical_page < swa_md.block_table.shape[1])
        raw_logical_page = raw_logical_page.clamp(
            min=0,
            max=swa_md.block_table.shape[1] - 1,
        )
        raw_pages = swa_md.block_table[:b].gather(
            1,
            raw_logical_page,
        )
        rope_cos, rope_sin = _native_rope_tables(layer)
        if cmp_md.seq_lens.dtype != torch.int32 or not cmp_md.seq_lens.is_contiguous():
            raise NativeLayoutError("seq_lens must be contiguous INT32 request rows")
        seq_lens = cmp_md.seq_lens[:b]
        token_valid = (
            raw_page_in_range
            & (raw_pages > 0)
            & (host_positions < rope_cos.shape[0])
            & (host_positions < seq_lens.view(b, 1))
        ).reshape(t)

        def native_table(name: str, table: torch.Tensor) -> torch.Tensor:
            if table.dtype != torch.int32:
                raise NativeLayoutError(f"{name} must be INT32, got {table.dtype}")
            if table.shape[0] < b:
                raise NativeLayoutError(f"{name} has {table.shape[0]} requests, expected at least {b}")
            if not table.is_contiguous():
                raise NativeLayoutError(f"{name} must be contiguous; no per-step copy is made")
            return table[:b]

        expected_dtypes = {
            "main state": (state_c, torch.float32),
            "inner state": (inner_state_cache, torch.float32),
            "raw KV": (swa_kv_c, torch.bfloat16),
            "compressed KV": (cmp_kv_c, torch.bfloat16),
            "index key page": (idx_k_c, torch.int8),
            "index scale view": (idx_s_c, torch.float16),
        }
        for name, (tensor, dtype) in expected_dtypes.items():
            if tensor.dtype != dtype:
                raise NativeLayoutError(f"{name} must be {dtype}, got {tensor.dtype}")

        a = dict(self._pto_attn_weights)
        a["x_normed"] = hidden_states[:t]
        a["attn_out"] = output[:t]
        for name in ("x_normed", "attn_out"):
            tensor = a[name]
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() or tensor.shape != (t, kcsa.D):
                raise NativeLayoutError(f"{name} must be contiguous BF16 [{t}, {kcsa.D}]")

        a["freqs_cos"], a["freqs_sin"] = rope_cos, rope_sin
        a["cmp_freqs_cos"], a["cmp_freqs_sin"] = rope_cos, rope_sin

        # Main state and compressed KV are different views of one physical page
        # pool; raw sliding KV uses a separate allocation.
        if state_c.stride(0) != 32768:
            raise NativeLayoutError(f"main state page stride is {state_c.stride(0)}, expected 32768 FP32")
        if swa_kv_c.stride(0) != 65536 or cmp_kv_c.stride(0) != 65536:
            raise NativeLayoutError("raw/compressed KV page stride does not match 131072 bytes")
        a["compress_state_pages"] = _full_page_view(
            state_c,
            kcsa.VLLM_COMPRESS_STATE_PAGE_ROWS,
            (kcsa.MAIN_STATE_DIM,),
        )
        a["kv_cache_pages"] = _full_page_view(
            swa_kv_c,
            kcsa.VLLM_KV_PAGE_ROWS,
            (1, kcsa.HEAD_DIM),
        )
        a["cmp_kv_pages"] = _full_page_view(
            cmp_kv_c,
            kcsa.VLLM_KV_PAGE_ROWS,
            (1, kcsa.HEAD_DIM),
        )
        if (
            a["compress_state_pages"].data_ptr() != a["cmp_kv_pages"].data_ptr()
            or a["compress_state_pages"].numel() * a["compress_state_pages"].element_size()
            != a["cmp_kv_pages"].numel() * a["cmp_kv_pages"].element_size()
        ):
            raise NativeLayoutError("main state and compressed KV must cover the same page pool")
        a["compress_state_block_table"] = native_table(
            "main state block table",
            cst_md.block_table,
        )
        a["inner_compress_state_block_table"] = native_table(
            "inner state block table",
            inner_state_md.block_table,
        )
        a["ori_block_table"] = native_table(
            "raw KV block table",
            swa_md.block_table,
        )
        a["cmp_block_table"] = native_table(
            "compressed KV block table",
            cmp_md.block_table,
        )
        shared_storage = idx_k_c.untyped_storage().data_ptr()
        if inner_state_cache.untyped_storage().data_ptr() != shared_storage:
            raise NativeLayoutError("inner state and index key do not share one vLLM allocation")
        if inner_state_cache.data_ptr() != idx_k_c.data_ptr():
            raise NativeLayoutError("inner state and index key do not start at the same physical page")
        if inner_state_cache.stride(0) != 4160:
            raise NativeLayoutError(f"inner state page stride is {inner_state_cache.stride(0)}, expected 4160 FP32")
        if idx_s_c.untyped_storage().data_ptr() != shared_storage:
            raise NativeLayoutError("index key and scale do not share one vLLM page")
        if idx_k_c.stride(0) != 16640 or idx_s_c.stride(0) != 8320:
            raise NativeLayoutError("index key/scale physical strides do not match the 16640-byte page")
        if idx_s_c.data_ptr() - idx_k_c.data_ptr() != 16384:
            raise NativeLayoutError("index scale does not start at byte 16384 of the packed page")
        a["inner_index_pages"] = _full_page_view(
            idx_k_c,
            kcsa.VLLM_INDEX_PAGE_ROWS,
            (kcsa.IDX_HEAD_DIM,),
        )
        a["index_block_table"] = native_table(
            "index block table",
            idx_md.block_table,
        )
        a["position_ids"] = pos
        a["token_valid"] = token_valid.to(torch.int32)
        a["kv_seq_lens"] = seq_lens

        return [a[name] for name in ARG_ORDER]

    def _decode_query_length(self, metadata_list, kv_cache) -> int | None:
        """Return the uniform native query length before any cache is mutated.

        DSpark verification can offer fewer than six tokens at scheduling boundaries.
        Use the native CPU query offsets, not a process-wide sequence-length override.
        Ragged batches stay on the native path; their packed rows cannot be reshaped
        into the kernel's rectangular [B, S] input without a separate packing layer.
        """
        if not isinstance(metadata_list, list) or kv_cache is None:
            return None
        config = self.vllm_config
        parallel = config.parallel_config
        spec = config.speculative_config
        if (
            self.compress_ratio != COMPRESS_RATIO
            or parallel.tensor_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or (
                spec is not None
                and (getattr(spec, "method", None) != "dspark" or getattr(spec, "num_speculative_tokens", None) != 5)
            )
            or config.kv_transfer_config is not None
            or self.skip_topk
            or self.use_index_cache
            or len(metadata_list) != 5
            or len(kv_cache) != 6
        ):
            return None
        expected_shapes = ((128, 1, 512), (128, 1, 512), (8, 1, 2048), (8, 1, 512), (128, 1, 128), (128, 1, 1))
        if any(tuple(cache.shape[1:]) != shape for cache, shape in zip(kv_cache, expected_shapes)):
            return None
        swa = metadata_list[-1].decode
        if (
            getattr(swa, "dspark_swa_indices", None) is not None
            or getattr(swa, "ori_win_left", None) not in (None, 127)
            or getattr(swa, "ori_win_right", None) not in (None, 0)
        ):
            return None
        seq = None
        batch = metadata_list[0].num_decodes
        for m in metadata_list:
            if (
                m.decode is None
                or m.num_prefills != 0
                or m.num_decodes <= 0
                or m.num_decodes != batch
                or m.num_actual_tokens != m.num_decode_tokens
            ):
                return None
            length, remainder = divmod(m.num_decode_tokens, batch)
            if remainder or not 1 <= length <= (6 if spec is not None else 1):
                return None
            if seq is not None and seq != length:
                return None
            d = m.decode
            if (
                d.input_positions.ndim != 1
                or d.input_positions.numel() != m.num_decode_tokens
                or d.seq_lens.ndim != 1
                or d.seq_lens.numel() != batch
                or d.block_table.ndim != 2
                or d.block_table.shape[0] < batch
                or d.block_table.shape[1] == 0
                or d.num_reqs_actual not in (None, batch)
            ):
                return None
            if spec is not None:
                offsets = m.decode.query_start_loc_cpu
                if offsets is None or offsets.device.type != "cpu" or offsets.ndim != 1 or offsets.numel() != batch + 1:
                    return None
                # Already host metadata: this never synchronizes an NPU tensor.
                if offsets.tolist() != list(range(0, (batch + 1) * length, length)):
                    return None
            seq = length
        return seq

    def _supports_configuration(self):
        config = self.vllm_config
        parallel = config.parallel_config
        spec = config.speculative_config
        return not (
            self.compress_ratio != COMPRESS_RATIO
            or get_ascend_device_type() != AscendDeviceType.A3
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or olora_tp_enable()
            or oproj_tp_enable()
            or config.kv_transfer_config is not None
            or config.lora_config is not None
            or self.skip_topk
            or self.use_index_cache
            or (spec is not None and (spec.method != "dspark" or spec.num_speculative_tokens != 5))
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if hasattr(self, "_pto_attn_weights"):
            del self._pto_attn_weights
        if self._supports_configuration():
            self._prepare_weights()

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: DSAMetadataList,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        seq = None
        if (
            not need_gather_q_kv
            and self._supports_configuration()
            and not getattr(get_forward_context(), "is_draft_model", False)
        ):
            seq = self._decode_query_length(attn_metadata, kv_cache)
        if seq is None:
            return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output)
        # Post-load owns weight conversion. Never perform first-time conversions
        # inside capture, and never mutate a native cache before layout checks.
        if not hasattr(self, "_pto_attn_weights"):
            raise RuntimeError("CSA weights must be prepared by the native post-load hook")
        try:
            args = self._kernel_args(hidden_states, kv_cache, attn_metadata, seq, layer_name, output)
        except NativeLayoutError:
            return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output)
        wait_for_kv_layer_from_connector(layer_name)
        record_attention_compute_start()
        _registered()(*args)
        # The original kernel fuses prolog cache writes and attention. Notify
        # after the fused launch; failures must propagate without native retry.
        notify_kv_cache_written(layer_name)
        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
        self._pto_calls = getattr(self, "_pto_calls", 0) + 1
        if self._pto_calls <= 3:
            print(f"[pto-native-csa] layer={layer_name} calls={self._pto_calls} seq={seq}", flush=True)
        return output
