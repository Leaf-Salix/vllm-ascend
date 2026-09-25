"""Feed vLLM's decode tensors to the PyPTO attention-only CSA kernel.

The kernel is ``pto_kernels.dspark.decode_csa.decode_csa_attn_tp1_test``: it takes
``x_normed [T, D] BF16`` and fills ``attn_out [T, D] BF16``, leaving ``npu_hc_pre``,
the input RMSNorm and ``npu_hc_post`` to the native path. Everything here is the
binding between vLLM's decode state and the kernel's native-layout arguments.

Two properties the per-step path must keep, because it has to survive ACLGraph
capture: no device-to-host read (no ``.item()``, no boolean-mask indexing, no
``torch.unique``), and no allocation whose shape depends on a device value.

"""

from __future__ import annotations

import sys

import torch

# --- kernel import -----------------------------------------------------------
# decode_csa fixes its TP specialization at import time from sys.argv, which a
# vLLM process never carries. Inject it so B = DECODE_BATCH // TP and T = B * S
# land on the intended shape instead of the module's own default of 1.
_TP = 1


def _decode_seq(impl) -> int:
    speculative_config = impl.vllm_config.speculative_config
    if speculative_config is None or speculative_config.method != "dspark":
        return 1
    return speculative_config.num_speculative_tokens + 1


def _import_kernel():
    argv = sys.argv
    if not any(a == "--tp" or a.startswith("--tp=") for a in argv):
        sys.argv = [*argv, "--tp", str(_TP)]
    try:
        from .pto_kernels.dspark import config as kcfg
        from .pto_kernels.dspark import decode_csa as kcsa
    finally:
        sys.argv = argv
    return kcsa, kcfg


_KCSA = None
_KCFG = None


def kernel():
    global _KCSA, _KCFG
    if _KCSA is None:
        _KCSA, _KCFG = _import_kernel()
    return _KCSA, _KCFG


# --- constants ---------------------------------------------------------------

VLLM_PAGE = 128  # swa / compressed / indexer KV page, in slots
VLLM_STATE_PAGE = 8
COMPRESS_RATIO = 4


class NativeLayoutError(ValueError):
    """The live vLLM allocation does not satisfy the native CSA ABI."""


# --- weights: one-time, cached on the impl -----------------------------------


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
    raise NativeLayoutError(f"weight is {shape}, expected {want} or its transpose")


def _native_tensor(name, tensor, dtype, shape=None):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
        raise NativeLayoutError(f"{name} must retain native {dtype}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise NativeLayoutError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    if not tensor.is_contiguous():
        raise NativeLayoutError(f"{name} must be contiguous; no per-step copy is made")
    return tensor


def _dense(linear, want: tuple):
    # Reject quantized weights: dequantization changes the native W8A8 boundary.
    if getattr(linear, "bias", None) is not None:
        raise NativeLayoutError("dense bias is not supported")
    w = linear.weight.detach()
    _native_tensor("dense weight", w, torch.bfloat16)
    return _oriented(w, want)


def _int8(linear, want: tuple, scale_len: int):
    if getattr(linear, "bias", None) is not None:
        raise NativeLayoutError("quantized bias is not supported")
    offset = getattr(linear, "weight_offset", None)
    if offset is not None and torch.count_nonzero(offset).item() != 0:
        raise NativeLayoutError("kernel requires symmetric INT8 weights")
    w = _to_nd(linear.weight.detach())
    _native_tensor("quantized weight", w, torch.int8)
    scale = getattr(linear, "weight_scale_fp32", None)
    if scale is None:
        scale = getattr(linear, "weight_scale", None)
    _native_tensor("weight scale", scale, torch.float32)
    return _oriented(w, want), _native_tensor(
        "weight scale",
        scale.detach().view(-1),
        torch.float32,
        (scale_len,),
    )


def prepare_weights(impl, hadamard):
    """Build and cache the kernel's weight arguments on the impl."""
    cached = getattr(impl, "_pto_attn_weights", None)
    if cached is not None:
        return cached

    if capture_active():
        raise NativeLayoutError("weights must be prepared before capture")
    kcsa, _ = kernel()
    D, QL, HD = kcsa.D, kcsa.Q_LORA, kcsa.HEAD_DIM
    IH, ID = kcsa.IDX_N_HEADS, kcsa.IDX_HEAD_DIM
    indexer = impl.indexer
    compressor = impl.compressor
    if indexer is None or compressor is None or indexer.compressor is None:
        raise NativeLayoutError("ratio-4 CSA requires indexer and compressor modules")
    wq_b, wq_b_scale = _int8(impl.wq_b, (QL, kcsa.H * HD), kcsa.H * HD)
    idx_wq_b, idx_wq_b_scale = _int8(indexer.wq_b, (QL, IH * ID), IH * ID)
    wo_b, wo_b_scale = _int8(impl.wo_b, (D, kcsa.O_GROUPS * kcsa.O_LORA), D)

    w = {
        "wq_a": _dense(impl.wq_a, (D, QL)),
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": _dense(impl.wkv, (D, HD)),
        "gamma_cq": _native_tensor("gamma_cq", impl.q_norm.weight.detach(), torch.bfloat16),
        "gamma_ckv": _native_tensor("gamma_ckv", impl.kv_norm.weight.detach(), torch.bfloat16),
        "cmp_wkv": _dense(compressor.wkv, (kcsa.MAIN_OUT_DIM, D)),
        "cmp_wgate": _dense(compressor.wgate, (kcsa.MAIN_OUT_DIM, D)),
        "cmp_ape": _native_tensor("cmp_ape", compressor.ape.detach(), torch.float32),
        # Kernel ABI requires FP32 for the RMS norm weight (PyPTO signature is
        # float32); vllm-ascend's process_weights_after_loading widens this from
        # BF16 to FP32, which matches.
        "cmp_norm_w": _native_tensor("cmp_norm_w", compressor.norm.weight.detach(), torch.float32),
        "idx_wq_b": idx_wq_b,
        "idx_wq_b_scale": idx_wq_b_scale,
        "weights_proj": _dense(indexer.weights_proj, (D, IH)),
        # Native rotate_activation keeps the matrix unscaled and applies
        # dim**-0.5 after the matmul. Folding the scale in here would round it
        # into BF16 ahead of the product, so pass the native matrix untouched
        # and let the kernel scale at the native position.
        "hadamard_idx": _native_tensor("hadamard", hadamard, torch.bfloat16, (ID, ID)).T.contiguous(),
        "inner_wkv": _dense(indexer.compressor.wkv, (kcsa.INNER_OUT_DIM, D)),
        "inner_wgate": _dense(indexer.compressor.wgate, (kcsa.INNER_OUT_DIM, D)),
        "inner_ape": _native_tensor("inner_ape", indexer.compressor.ape.detach(), torch.float32),
        # Same: kernel ABI is FP32.
        "inner_norm_w": _native_tensor("inner_norm_w", indexer.compressor.norm.weight.detach(), torch.float32),
        "attn_sink": _native_tensor("attn_sink", impl.attn_sink.detach(), torch.float32),
        # vLLM keeps [G, O_GROUP_IN, O_LORA]; the kernel wants the transpose.
        "wo_a": _native_tensor("wo_a", impl.wo_a.weight.detach(), torch.bfloat16).transpose(1, 2).contiguous(),
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
    }
    impl._pto_attn_weights = w
    return w


# --- per-step derivation: everything below must stay device-only ------------


_DEBUG_REFUSED = set()


def _requests(md):
    """Return the original request objects, without compatibility wrappers."""
    try:
        reqs = (
            md.attention.req_metadata,
            md.compressor.state.req_metadata,
            md.indexer.compressor.state.req_metadata,
            md.indexer.compressor.cache.req_metadata,
            md.swa.req_metadata,
            md.compressor.cache.req_metadata,
        )
    except AttributeError as error:
        raise NativeLayoutError("ratio-4 CSA metadata is incomplete") from error
    if any(req is None for req in reqs):
        raise NativeLayoutError("ratio-4 CSA request metadata is incomplete")
    return reqs


def capture_active() -> bool:
    """Whether an ACLGraph capture is recording right now.

    vllm-ascend clears ``forward_context.capturing`` at the start of every forward
    and sets it immediately before entering the graph context, so it is true for
    the recorded pass and false for the warm-up that precedes it.
    """
    try:
        from vllm.forward_context import get_forward_context

        return bool(getattr(get_forward_context(), "capturing", False))
    except Exception:
        return False


ARG_ORDER = (
    "x_normed",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "kv_projected",
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
    "kv_seq_lens",
    "query_start_loc",
    "attn_sink",
    "ori_slot_mapping",
    "state_slot_mapping",
    "inner_state_slot_mapping",
    "cmp_slot_mapping",
    "idx_slot_mapping",
    "cmp_query_start_loc",
    "cmp_start_pos",
    "idx_query_start_loc",
    "idx_start_pos",
    "inner_freqs_cos",
    "inner_freqs_sin",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "attn_out",
)


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
    view = torch.as_strided(
        cache,
        size=shape,
        stride=strides,
        storage_offset=cache.storage_offset(),
    )
    if not view.is_contiguous():
        raise NativeLayoutError(f"native page view {shape} is not contiguous: {view.stride()}")
    return view


def _cached_page_view(impl, name, cache, rows, row_shape):
    views = getattr(impl, "_pto_page_views", None)
    if views is None:
        views = impl._pto_page_views = {}
    key = (
        id(cache),
        cache.data_ptr(),
        cache.storage_offset(),
        tuple(cache.shape),
        tuple(cache.stride()),
        cache.dtype,
        cache.device,
        rows,
        row_shape,
    )
    previous = views.get(name)
    if previous is None or previous[0] != key:
        previous = (key, cache, _full_page_view(cache, rows, row_shape))
        views[name] = previous
    return previous[2]


def build_args(impl, hidden_states, kv_cache, layer_metadata, seq: int, layer: str, output=None):
    """Bind original 0.29 metadata and cache views; never derive device tensors."""
    if not isinstance(layer, str):
        # RopeDataProxy takes a non-string key as a slice and hands back another
        # proxy, so a wrong name surfaces two frames later as a missing reshape.
        raise NativeLayoutError(f"layer must be the layer's name, got {type(layer).__name__}")
    if len(kv_cache) != 6:
        raise NativeLayoutError("ratio-4 CSA requires 6 native cache views")
    kcsa, _ = kernel()
    cmp_md, cst_md, inner_state_md, idx_md, swa_md, comp_md = _requests(layer_metadata)
    cmp_kv_c, swa_kv_c, state_c, inner_state_c, idx_k_c, idx_s_c = kv_cache

    host_pos = cmp_md.input_positions
    if host_pos.dtype != torch.int64 or host_pos.ndim != 1 or not host_pos.is_contiguous():
        raise NativeLayoutError("input_positions must be contiguous INT64 token rows")
    n_real = layer_metadata.attention.num_decodes
    if not 1 <= n_real <= kcsa.B:
        raise NativeLayoutError(f"{n_real} requests exceed the kernel's B={kcsa.B}")
    if hidden_states.shape[0] < host_pos.shape[0]:
        raise NativeLayoutError(
            f"hidden_states has {hidden_states.shape[0]} rows, expected at least {host_pos.shape[0]}"
        )

    b, t = n_real, host_pos.shape[0]
    if t > kcsa.T:
        raise NativeLayoutError(f"{t} token rows exceed the kernel's T={kcsa.T}")
    pos = host_pos
    seq_lens = _native_tensor("seq_lens", cmp_md.seq_lens, torch.int32)[:b]
    for name, md, expected in (
        ("swa", swa_md, VLLM_PAGE),
        ("compressed", cmp_md, VLLM_PAGE),
        ("index", idx_md, VLLM_PAGE),
        ("state", cst_md, VLLM_STATE_PAGE),
        ("inner state", inner_state_md, VLLM_STATE_PAGE),
    ):
        if md.storage_block_size != expected:
            raise NativeLayoutError(f"{name} logical block size must be {expected}")

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
        "inner state": (inner_state_c, torch.float32),
        "raw KV": (swa_kv_c, torch.bfloat16),
        "compressed KV": (cmp_kv_c, torch.bfloat16),
        "index key page": (idx_k_c, torch.int8),
        "index scale view": (idx_s_c, torch.float16),
    }
    for name, (tensor, dtype) in expected_dtypes.items():
        if tensor.dtype != dtype:
            raise NativeLayoutError(f"{name} must be {dtype}, got {tensor.dtype}")

    a = dict(prepare_weights(impl, layer_metadata.indexer.compressor.cache.hadamard))
    a["x_normed"] = hidden_states[:t]
    a["attn_out"] = output[:t] if output is not None else torch.empty_like(a["x_normed"])
    # The kv LoRA stays on native's own linear: its accumulation order is not
    # reproducible from the DSL, so computing it here is what keeps the
    # projection bit-identical.
    kv_projected = impl.wkv(a["x_normed"])
    if isinstance(kv_projected, tuple):
        kv_projected = kv_projected[0]
    a["kv_projected"] = kv_projected.contiguous()
    for name in ("x_normed", "attn_out"):
        tensor = a[name]
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() or tensor.shape != (t, kcsa.D):
            raise NativeLayoutError(f"{name} must be contiguous BF16 [{t}, {kcsa.D}]")
    if a["kv_projected"].dtype != torch.bfloat16 or a["kv_projected"].shape != (t, kcsa.HEAD_DIM):
        raise NativeLayoutError(
            f"kv_projected must be BF16 [{t}, {kcsa.HEAD_DIM}], got "
            f"{a['kv_projected'].dtype} {tuple(a['kv_projected'].shape)}"
        )

    def rope(name, value):
        _native_tensor(name, value, torch.float32)
        if value.shape[-1] != kcsa.ROPE_HEAD_DIM:
            raise NativeLayoutError(f"{name} must have {kcsa.ROPE_HEAD_DIM} columns")
        return value.view(-1, kcsa.ROPE_HEAD_DIM)

    a["freqs_cos"] = rope("cos", cmp_md.cos[layer])[:t]
    a["freqs_sin"] = rope("sin", cmp_md.sin[layer])[:t]
    if a["freqs_cos"].shape[0] != t or a["freqs_sin"].shape[0] != t:
        raise NativeLayoutError("normal RoPE must cover every token")
    # Native helper performs the required DeviceMetadataStage.COMPRESSOR wait.
    cc, cs, cslots = impl.compressor._compute_metadata(comp_md)
    ic, ins, islots = impl.indexer.compressor._compute_metadata(idx_md)
    for prefix, cos, sin, slots, md in (
        ("cmp", cc, cs, cslots, comp_md),
        ("inner", ic, ins, islots, idx_md),
    ):
        a[prefix + "_freqs_cos"] = rope(prefix + " cos", cos)
        a[prefix + "_freqs_sin"] = rope(prefix + " sin", sin)
        n = a[prefix + "_freqs_cos"].shape[0]
        if a[prefix + "_freqs_sin"].shape[0] != n:
            raise NativeLayoutError("compact RoPE row counts differ")
        slot_name = "cmp_slot_mapping" if prefix == "cmp" else "idx_slot_mapping"
        a[slot_name] = _native_tensor(slot_name, slots, torch.int32, (n, 2))
        key = "cmp" if prefix == "cmp" else "idx"
        a[key + "_query_start_loc"] = _native_tensor(
            key + " query bounds",
            md.query_start_loc,
            torch.int32,
        )[: b + 1]
        a[key + "_start_pos"] = _native_tensor(key + " start_pos", md.start_pos, torch.int32)[:b]
        if a[key + "_query_start_loc"].shape != (b + 1,) or a[key + "_start_pos"].shape != (b,):
            raise NativeLayoutError("compact request metadata does not cover the batch")
    for name, md in (("ori", swa_md), ("state", cst_md), ("inner_state", inner_state_md)):
        a[name + "_slot_mapping"] = _native_tensor(
            name + " slots",
            md.slot_mapping[:t],
            torch.int32,
            (t, 2),
        )

    # Main state and compressed KV are different views of one physical page
    # pool; raw sliding KV uses a separate allocation.
    if state_c.stride(0) != 32768:
        raise NativeLayoutError(f"main state page stride is {state_c.stride(0)}, expected 32768 FP32")
    if swa_kv_c.stride(0) != 65536 or cmp_kv_c.stride(0) != 65536:
        raise NativeLayoutError("raw/compressed KV page stride does not match 131072 bytes")
    a["compress_state_pages"] = _cached_page_view(
        impl,
        "main",
        state_c,
        kcsa.VLLM_COMPRESS_STATE_PAGE_ROWS,
        (kcsa.MAIN_STATE_DIM,),
    )
    a["kv_cache_pages"] = _cached_page_view(
        impl,
        "swa",
        swa_kv_c,
        kcsa.VLLM_KV_PAGE_ROWS,
        (1, kcsa.HEAD_DIM),
    )
    a["cmp_kv_pages"] = _cached_page_view(
        impl,
        "cmp",
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
    if inner_state_c.untyped_storage().data_ptr() != shared_storage:
        raise NativeLayoutError("inner state and index key do not share one vLLM allocation")
    if inner_state_c.data_ptr() != idx_k_c.data_ptr():
        raise NativeLayoutError("inner state and index key do not start at the same physical page")
    if inner_state_c.stride(0) != 4160:
        raise NativeLayoutError(f"inner state page stride is {inner_state_c.stride(0)}, expected 4160 FP32")
    if idx_s_c.untyped_storage().data_ptr() != shared_storage:
        raise NativeLayoutError("index key and scale do not share one vLLM page")
    if idx_k_c.stride(0) != 16640 or idx_s_c.stride(0) != 8320:
        raise NativeLayoutError("index key/scale physical strides do not match the 16640-byte page")
    if idx_s_c.data_ptr() - idx_k_c.data_ptr() != 16384:
        raise NativeLayoutError("index scale does not start at byte 16384 of the packed page")
    a["inner_index_pages"] = _cached_page_view(
        impl,
        "inner",
        idx_k_c,
        kcsa.VLLM_INDEX_PAGE_ROWS,
        (kcsa.IDX_HEAD_DIM,),
    )
    a["index_block_table"] = native_table(
        "index block table",
        idx_md.block_table,
    )
    a["position_ids"] = pos
    a["kv_seq_lens"] = seq_lens
    a["query_start_loc"] = _native_tensor(
        "query_start_loc",
        swa_md.query_start_loc,
        torch.int32,
    )[: b + 1]
    if a["query_start_loc"].shape != (b + 1,):
        raise NativeLayoutError("native query bounds do not cover the batch")

    return [a[name] for name in ARG_ORDER], (pos, n_real)


# --- kernel registration -----------------------------------------------------

_OP = None


def _registered():
    global _OP
    if _OP is None:
        from pypto.torch import init, register

        kcsa, _ = kernel()
        init()
        _OP = register(kcsa.decode_csa_attn_tp1_test, "pypto_csa::attention_csa")
    return _OP


_RAN = [0]


def _decline(reason):
    if reason not in _DEBUG_REFUSED:
        _DEBUG_REFUSED.add(reason)
        print(f"[pto-attn] declined: {reason}", flush=True)
    return False


def substitute(impl, layer, hidden_states, kv_cache, layer_metadata, output, *, cache_is_prepared=False) -> bool:
    """Run the kernel in place of the native attention and publish its result.

    The kernel directly updates
    vLLM's live state and cache pages, and writes directly to its output buffer.
    """
    ratio = getattr(impl, "compress_ratio", 0)
    attention_metadata = layer_metadata.attention
    has_decode = attention_metadata is not None and attention_metadata.num_prefills == 0
    if ratio != COMPRESS_RATIO or not has_decode:
        return False
    if cache_is_prepared:
        return _decline("cache was already prepared by the native path")
    indexer = getattr(impl, "indexer", None)
    if getattr(indexer, "skip_topk", False) or getattr(indexer, "use_index_cache", False):
        return _decline("indexer top-k reuse is not supported by this kernel")
    speculative_config = impl.vllm_config.speculative_config
    if speculative_config is None or speculative_config.method != "dspark":
        return False
    seq = _decode_seq(impl)
    kcsa, _ = kernel()
    try:
        reqs = _requests(layer_metadata)
    except NativeLayoutError as error:
        key = f"metadata:{error}"
        if key not in _DEBUG_REFUSED:
            _DEBUG_REFUSED.add(key)
            print(f"[pto-attn] declined metadata: {error}", flush=True)
        return False
    decode = reqs[0]
    swa = reqs[4]
    left = getattr(swa, "ori_win_left", None)
    right = getattr(swa, "ori_win_right", None)
    if (
        (getattr(impl, "window_size", 128) - 1 if left is None else left) != 127
        or right not in (None, 0)
        or getattr(swa, "dspark_swa_indices", None) is not None
    ):
        return _decline("non-causal or sparse DSpark sliding window is not supported")
    parallel = impl.vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1
        for name in ("tensor_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        return _decline("attention TP/CP must be one")
    position_rows = 0 if decode.input_positions is None else decode.input_positions.shape[0]
    if position_rows == 0:
        return _decline("decode metadata contains no token rows")
    if position_rows > kcsa.T:
        return _decline(f"token rows {position_rows} exceed the kernel capacity T={kcsa.T}")
    n_offered = attention_metadata.num_decodes
    if n_offered > kcsa.B:
        # Under capture this is the padded graph batch, not the live request
        # count, and raising here would abort capture_model with a half-recorded
        # graph. Declining leaves that one descriptor on the native path.
        if "batch" not in _DEBUG_REFUSED:
            _DEBUG_REFUSED.add("batch")
            print(f"[pto-attn] declined a batch of {n_offered}: the kernel takes B={kcsa.B}", flush=True)
        return False
    try:
        args, plan = build_args(
            impl,
            hidden_states,
            kv_cache,
            layer_metadata,
            seq,
            layer,
            output=output,
        )
        _pos, n_real = plan
    except NativeLayoutError as error:
        key = f"layout:{error}"
        if key not in _DEBUG_REFUSED:
            _DEBUG_REFUSED.add(key)
            print(f"[pto-attn] declined native layout: {error}", flush=True)
        return False

    if _OP is None and capture_active():
        return _decline("kernel registration must finish before capture")
    _registered()(*args)

    _RAN[0] += 1
    # Capture happens once per descriptor and Python never runs on replay, so an
    # ungated print there costs one line per captured shape and is the only way
    # to tell a recorded pass from the warm-up that precedes it.
    cap = capture_active()
    if cap or _RAN[0] <= 5 or _RAN[0] % 10 == 0:
        print(f"[pto-attn-ran] n={_RAN[0]} tokens={position_rows} requests={n_real} capturing={cap}", flush=True)
    return True
