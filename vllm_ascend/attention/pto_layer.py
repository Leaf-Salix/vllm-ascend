# SPDX-License-Identifier: Apache-2.0
"""Complete BSH attention half bound directly to native decoder-layer state.

The numerical chain is the pinned reference kernel. The native layer owns
prepared weights and scratch; native builders own metadata and KV allocations.
"""

import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.attention.pto_layer_storage import indexer_storage, physical_pages, table_storage
from vllm_ascend.attention.pto_layer_weights import prepare_weights
from vllm_ascend.attention.utils import (
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.memcache_comm_fence import record_attention_compute_start

_QUERY_TOKENS = 6
_MAX_BATCH_SIZE = 64
_COMPACT_METADATA_CACHE = "pto_csa_compact_compressor_metadata"


def can_replay_csa_graph(*, num_tokens, num_reqs, uniform_decode, padded_tokens):
    """A captured S6 bucket must not replay nonuniform native requests."""
    captures_pto = 0 < padded_tokens <= _MAX_BATCH_SIZE * _QUERY_TOKENS and padded_tokens % _QUERY_TOKENS == 0
    return not captures_pto or (
        uniform_decode and num_tokens <= padded_tokens and num_tokens == num_reqs * _QUERY_TOKENS
    )


def eligible(layer, context, hidden, positions):
    metadata = context.attn_metadata
    if metadata is None or getattr(context, "is_draft_model", False):
        return False
    if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (4, 4096) or hidden.dtype != torch.bfloat16:
        return False
    tokens = hidden.shape[0]
    if tokens % _QUERY_TOKENS or not 1 <= tokens // _QUERY_TOKENS <= layer._pto_csa_batch_capacity:
        return False
    if not hidden.is_contiguous() or positions.dtype != torch.int64 or positions.shape != (tokens,):
        return False
    batch = tokens // _QUERY_TOKENS
    for prefix in layer._pto_csa_prefixes.values():
        item = metadata.get(prefix)
        if item is None or item.num_prefills or item.num_actual_tokens > tokens:
            return False
        if item.num_actual_tokens % _QUERY_TOKENS or item.num_decodes != batch:
            return False
        req = item.decode
        if req is None or req.seq_lens.numel() != batch or req.query_start_loc.numel() != batch + 1:
            return False
        actual = item.num_actual_tokens // _QUERY_TOKENS
        if req.max_seqlen_q != _QUERY_TOKENS or req.num_reqs_actual not in (None, actual):
            return False
        if req.ori_win_right not in (None, 0) or req.dspark_swa_indices is not None:
            return False
    main = metadata[layer._pto_csa_prefixes["compressed"]]
    name = layer.self_attn.dsa_attn.dsa_attn.layer_name
    if main.decode.cos[name].shape[0] < tokens or main.decode.sin[name].shape[0] < tokens:
        return False
    return metadata[layer._pto_csa_prefixes["swa"]].decode.ori_win_left in (None, 127)


def _compact_metadata(layer, context, req):
    # Native creates a new additional_kwargs for every forward. Retain the
    # object along with its id to prevent accidental identity reuse.
    cache = context.additional_kwargs.setdefault(_COMPACT_METADATA_CACHE, {})
    cached = cache.get(id(req))
    if cached is not None and cached[0] is req:
        return cached[1]
    value = layer.self_attn.dsa_attn.dsa_attn.impl._compute_compressor_metadata(req)
    cache[id(req)] = (req, value)
    return value


def _kernel_arguments(layer, hidden, positions, output, metadata, compact, kv_cache):
    compressed, swa, state, inner_state, indexer_key, indexer_scale = kv_cache
    for name, view, width in (("state", state, 2048), ("indexer_state", inner_state, 512)):
        if view.dtype != torch.float32 or tuple(view.shape[1:]) != (2, 1, width):
            raise ValueError(f"{name}: full CSA requires native two-token FP32 state pages")
    for name, view in (("swa", swa), ("compressed", compressed)):
        if view.dtype != torch.bfloat16 or tuple(view.shape[1:]) != (32, 1, 512) or not view.is_contiguous():
            raise ValueError(f"{name}: full CSA requires native contiguous BF16 32-token pages")
    req = {name: item.decode for name, item in metadata.items()}
    tokens = hidden.shape[0]
    name = layer.self_attn.dsa_attn.dsa_attn.layer_name
    args = dict(layer._pto_csa_weights)
    args.update(
        x_hc=hidden,
        kv_cache=swa,
        cmp_kv=compressed,
        idx_native_kv_cache=indexer_storage(indexer_key, indexer_scale),
        cmp_block_table=table_storage(req["compressed"].block_table),
        idx_block_table=table_storage(req["indexer"].block_table),
        kv_seq_lens=req["indexer"].seq_lens,
        position_ids=positions,
        ori_slot_mapping=req["swa"].slot_mapping,
        state_slot_mapping=req["state"].slot_mapping,
        inner_state_slot_mapping=req["indexer_state"].slot_mapping,
        ori_block_table=table_storage(req["swa"].block_table),
        compress_state=physical_pages(state),
        inner_compress_state=physical_pages(inner_state),
        state_block_table=table_storage(req["state"].block_table),
        inner_state_block_table=table_storage(req["indexer_state"].block_table),
        idx_topk_scores=layer._pto_csa_scores[:tokens],
        idx_topk=layer._pto_csa_topk[:tokens],
        x_out=output,
        cmp_query_start_loc=req["compressed"].query_start_loc,
        cmp_seq_lens=req["compressed"].seq_lens,
        idx_query_start_loc=req["indexer"].query_start_loc,
        freqs_cos=req["compressed"].cos[name][:tokens].view(tokens, 64),
        freqs_sin=req["compressed"].sin[name][:tokens].view(tokens, 64),
    )
    for group, slot, rope in (
        ("compressed", "cmp_slot_mapping", "cmp_freqs"),
        ("indexer", "idx_slot_mapping", "inner_freqs"),
    ):
        cos, sin, slots = compact[group]
        args[slot] = slots
        args[f"{rope}_cos"] = cos.view(-1, 64)
        args[f"{rope}_sin"] = sin.view(-1, 64)
    return tuple(args[name] for name in layer._pto_csa_param_names)


def _native_attention_half(layer, hidden, positions, output):
    residual = hidden.clone()
    mixed, post, comb = layer.hc_pre(hidden, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
    normed = layer.input_layernorm(mixed)
    attention = layer.self_attn(positions=positions, hidden_states=normed, llama_4_scaling=None)
    output.copy_(layer.hc_post(attention, residual, post, comb))


def full_csa_forward(hidden: torch.Tensor, positions: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    from vllm_ascend.ops.dsa import _build_kv_cache

    context = get_forward_context()
    wrapper = context.no_compile_layers[layer_name]
    layer = wrapper._pto_csa_layer[0]
    if not eligible(layer, context, hidden, positions):
        _native_attention_half(layer, hidden, positions, output)
        return
    metadata = {name: context.attn_metadata[prefix] for name, prefix in layer._pto_csa_prefixes.items()}
    hadamard = metadata["indexer"].hadamard
    if hadamard is None:
        raise ValueError("Native indexer metadata did not provide Hadamard")
    if layer._pto_csa_hadamard is not hadamard:
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("CSA requires native metadata warmup before capture")
        layer._pto_csa_weights["hadamard_idx"] = hadamard.detach().T.to(torch.bfloat16).contiguous()
        layer._pto_csa_hadamard = hadamard
    name = wrapper.dsa_attn.layer_name
    wait_for_kv_layer_from_connector(name)
    compact = {group: _compact_metadata(layer, context, metadata[group].decode) for group in ("compressed", "indexer")}
    kv_cache = _build_kv_cache(wrapper, context)
    args = _kernel_arguments(layer, hidden, positions, output, metadata, compact, kv_cache)
    record_attention_compute_start()
    layer._pto_csa_operator(*args)
    notify_kv_cache_written(name)
    maybe_save_kv_layer_to_connector(name, list(kv_cache))


def full_csa_forward_fake(hidden: torch.Tensor, positions: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="full_csa_forward",
    op_func=full_csa_forward,
    mutates_args=["output"],
    fake_impl=full_csa_forward_fake,
    dispatch_key="PrivateUse1",
)


def prepare_model(model, config):
    """Initialize only after native quantization and before graph warmup."""
    import pypto.torch
    from vllm.logger import logger

    from vllm_ascend.attention.dsa_v1 import AscendDSAImpl

    from .pto_kernels.dspark_layer.decode_csa import _decode_csa_tp1_layer, decode_csa_tp1_layer_test
    from .pto_kernels.dspark_layer.reduction import validate_reduction_mode
    from .pto_layer_config import validate_configuration

    validate_configuration(config)
    validate_reduction_mode()
    layers = [layer for layer in model.model.layers if layer.self_attn.compress_ratio == 4]
    if not layers:
        raise ValueError("No target C4 layers for full CSA")
    if any(type(layer.self_attn.dsa_attn.dsa_attn.impl) is not AscendDSAImpl for layer in layers):
        raise ValueError("Full CSA requires the original native DSA fallback, not another CSA implementation")
    pypto.torch.init(device=torch.npu.current_device(), platform="a2a3", runtime="tensormap_and_ringbuffer")
    operator = pypto.torch.register(decode_csa_tp1_layer_test, "pypto_csa::full_attention_half")
    for layer in layers:
        if getattr(layer, "_pto_csa_enabled", False):
            raise RuntimeError("Full CSA was already initialized")
        attention = layer.self_attn
        wrapper = attention.dsa_attn
        layer._pto_csa_weights = prepare_weights(attention, None, layer, root_function=_decode_csa_tp1_layer)
        layer._pto_csa_operator = operator
        layer._pto_csa_param_names = tuple(decode_csa_tp1_layer_test.param_names)
        layer._pto_csa_batch_capacity = min(config.scheduler_config.max_num_seqs, _MAX_BATCH_SIZE)
        tokens = layer._pto_csa_batch_capacity * _QUERY_TOKENS
        layer._pto_csa_scores = torch.empty((tokens, 512), dtype=torch.float32, device=attention.wq_a.weight.device)
        layer._pto_csa_topk = torch.empty((tokens, 512), dtype=torch.int32, device=attention.wq_a.weight.device)
        layer._pto_csa_hadamard = None
        layer._pto_csa_prefixes = {
            "swa": wrapper.swa_cache_layer.prefix,
            "compressed": wrapper.dsa_attn.layer_name,
            "state": attention.compressor.state_cache.prefix,
            "indexer": attention.indexer.k_cache.prefix,
            "indexer_state": attention.indexer.compressor.state_cache.prefix,
        }
        # A direct nn.Module attribute here would create a parent/child cycle.
        wrapper._pto_csa_layer = (layer,)
        layer._pto_csa_enabled = True
    logger.info("Prepared full BSH CSA for %d native C4 layers", len(layers))
