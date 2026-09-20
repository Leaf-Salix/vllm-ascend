"""Validate one fused Qwen3 attention residual block on an NPU."""

from __future__ import annotations

import argparse

import torch
import torch_npu

from vllm_ascend.ops import pypto_qwen3_full as qwen3


def _native_block(
    *,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    input_norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = hidden_states.shape[0]
    if residual is None:
        attention_residual = hidden_states
        normalized, _ = torch_npu.npu_rms_norm(hidden_states, input_norm_weight, 1e-6)
    else:
        normalized, _, attention_residual = torch_npu.npu_add_rms_norm(
            hidden_states,
            residual,
            input_norm_weight,
            1e-6,
        )
    qkv = torch.nn.functional.linear(normalized, qkv_weight)
    query, key, value = qkv.split((5120, 1024, 1024), dim=-1)
    query, _ = torch_npu.npu_rms_norm(query.view(rows, 40, 128), q_norm_weight, 1e-6)
    key, _ = torch_npu.npu_rms_norm(key.view(rows, 8, 128), k_norm_weight, 1e-6)
    query = query.view(rows, 5120)
    key = key.view(rows, 1024)
    torch_npu._npu_rotary_embedding(positions, query, key, 128, cos_sin_cache, True)
    key_cache.view(-1, 1024).index_copy_(0, slot_mapping.to(torch.int64), key)
    value_cache.view(-1, 1024).index_copy_(0, slot_mapping.to(torch.int64), value)
    attention = torch.empty((rows, 40, 128), dtype=torch.bfloat16, device=hidden_states.device)
    torch_npu._npu_paged_attention(
        query=query.view(rows, 40, 128),
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=8,
        num_heads=40,
        scale_value=1.0 / (128**0.5),
        block_table=block_tables,
        context_lens=seq_lens.cpu(),
        out=attention,
    )
    projected = torch.nn.functional.linear(attention.view(rows, 5120), o_proj_weight)
    mlp_input, _, updated_residual = torch_npu.npu_add_rms_norm(
        projected,
        attention_residual,
        post_attention_norm_weight,
        1e-6,
    )
    return mlp_input, updated_residual


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    difference = (actual.float() - expected.float()).abs()
    print(name, "max=", float(difference.max()), "mean=", float(difference.mean()), flush=True)
    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=8e-2)


def _test_replay(
    *,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    input_norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    actual_key_cache = torch.zeros((2, 128, 8, 128), dtype=torch.bfloat16, device=hidden_states.device)
    actual_value_cache = torch.zeros_like(actual_key_cache)
    actual_mlp_input = torch.empty_like(hidden_states)
    actual_updated_residual = torch.empty_like(hidden_states)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        qwen3.attention_residual_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            input_norm_weight=input_norm_weight,
            qkv_weight=qkv_weight,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            cos_sin_cache=cos_sin_cache,
            o_proj_weight=o_proj_weight,
            post_attention_norm_weight=post_attention_norm_weight,
            slot_mapping=slot_mapping,
            key_cache=actual_key_cache.view(-1, 1024),
            value_cache=actual_value_cache.view(-1, 1024),
            block_table=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            mlp_input_out=actual_mlp_input,
            updated_residual_out=actual_updated_residual,
        )

    for replay, physical_block in enumerate((0, 1), start=1):
        hidden_states.copy_(torch.randn_like(hidden_states))
        residual.copy_(torch.randn_like(residual))
        slot_mapping.fill_(physical_block * 128)
        block_tables.fill_(physical_block)
        actual_key_cache.zero_()
        actual_value_cache.zero_()
        expected_key_cache = torch.zeros_like(actual_key_cache)
        expected_value_cache = torch.zeros_like(actual_value_cache)
        expected_mlp_input, expected_updated_residual = _native_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            input_norm_weight=input_norm_weight,
            qkv_weight=qkv_weight,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            cos_sin_cache=cos_sin_cache,
            o_proj_weight=o_proj_weight,
            post_attention_norm_weight=post_attention_norm_weight,
            slot_mapping=slot_mapping,
            key_cache=expected_key_cache,
            value_cache=expected_value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
        )
        graph.replay()
        torch.npu.synchronize()
        _assert_close(f"replay{replay}.mlp_input", actual_mlp_input, expected_mlp_input)
        _assert_close(
            f"replay{replay}.updated_residual",
            actual_updated_residual,
            expected_updated_residual,
        )
        _assert_close(f"replay{replay}.key_cache", actual_key_cache, expected_key_cache)
        _assert_close(f"replay{replay}.value_cache", actual_value_cache, expected_value_cache)

    graph.reset()


def _test_cross_page(
    *,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    input_norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
) -> None:
    device = hidden_states.device
    positions = torch.tensor([128], dtype=torch.int64, device=device)
    slot_mapping = torch.tensor([128], dtype=torch.int32, device=device)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([129], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    initial_key_cache = torch.randn((2, 128, 8, 128), dtype=torch.bfloat16, device=device)
    initial_value_cache = torch.randn_like(initial_key_cache)
    expected_key_cache = initial_key_cache.clone()
    expected_value_cache = initial_value_cache.clone()
    actual_key_cache = initial_key_cache.clone()
    actual_value_cache = initial_value_cache.clone()

    expected_mlp_input, expected_updated_residual = _native_block(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
        input_norm_weight=input_norm_weight,
        qkv_weight=qkv_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=cos_sin_cache,
        o_proj_weight=o_proj_weight,
        post_attention_norm_weight=post_attention_norm_weight,
        slot_mapping=slot_mapping,
        key_cache=expected_key_cache,
        value_cache=expected_value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
    )
    actual_mlp_input, actual_updated_residual = qwen3.attention_residual_block(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
        input_norm_weight=input_norm_weight,
        qkv_weight=qkv_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=cos_sin_cache,
        o_proj_weight=o_proj_weight,
        post_attention_norm_weight=post_attention_norm_weight,
        slot_mapping=slot_mapping,
        key_cache=actual_key_cache.view(-1, 1024),
        value_cache=actual_value_cache.view(-1, 1024),
        block_table=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    torch.npu.synchronize()
    _assert_close("cross_page.mlp_input", actual_mlp_input, expected_mlp_input)
    _assert_close(
        "cross_page.updated_residual",
        actual_updated_residual,
        expected_updated_residual,
    )
    _assert_close("cross_page.key_cache", actual_key_cache, expected_key_cache)
    _assert_close("cross_page.value_cache", actual_value_cache, expected_value_cache)


def _test_padded_slot(
    *,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    input_norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
) -> None:
    device = hidden_states.device
    initial_key_cache = torch.randn((1, 128, 8, 128), dtype=torch.bfloat16, device=device)
    initial_value_cache = torch.randn_like(initial_key_cache)
    actual_key_cache = initial_key_cache.clone()
    actual_value_cache = initial_value_cache.clone()
    qwen3.attention_residual_block(
        positions=torch.tensor([0], dtype=torch.int64, device=device),
        hidden_states=hidden_states,
        residual=residual,
        input_norm_weight=input_norm_weight,
        qkv_weight=qkv_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=cos_sin_cache,
        o_proj_weight=o_proj_weight,
        post_attention_norm_weight=post_attention_norm_weight,
        slot_mapping=torch.tensor([-1], dtype=torch.int32, device=device),
        key_cache=actual_key_cache.view(-1, 1024),
        value_cache=actual_value_cache.view(-1, 1024),
        block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([0], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual_key_cache, initial_key_cache, rtol=0, atol=0)
    torch.testing.assert_close(actual_value_cache, initial_value_cache, rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20260920)
    qwen3.init()
    qwen3.registered_attention_block_ops()
    device = torch.device("npu", args.device)

    rows = 1
    hidden_states = torch.randn((rows, 5120), dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(hidden_states)
    input_norm_weight = torch.randn((5120,), dtype=torch.bfloat16, device=device)
    qkv_weight = torch.randn((7168, 5120), dtype=torch.bfloat16, device=device) / 64
    q_norm_weight = torch.randn((128,), dtype=torch.bfloat16, device=device)
    k_norm_weight = torch.randn((128,), dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn((512, 128), dtype=torch.bfloat16, device=device)
    o_proj_weight = torch.randn((5120, 5120), dtype=torch.bfloat16, device=device) / 64
    post_attention_norm_weight = torch.randn((5120,), dtype=torch.bfloat16, device=device)
    positions = torch.tensor([0], dtype=torch.int64, device=device)
    slot_mapping = torch.tensor([128], dtype=torch.int32, device=device)
    block_tables = torch.tensor([[1]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([1], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)

    for case_residual in (None, residual):
        expected_key_cache = torch.zeros((2, 128, 8, 128), dtype=torch.bfloat16, device=device)
        expected_value_cache = torch.zeros_like(expected_key_cache)
        actual_key_cache = expected_key_cache.clone()
        actual_value_cache = expected_value_cache.clone()
        expected_mlp_input, expected_updated_residual = _native_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=case_residual,
            input_norm_weight=input_norm_weight,
            qkv_weight=qkv_weight,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            cos_sin_cache=cos_sin_cache,
            o_proj_weight=o_proj_weight,
            post_attention_norm_weight=post_attention_norm_weight,
            slot_mapping=slot_mapping,
            key_cache=expected_key_cache,
            value_cache=expected_value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
        )
        actual_mlp_input, actual_updated_residual = qwen3.attention_residual_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=case_residual,
            input_norm_weight=input_norm_weight,
            qkv_weight=qkv_weight,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            cos_sin_cache=cos_sin_cache,
            o_proj_weight=o_proj_weight,
            post_attention_norm_weight=post_attention_norm_weight,
            slot_mapping=slot_mapping,
            key_cache=actual_key_cache.view(-1, 1024),
            value_cache=actual_value_cache.view(-1, 1024),
            block_table=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
        )
        torch.npu.synchronize()
        case = "first" if case_residual is None else "regular"
        _assert_close(f"{case}.mlp_input", actual_mlp_input, expected_mlp_input)
        _assert_close(f"{case}.updated_residual", actual_updated_residual, expected_updated_residual)
        _assert_close(f"{case}.key_cache", actual_key_cache[1, 0], expected_key_cache[1, 0])
        _assert_close(f"{case}.value_cache", actual_value_cache[1, 0], expected_value_cache[1, 0])

    _test_replay(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
        input_norm_weight=input_norm_weight,
        qkv_weight=qkv_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=cos_sin_cache,
        o_proj_weight=o_proj_weight,
        post_attention_norm_weight=post_attention_norm_weight,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
    )
    _test_cross_page(
        hidden_states=hidden_states,
        residual=residual,
        input_norm_weight=input_norm_weight,
        qkv_weight=qkv_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=cos_sin_cache,
        o_proj_weight=o_proj_weight,
        post_attention_norm_weight=post_attention_norm_weight,
    )
    _test_padded_slot(
        hidden_states=hidden_states,
        residual=residual,
        input_norm_weight=input_norm_weight,
        qkv_weight=qkv_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        cos_sin_cache=cos_sin_cache,
        o_proj_weight=o_proj_weight,
        post_attention_norm_weight=post_attention_norm_weight,
    )

    print("QWEN3_ATTENTION_RESIDUAL_BLOCK PASS", flush=True)


if __name__ == "__main__":
    main()
