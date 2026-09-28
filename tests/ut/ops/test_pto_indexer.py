"""CPU reference checks for the native Indexer score storage boundaries."""

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / "vllm_ascend/attention/pto_kernels/dspark"


def load_golden():
    # Keep this CPU-only: importing the DSL module initializes the PyPTO runtime.
    functions = []
    for filename, name in (("utils.py", "int8_quant_per_row"), ("decode_indexer.py", "golden_indexer")):
        tree = ast.parse((KERNELS / filename).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        function.body = [n for n in function.body if not isinstance(n, ast.ImportFrom)]
        functions.append(function)
    namespace = dict(
        torch=torch,
        golden_compressor=lambda tensors: None,
        S=1,
        D=4,
        COMPRESS_RATIO=4,
        ROPE_HEAD_DIM=2,
        IDX_N_HEADS=2,
        IDX_HEAD_DIM=4,
        HADAMARD_SCALE=0.5,
        WEIGHTS_SCALE=0.125,
        INT8_SCALE_MAX=127.0,
        INT8_AMAX_EPS=1e-12,
        IDX_TOPK=3,
        BLOCK_SIZE=8,
        TOPK_MAX_CANDIDATES=8,
        TOPK_CANDIDATES_PER_LEAF=8,
        FP32_NEG_INF=float("-inf"),
        SCORE_DEQUANT_SCALE=2.0**-10,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(KERNELS), "exec"), namespace)
    return namespace["golden_indexer"]


@pytest.mark.parametrize("weight_magnitude", [1.0, 2.0**-24])
def test_indexer_reference_matches_native_half_score_contract(weight_magnitude):
    rng = torch.Generator().manual_seed(8103)
    data = dict.fromkeys(
        (
            "inner_kv",
            "inner_wkv",
            "inner_wgate",
            "inner_ape",
            "inner_norm_w",
            "cmp_cos",
            "cmp_sin",
            "inner_compress_state",
            "inner_compress_state_block_table",
            "idx_slot_mapping",
            "inner_state_slot_mapping",
        ),
        torch.empty(0),
    )
    data.update(
        x=torch.randn(1, 4, generator=rng).bfloat16(),
        qr=torch.randint(-127, 128, (1, 4), generator=rng, dtype=torch.int8),
        qr_scale=torch.tensor([[0.0327]]),
        wq_b=torch.randint(-127, 128, (4, 8), generator=rng, dtype=torch.int8),
        wq_b_scale=torch.rand(8, generator=rng) * 0.003,
        weights_proj=(torch.randn(4, 2, generator=rng) * weight_magnitude).bfloat16(),
        cos=torch.ones(1, 2),
        sin=torch.zeros(1, 2),
        hadamard=torch.eye(4).bfloat16(),
        kv_seq_lens=torch.tensor([32], dtype=torch.int32),
        position_ids=torch.tensor([31], dtype=torch.int32),
        idx_kv_cache=torch.randint(-127, 128, (1, 8, 1, 4), generator=rng, dtype=torch.int8),
        idx_kv_scale=(torch.rand(1, 8, 1, 1, generator=rng) * 0.05).half(),
        idx_block_table=torch.zeros(1, 1, dtype=torch.int32),
        topk_scores=torch.empty(1, 3),
        topk_idxs=torch.empty(1, 3, dtype=torch.int32),
    )

    # Model the public native stages using BF16 Linear outputs and half QLI
    # operands. Identity RoPE keeps this fixture focused on score boundaries.
    factor = data["qr_scale"] * data["wq_b_scale"]
    projected = ((data["qr"].float() @ data["wq_b"].float()) * factor).bfloat16()
    query = F.linear(projected.view(2, 4), data["hadamard"]) * 0.5
    multiplier = 127.0 / query.float().abs().amax(-1, keepdim=True)
    codes = torch.round(query.float() * multiplier).to(torch.int8)
    query_scale = multiplier.reciprocal().half()
    weights = (F.linear(data["x"], data["weights_proj"].T) * 0.125).half().view(2, 1)
    coefficient = weights * query_scale
    dots = codes.int() @ data["idx_kv_cache"].view(8, 4).int().T
    stored_dots = (dots.clamp_min(0).float() / 1024).half()
    scores = (stored_dots.float() * coefficient.float()).sum(0) * data["idx_kv_scale"].float().flatten()
    expected_scores, expected_indices = scores.topk(3)

    load_golden()(data)
    torch.testing.assert_close(data["topk_scores"][0], expected_scores, rtol=0, atol=0)
    torch.testing.assert_close(data["topk_idxs"][0], expected_indices.int(), rtol=0, atol=0)

    if weight_magnitude == 1.0:
        # The historical FP32 score path loses these native storage boundaries.
        old = (dots.float().clamp_min(0) * query_scale.float() * weights.float()).sum(0)
        old *= data["idx_kv_scale"].float().flatten() / 1024
        assert not torch.equal(old.topk(3).values, expected_scores)
