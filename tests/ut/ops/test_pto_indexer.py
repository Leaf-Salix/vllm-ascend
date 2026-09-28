"""CPU reference checks for the native Indexer score storage boundaries."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

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


def load_inner_projection(worker):
    """Execute the real DSL projection with CPU FP32 accumulator operations."""
    path = KERNELS / "decode_indexer_compressor.py"
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "indexer_compressor_project_vllm"
    )
    function.decorator_list = []
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None

    def load_tile(tensor, shape, offset, valid_shape):
        # Model the zero-filled invalid rows of a partial cube tile.
        tile = torch.zeros(shape, dtype=tensor.dtype)
        rows, columns = valid_shape
        row, column = offset
        tile[:rows, :columns] = tensor[row : row + rows, column : column + columns]
        return tile

    def accumulate(accumulator, lhs, rhs, b_trans, init_cond):
        product = lhs.float() @ (rhs.float().T if b_trans else rhs.float())
        return product if init_cond else accumulator + product

    pl = SimpleNamespace(
        tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]),
        create_tensor=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        FP32=torch.float32,
        INT32=torch.int32,
        INDEX=torch.int64,
        min=min,
        range=range,
        pipeline=lambda begin, end, stage: range(begin, end),
        spmd=lambda *args, **kwargs: nullcontext(0),
        tile=SimpleNamespace(get_block_idx=lambda: worker),
        read=lambda tensor, index: int(tensor[tuple(index)]),
        # Device comparison masks sign-extend True to -1, not Python's +1.
        cast=lambda value, dtype: -int(value) if isinstance(value, bool) else int(value),
        slice=load_tile,
        matmul_acc=accumulate,
    )
    namespace = dict(
        pl=pl,
        D=1024,
        HEAD_DIM=128,
        OUT_DIM=256,
        MM_B_TILE=16,
        KV_SCORE_WORKERS=24,
        NATIVE_D_BASE=64,
        NATIVE_NARROW_D_PARTS=8,
        NATIVE_PROJ_OUT_TILE=16,
        NATIVE_PROJ_K_TILE=128,
        NATIVE_K_ROTATION_STEP=256,
        NATIVE_NARROW_MAX_TOKENS=1536,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def native_inner_dot(weight, cube_column_group):
    # Directly follow ComputeMm1's h / k / kL0 loops rather than the candidate
    # kernel's flattened loop. Each test tile has at most one nonzero product,
    # so the CPU dot reduction itself cannot hide the inter-tile ordering.
    accumulator = torch.tensor(0.0)
    for h in range(0, len(weight), 512):
        for k in (0, 256):
            h_index = (h + k + cube_column_group * 256) % len(weight)
            for k_l0 in (0, 128):
                accumulator = accumulator + weight[h_index + k_l0 : h_index + k_l0 + 128].float().sum()
    return accumulator


@pytest.mark.parametrize(
    "lengths,d_base",
    [((2, 2), 16), ((1, 3), 64), ((2, 2, 0), 16), ((2, 0, 2), 64)],
    ids=["uniform-dbase16", "tnd-dbase64", "trailing-padding-dbase16", "interior-empty-dbase64"],
)
def test_inner_state_projection_preserves_native_k_order(lengths, d_base):
    # Equal total tokens, different request lengths: replay must recompute
    # native SetBaseSize's column grouping from device query boundaries.
    bounds = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    hidden = torch.ones(sum(lengths), 1024, dtype=torch.bfloat16)
    weights = torch.zeros(256, 1024, dtype=torch.bfloat16)
    weights[:, [0, 256, 512, 768]] = torch.tensor([2.0**24, 1.0, -(2.0**24), 1.0]).bfloat16()
    gate_weights = weights * 0.5
    kv, scores = torch.empty(16, 256), torch.empty(16, 256)
    for worker in range(24):
        load_inner_projection(worker)(hidden, weights, gate_weights, kv, scores, bounds, 0, 0)

    expected = torch.empty(256)
    for half in range(2):
        for group in range(128 // d_base):
            column = half * 128 + group * d_base
            expected[column : column + d_base] = native_inner_dot(weights[0], group)
    torch.testing.assert_close(kv[: len(hidden)], expected.expand(len(hidden), -1), rtol=0, atol=0)
    torch.testing.assert_close(scores[: len(hidden)], (expected * 0.5).expand(len(hidden), -1), rtol=0, atol=0)

    # Unrotated accumulation cannot produce the native column-dependent result.
    assert expected[d_base] != expected[0]
    assert not torch.equal(expected, expected[0].expand_as(expected))


def load_weights_projection():
    """Execute the actual weights path without loading the device runtime."""
    path = KERNELS / "decode_indexer.py"
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "indexer_weights_score_vllm"
    )
    function.decorator_list = []
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None

    def load_tile(tensor, shape, offset, valid_shape):
        tile = torch.zeros(shape, dtype=tensor.dtype)
        rows, columns = valid_shape
        row, column = offset
        tile[:rows, :columns] = tensor[row : row + rows, column : column + columns]
        return tile

    def accumulate(accumulator, lhs, rhs, init_cond):
        product = lhs.float() @ rhs.float()
        return product if init_cond else accumulator + product

    pl = SimpleNamespace(
        tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]),
        create_tensor=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        FP32=torch.float32,
        BF16=torch.bfloat16,
        min=min,
        range=range,
        spmd=lambda *args, **kwargs: nullcontext(0),
        tile=SimpleNamespace(get_block_idx=lambda: 0),
        slice=load_tile,
        matmul_acc=accumulate,
        cast=lambda value, target_type, **kwargs: value.to(target_type),
        mul=torch.mul,
    )
    namespace = dict(
        pl=pl,
        D=4096,
        D_TILE=512,
        T_PAD=16,
        MM_ROW_TILE=16,
        IDX_N_HEADS=2,
        WEIGHTS_SCALE=0.125,
        # Inspect the real weights passed to score selection, not a copied
        # projection implementation. No score computation is needed here.
        indexer_score_topk_forest_vllm=lambda *args: (args[2], None, 0),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name], {arg.arg: None for arg in function.args.args}


@pytest.mark.parametrize("tokens", [1, 13])
def test_weights_projection_preserves_one_accumulator_across_k(tokens):
    hidden = torch.ones(tokens, 4096, dtype=torch.bfloat16)
    weight = torch.zeros(4096, 2, dtype=torch.bfloat16)
    # One nonzero product per K512 tile isolates inter-tile accumulation.
    # Cancelling +/-2**24 first leaves 1 + 2**-8 + 2**-16, just above the
    # BF16 midpoint. Split-K loses 2**-8 inside (-2**24 + 2**-8), so its
    # later reduction rounds to 1 instead of 1 + 2**-7.
    terms = torch.tensor([2.0**24, 0, -(2.0**24), 2.0**-8, 1, 2.0**-16, 0, 0])
    weight[::512, 0] = terms.bfloat16()
    weight[:, 1] = -weight[:, 0]
    project, args = load_weights_projection()
    args.update(x=hidden, weights_proj=weight, weights_workers=1)
    actual, _, _ = project(**args)

    expected = torch.tensor([1 + 2.0**-7, -(1 + 2.0**-7)]) * 0.125
    torch.testing.assert_close(actual[:tokens], expected.expand(tokens, -1), rtol=0, atol=0)
    assert torch.count_nonzero(actual[tokens:]) == 0

    # Negative control: independently model the previous four FP32 partials.
    partials = terms[::2] + terms[1::2]
    old = partials[0]
    for partial in partials[1:]:
        old = old + partial
    old = (old.bfloat16().float() * 0.125).bfloat16().float()
    assert old == 0.125
    assert old != expected[0]
