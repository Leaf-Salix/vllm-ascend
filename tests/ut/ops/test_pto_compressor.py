"""CPU checks for native compressor pooling order and TND state selection.

The actual DSL function runs with Torch operations; this validates its indexing
and operation order, not bitwise equivalence of CPU and NPU Exp/Div instructions.
"""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/pto_kernels/dspark/decode_compressor_ratio4.py"


def load_pool(worker):
    # Loading only this function avoids importing vLLM or initializing PyPTO.
    tree = ast.parse(SOURCE.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compressor_ratio4_pool_projected_vllm"
    )
    function.decorator_list = []
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    pl = SimpleNamespace(
        tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]),
        reshape=torch.reshape,
        create_tensor=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        FP32=torch.float32,
        INT32=torch.int32,
        INDEX=torch.int64,
        min=min,
        range=range,
        spmd=lambda *args, **kwargs: nullcontext(0),
        tile=SimpleNamespace(get_block_idx=lambda: worker),
        read=lambda tensor, index: int(tensor[tuple(index)]),
        cast=lambda value, *args, **kwargs: int(value),
        full=lambda shape, dtype, value: torch.full(shape, value, dtype=dtype),
        add=torch.add,
        maximum=torch.maximum,
        exp=torch.exp,
        mul=torch.mul,
        col_expand_sub=torch.sub,
        col_expand_div=torch.div,
    )
    namespace = dict(
        pl=pl,
        HEAD_DIM=512,
        OUT_DIM=1024,
        STATE_LEN=8,
        BS_PAD=384,
        COMPRESS_STATE_DIM=2048,
        VLLM_COMPRESS_STATE_PAGE_ROWS=16,
        VLLM_COMPRESS_STATE_LIVE_ROWS=8,
        POOL_WORKERS=48,
        POOL_HEAD_TILE=512,
        COMPRESS_RATIO=4,
        FP32_NEG_INF=-1e30,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name]


def pool_case():
    rng = torch.Generator().manual_seed(407)
    return dict(
        compress_state_pages=torch.randn(3, 16, 2048, generator=rng),
        compress_state_block_table=torch.tensor([[1], [2]], dtype=torch.int32),
        ape=torch.randn(4, 1024, generator=rng),
        position_ids=torch.tensor([5, 6, 7, 3, 4, 5, 6, 7], dtype=torch.int32),
        token_valid=torch.ones(8, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 8], dtype=torch.int32),
        pooled_kv=torch.empty(384, 512),
        kv_proj_pad=torch.randn(384, 1024, generator=rng),
        score_proj_pad=torch.randn(384, 1024, generator=rng) * 5,
        late_dep=0,
    )


def native_column_fold(values, operation):
    # compressor_vector_comm.h::ColumnSum/ColumnMax fold rows in half.
    scratch = values.clone()
    rows = scratch.shape[0]
    while rows > 1:
        rows //= 2
        scratch[:rows] = operation(scratch[:rows], scratch[rows : 2 * rows])
    return scratch[0]


def native_last_window(case, request):
    begin = int(case["query_start_loc"][request])
    first_position = int(case["position_ids"][begin])
    page = int(case["compress_state_block_table"][request, 0])
    values = torch.zeros(8, 512)
    scores = torch.full((8, 512), -1e30)
    # Both requests end at position 7. Build chronological window members;
    # Native PadAlign then interleaves the older and newer groups of four.
    for position in range(8):
        half = (position // 4) * 512
        if position < first_position:
            state = case["compress_state_pages"][page, position]
            values[position] = state[half : half + 512]
            scores[position] = state[1024 + half : 1024 + half + 512]
        else:
            token = begin + position - first_position
            if case["token_valid"][token]:
                values[position] = case["kv_proj_pad"][token, half : half + 512]
                scores[position] = (
                    case["score_proj_pad"][token, half : half + 512] + case["ape"][position % 4, half : half + 512]
                )
    physical_order = [0, 4, 1, 5, 2, 6, 3, 7]
    values, scores = values[physical_order], scores[physical_order]
    exponent = (scores - native_column_fold(scores, torch.maximum)).exp()
    probabilities = exponent / native_column_fold(exponent, torch.add)
    return native_column_fold(values * probabilities, torch.add)


@pytest.mark.parametrize("invalid_overlay", [False, True])
def test_pool_matches_native_column_order_for_nonuniform_tnd(invalid_overlay):
    case = pool_case()
    if invalid_overlay:
        case["token_valid"][4] = 0
    for request in range(2):
        # Execute each SPMD worker separately against the same output buffer.
        load_pool(request)(**case)
        begin, end = (int(x) for x in case["query_start_loc"][request : request + 2])
        torch.testing.assert_close(case["pooled_kv"][end - 1], native_last_window(case, request), rtol=0, atol=0)
        for token in range(begin, end):
            if case["position_ids"][token] % 4 != 3 or not case["token_valid"][token]:
                torch.testing.assert_close(case["pooled_kv"][token], torch.zeros(512), rtol=0, atol=0)


def test_pool_preserves_native_rounding_instead_of_online_accumulation():
    case = pool_case()
    for name in ("compress_state_pages", "ape", "kv_proj_pad", "score_proj_pad"):
        case[name].zero_()
    # Equal logits isolate the FP32 addition tree. At 2**24, a unit increment
    # is a rounding boundary; the native tree and historical left fold differ.
    values = torch.tensor([0.0, 0.0, 1.0, 2.0**24, 0.0, 0.0, 0.0, 1.0])
    case["compress_state_pages"][1, 2, :512] = values[2]
    case["compress_state_pages"][1, 3, :512] = values[3]
    case["kv_proj_pad"][2, 512:] = values[7]

    load_pool(0)(**case)
    expected = torch.full((512,), 2.0**21)
    torch.testing.assert_close(case["pooled_kv"][2], expected, rtol=0, atol=0)
    torch.testing.assert_close(native_last_window(case, 0), expected, rtol=0, atol=0)

    # With equal logits, the old online algorithm seeds the newest value,
    # adds the older values sequentially, and divides by the window length.
    online_sum = values[-1].clone()
    for value in values[:-1]:
        online_sum = online_sum + value
    historical = online_sum / len(values)
    assert historical.item() == 2.0**21 + 0.25
    assert historical != expected[0]


@pytest.mark.parametrize("lengths,expected", [([6, 6, 6, 6], 32), ([3, 4, 5, 6], 64), ([6, 6, 0], 64), ([6] * 65, 64)])
def test_projection_uses_native_equal_length_tiling(lengths, expected):
    function = next(
        node
        for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "compressor_ratio4_project_vllm"
    )
    body = next(node for node in function.body if isinstance(node, ast.With)).body
    end = next(i for i, node in enumerate(body) if isinstance(node, ast.For) and node.target.id == "unit")
    bounds = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0)
    # The actual compiler sign-extends a predicate: True becomes -1. This
    # regression failed for equal lengths when the code counted bool casts.
    pl = SimpleNamespace(
        tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]),
        tile=SimpleNamespace(get_block_idx=lambda: 0),
        range=range,
        INT32=torch.int32,
        read=lambda tensor, index: int(tensor[tuple(index)]),
        cast=lambda value, dtype: -int(value) if isinstance(value, bool) else int(value),
    )
    namespace = dict(
        pl=pl,
        query_start_loc=bounds,
        NATIVE_D_BASE=64,
        NATIVE_NARROW_MAX_TOKENS=384,
        HEAD_DIM=512,
        NATIVE_NARROW_D_PARTS=16,
    )
    exec(compile(ast.Module(body=body[:end], type_ignores=[]), str(SOURCE), "exec"), namespace)
    assert namespace["d_base"] == expected
