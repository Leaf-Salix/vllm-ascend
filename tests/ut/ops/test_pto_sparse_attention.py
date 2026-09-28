"""CPU contracts for the native SCFA chunk and probability boundaries."""

import ast
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/pto_kernels/dspark/decode_sparse_attn_csa.py"


def load_golden():
    # Importing the full module requires PyPTO/NPU; the reference is CPU-only.
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "golden_sparse_attn")
    namespace = dict(
        H=2,
        HEAD_DIM=4,
        WIN=2,
        CMP_TOPK=8,
        COMPRESS_RATIO=4,
        BLOCK_SIZE=2,
        PADDED_TOPK=10,
        TOPK=10,
        ATTN_K_TILE=2,
        NEG_INF=-1e20,
        SOFTMAX_SCALE=0.5,
        NOPE_DIM=2,
        HALF_ROPE=1,
        O_GROUPS=2,
        O_GROUP_IN=4,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name]


def make_tensors(indices, *, window=True, zero_query=True):
    generator = torch.Generator().manual_seed(62)
    q = torch.randn(1, 2, 4, generator=generator).bfloat16()
    if zero_query:
        q.zero_()
    return {
        "q": q,
        "ori_kv": torch.randn(1, 2, 1, 4, generator=generator).bfloat16(),
        "window_swa_indices": torch.tensor([[0, 1] if window else [-1, -1]]),
        "cmp_kv": torch.randn(4, 2, 1, 4, generator=generator).bfloat16(),
        "cmp_block_table": torch.arange(4).reshape(1, 4),
        "idx_topk": torch.tensor([indices]),
        "position_ids": torch.tensor([[39]]),
        "attn_sink": torch.zeros(2),
        "freqs_cos": torch.ones(1, 2),
        "freqs_sin": torch.zeros(1, 2),
        "o_packed_heads": torch.full((2, 4, 4), -100.0, dtype=torch.bfloat16),
    }


@pytest.mark.parametrize(
    ("indices", "window"),
    [
        ([-1] * 8, True),
        ([-1, -1, 0, 1, 2, -1, 4, 5], True),
        ([-1, -1, 0, 1, 2, -1, 4, 5], False),
        (list(range(8)), True),
        ([-1] * 8, False),
    ],
)
def test_sparse_empty_blocks_keep_other_keys_and_sink(indices, window):
    tensors = make_tensors(indices, window=window)
    # Q=0 and sink=0 give each valid key and the sink exactly unit mass.
    # A missing first compressed block must not discard later valid keys.
    values = []
    if window:
        values.extend(tensors["ori_kv"].reshape(2, 4).float())
    compressed = tensors["cmp_kv"].reshape(8, 4).float()
    values.extend(compressed[i] for i in indices if i >= 0)
    expected = torch.stack(values).sum(0) / (len(values) + 1) if values else torch.zeros(4)
    expected = expected.bfloat16().expand(2, 4)
    load_golden()(tensors)
    torch.testing.assert_close(tensors["o_packed_heads"][:, 0], expected, rtol=0, atol=0)
    assert (tensors["o_packed_heads"][:, 1:] == -100).all()


def round_positive_bf16(value):
    # An independent arithmetic definition of CAST_ROUND, including midpoints.
    mantissa, exponent = torch.frexp(value)
    return torch.ldexp(torch.floor(mantissa * 256 + 0.5) / 256, exponent)


def attention_reference(tensors, chunk_widths):
    kv = torch.cat((tensors["ori_kv"].reshape(2, 4), tensors["cmp_kv"].reshape(8, 4))).float()
    scores = tensors["q"][0].float() @ kv.T * 0.5
    maximum = tensors["attn_sink"].reshape(2, 1)
    denominator = torch.ones(2, 1)
    numerator = torch.zeros(2, 4)
    begin = 0
    for width in chunk_widths:
        chunk = scores[:, begin : begin + width]
        next_maximum = torch.maximum(maximum, chunk.max(-1, keepdim=True).values)
        probability = torch.exp(chunk - next_maximum)
        correction = torch.exp(maximum - next_maximum)
        numerator = round_positive_bf16(probability) @ kv[begin : begin + width] + numerator * correction
        denominator = probability.sum(-1, keepdim=True) + denominator * correction
        maximum = next_maximum
        begin += width
    return (numerator / denominator).bfloat16()


def test_compressed_keys_share_one_probability_grid():
    tensors = make_tensors(list(range(8)), zero_query=False)
    # Increase the score spread so normalizing each physical tile separately
    # differs after BF16 probability rounding from one compressed flash chunk.
    tensors["cmp_kv"] *= 3
    expected = attention_reference(tensors, [2, 8])
    historical = attention_reference(tensors, [2, 2, 2, 2, 2])
    assert not torch.equal(expected, historical)
    load_golden()(tensors)
    torch.testing.assert_close(tensors["o_packed_heads"][:, 0], expected, rtol=0, atol=0)
