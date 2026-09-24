"""CPU regression for the native O-projection quantization boundary."""

import ast
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/pto_kernels/dspark/decode_o_proj.py"


def load_golden():
    # Load only the CPU reference; importing the kernel requires PyPTO and TP setup.
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "golden_decode_o_proj_tp1")
    namespace = dict(O_GROUPS=2, T_PAD=4, O_GROUP_IN=4, O_LORA=4, D=4, INT8_AMAX_EPS=1e-12, INT8_SCALE_MAX=127.0)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("tokens", [1, 3])
def test_oproj_matches_native_bf16_and_full_row_quantization(tokens):
    generator = torch.Generator().manual_seed(271)
    heads = torch.randn(2, 4, 4, generator=generator).bfloat16()
    heads[1] *= 20  # Unequal group ranges distinguish per-group from shared scale.
    wa = torch.randn(2, 4, 4, generator=generator).bfloat16()
    wb = torch.randint(-127, 128, (4, 8), generator=generator, dtype=torch.int8)
    weight_scale = torch.tensor([0.01, 0.02, 0.03, 0.04])
    heads[:, tokens:] = -2000  # Capacity padding must not enter the row reduction.

    # Native boundary: BF16 grouped matmul, flatten groups, dynamic quantize.
    oa = torch.bmm(heads[:, :tokens], wa.transpose(1, 2))
    flattened = oa.transpose(0, 1).reshape(tokens, 8).float()
    scale = flattened.abs().amax(-1, keepdim=True) / 127
    quantized = torch.round(flattened / scale).to(torch.int8)
    expected = ((quantized.float() @ wb.float().T) * scale * weight_scale).bfloat16()
    actual = load_golden()(heads.reshape(8, 4), wa, wb, weight_scale, tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # The historical per-group quantization gives a different result on this fixture.
    group_scale = oa.float().abs().amax(-1, keepdim=True) / 127
    group_q = torch.round(oa.float() / group_scale)
    old = sum((group_q[g] @ wb[:, g * 4 : (g + 1) * 4].float().T) * group_scale[g] for g in range(2))
    assert not torch.equal((old * weight_scale).bfloat16(), expected)
