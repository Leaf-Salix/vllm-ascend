import pytest
import torch

from vllm_ascend.ops import pypto_qwen3_full


def test_linear_rejects_unaligned_reduction_dimension():
    x = torch.empty((2, 257), dtype=torch.bfloat16)
    weight = torch.empty((256, 257), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="K % 256 == 0"):
        pypto_qwen3_full.linear(x, weight)
