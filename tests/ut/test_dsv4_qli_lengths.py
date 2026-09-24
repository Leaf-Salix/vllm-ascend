# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise the production QLI builder method without an NPU installation."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_shared_metadata_refreshes_each_builders_lengths(monkeypatch, dtype):
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/attention/dsa_v1.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendDSAMetadataBuilder")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_qli_metadata")
    namespace = {
        "torch": torch,
        "DSA_METADATA_BUFFER_SIZE": 4,
        "DeviceOperator": NS(get_dsa_indexer_quant_mode=lambda: 0),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    build = namespace[method.name]
    op = Mock(side_effect=lambda **kwargs: torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata", op, raising=False)
    builders = [
        NS(
            model_config=NS(hf_config=NS(index_n_heads=64, index_head_dim=128, index_topk=2048)),
            qli_seqused_k=torch.full((4,), -99, dtype=torch.int32),
            qli_cmp_residual_k=torch.full((4,), -99, dtype=torch.int32),
            qli_metadata_buffer=torch.zeros(4, dtype=torch.int32),
            seqused_q=torch.zeros(4, dtype=torch.int32),
        )
        for _ in range(2)
    ]
    pointers = [(b.qli_seqused_k.data_ptr(), b.qli_cmp_residual_k.data_ptr()) for b in builders]
    for step, lengths in enumerate(([8191, 8192, 8193], [8198, 8203])):
        shared = {}  # Shared within one step, across independent cache-group builders.
        lens = torch.tensor(lengths, dtype=dtype)
        bounds = torch.arange(len(lengths) + 1, dtype=torch.int32) * 6
        for b, expected_ptrs in zip(builders, pointers):
            output = build(b, shared, bounds, lens, 6, max(lengths))
            assert output is b.qli_metadata_buffer
            assert torch.equal(output, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
            assert torch.equal(b.qli_seqused_k[: len(lengths)], lens // 4)
            assert torch.equal(b.qli_cmp_residual_k[: len(lengths)], lens % 4)
            assert (b.qli_seqused_k.data_ptr(), b.qli_cmp_residual_k.data_ptr()) == expected_ptrs
        assert op.call_count == step + 1
