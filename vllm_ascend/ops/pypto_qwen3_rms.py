# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Experimental Qwen3 q/k RMSNorm executed by PyPTO L2 kernel mode."""

import pypto.language as pl
import torch

_ROWS = pl.dynamic("QWEN3_QK_RMS_ROWS")
_HEAD_DIM = 128
_TILE_ROWS = 8
_EPS = 1e-6
_warmed_devices: set[int] = set()


@pl.jit
def _qk_rms(
    x: pl.Tensor[[_ROWS, _HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[1, _HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[_ROWS, _HEAD_DIM], pl.BF16]],
) -> pl.Tensor[[_ROWS, _HEAD_DIM], pl.BF16]:
    x.bind_dynamic(0, _ROWS)
    out.bind_dynamic(0, _ROWS)
    rows = pl.tensor.dim(x, 0)
    for row in pl.parallel(0, rows, _TILE_ROWS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_qk_rms"):
            tile = pl.cast(pl.load(x, [row, 0], [_TILE_ROWS, _HEAD_DIM]), pl.FP32)
            tmp = pl.create_tile([_TILE_ROWS, _HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            sq = pl.row_sum(pl.mul(tile, tile), tmp)
            inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq, 1.0 / _HEAD_DIM), _EPS)), [_TILE_ROWS, 1])
            gamma = pl.cast(pl.load(weight, [0, 0], [1, _HEAD_DIM]), pl.FP32)
            normalized = pl.col_expand_mul(pl.row_expand_mul(tile, inv), gamma)
            out = pl.store(pl.cast(normalized, pl.BF16), [row, 0], out)
    return out


def run(x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Use only device-owned buffers; no tensor allocation occurs in capture."""
    if x.ndim != 3 or x.shape[-1] != _HEAD_DIM or x.shape[1] not in (8, 40):
        raise ValueError("Qwen3 q/k RMSNorm requires [tokens, 8|40, 128] input")
    if weight.shape != (_HEAD_DIM,) or output.shape != x.shape:
        raise ValueError("Qwen3 q/k RMSNorm weight/output shape mismatch")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise ValueError("Qwen3 q/k RMSNorm requires BF16 input, weight, and output")
    if any(t.device.type != "npu" or not t.is_contiguous() for t in (x, weight, output)):
        raise ValueError("Qwen3 q/k RMSNorm requires contiguous NPU tensors")
    rows = x.shape[0] * x.shape[1]
    _qk_rms(x.view(rows, _HEAD_DIM), weight.view(1, _HEAD_DIM), output.view(rows, _HEAD_DIM))
    return output


def registered_op():
    """Expose the same JIT specialization as an opaque torch dispatcher op."""
    from pypto.torch import register

    return register(_qk_rms, "vllm_ascend_pypto::qwen3_qk_rms")


def is_warmed(device: torch.device) -> bool:
    index = torch.device(device).index
    if index is None:
        index = torch.npu.current_device()
    return index in _warmed_devices


def warmup(device: torch.device) -> None:
    """Initialize and prepare the one dynamic callable before graph capture."""
    if is_warmed(device):
        return
    from pypto.torch import init as pypto_init

    from vllm_ascend.ops.pypto_swimlane import init_pypto

    init_pypto(pypto_init)
    x = torch.ones((1, 8, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    weight = torch.ones((_HEAD_DIM,), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)
    run(x, weight, out)
    torch.npu.synchronize(device)
    index = torch.device(device).index
    _warmed_devices.add(torch.npu.current_device() if index is None else index)
