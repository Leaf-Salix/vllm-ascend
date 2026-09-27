# SPDX-License-Identifier: Apache-2.0
"""单卡检查 Native Score 链路的 NZ 子块读取和单行写回。"""

import importlib
import os

import pytest
import torch
from dsv4_csa_env import activate

activate()

# Source selection must precede PyPTO's first import in this release environment.
pl = importlib.import_module("pypto.language")
RunConfig = importlib.import_module("pypto.runtime").RunConfig


@pl.jit
def native_score_tile(q: pl.Tensor, k: pl.Tensor, c: pl.Tensor, out: pl.InOut[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        query = pl.load(q, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
        key = pl.load(k, [0, 0], [768, 128], target_memory=pl.MemorySpace.Mat)
        coefficient = pl.load(c, [0, 0], [16, 64], target_memory=pl.MemorySpace.Mat)
        for panel in pl.pipeline(0, 768, 128, stage=2):
            key_right = pl.tile.extract(
                pl.tile.transpose_view(key), 0, panel, [128, 128],
                target_memory=pl.MemorySpace.Right,
            )
            query_left = pl.tile.move(query, target_memory=pl.MemorySpace.Left)
            acc = pl.tile.matmul(query_left, key_right)
            score_l1 = pl.tile.create([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
            score_l1 = pl.tile.assemble(score_l1, acc, [0, 0], pre_quant=1.0 / 1024, pre_relu=True)
            reduced = pl.matmul(coefficient, score_l1, out_dtype=pl.FP32)
            row = pl.set_validshape(reduced, 1, 128)
            pl.store(row, [0, panel], out)
    return out


@pytest.mark.skipif("TASK_DEVICE" not in os.environ, reason="通过 task-submit 指定单卡")
def test_native_score_panels_and_one_row_store():
    torch.manual_seed(20260927)
    torch.npu.set_device(int(os.environ["TASK_DEVICE"]))
    q = torch.randint(-64, 65, (64, 128), dtype=torch.int8)
    k = torch.randint(-64, 65, (768, 128), dtype=torch.int8)
    c = (torch.rand(1, 64) * 0.05).half()
    # Independent whole-matrix reference, including Native's FP16 QK step.
    scores_half = ((q.float() @ k.float().T).clamp_min(0) / 1024).half()
    expected = c.float() @ scores_half.float()
    out = torch.full((2, 768), -12345.0, dtype=torch.float32)
    args = (q, k, c.expand(16, 64).contiguous(), out)
    config = RunConfig(platform="a2a3", device_id=int(os.environ["TASK_DEVICE"]))
    native_score_tile.compile(*args, config=config)(*args, config=config)
    result = out.cpu()
    torch.testing.assert_close(result[:1], expected, rtol=2e-4, atol=2e-4)
    assert torch.all(result[1] == -12345.0), "Cube 的重复行不得写入下一行"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
