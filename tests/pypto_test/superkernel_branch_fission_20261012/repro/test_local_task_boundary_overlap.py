# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure same-core overlap with local task statements and unchanged math."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import pto_backend
from pypto.pypto_core.passes import RuntimeKind
from pypto.runtime import ChipWorker, RunConfig


@pytest.mark.platforms("a2a3")
def test_branch_wait_rejects_pre_wait_input_read(test_config: RunConfig, tmp_path: Path):
    """Fission must not turn a branch's unsafe input read into an approved prefix."""

    @pl.jit
    def unsafe_branch(
        x: pl.Tensor[[64], pl.FP32],
        out: pl.Out[pl.Tensor[[64], pl.FP32]],
        choose: pl.Scalar[pl.INT64],
    ):
        with pl.spmd(1, name_hint="unsafe_branch_wait"):
            block = pl.tile.get_block_idx()
            if choose > 0:
                value = pl.tensor.read(x, [block])
                pl.system.wait_pre_task_end()
                pl.tensor.write(out, [block], value + 1.0)
            else:
                value = pl.const(3.0, pl.FP32)
                pl.system.wait_pre_task_end()
                pl.tensor.write(out, [block], value + 2.0)
            pl.system.set_next_task_start()

    with pytest.raises(
        pto_backend.PartialCodegenError,
        match=r"unsafe operation before[\s|]*Scalar-prefix wait boundary",
    ):
        unsafe_branch.compile(
            config=replace(test_config, save_kernels_dir=str(tmp_path / "unsafe_branch")),
            runtime=RuntimeKind.TENSORMAP_AND_RINGBUFFER,
        )


def _program(cube: bool, branch_wait: bool = False):
    blocks = 24 if cube else 48
    tile_rows = 128 if cube else 64
    rows = blocks * tile_rows + 16

    @pl.jit
    def boundary_probe(
        x: pl.Tensor[[rows, 256], pl.FP16],
        y: pl.Tensor[[rows, 256], pl.FP16],
        w: pl.Tensor[[128, 256], pl.FP16],
        a_out: pl.Out[pl.Tensor[[rows, 256], pl.FP32]],
        b_out: pl.Out[pl.Tensor[[rows, 256], pl.FP32]],
        a_trace: pl.Out[pl.Tensor[[blocks + 1, 8], pl.INT64]],
        b_trace: pl.Out[pl.Tensor[[blocks + 1, 8], pl.INT64]],
        seed: pl.Scalar[pl.INT64],
        steps: pl.Scalar[pl.INT64],
        clock_modulus: pl.Scalar[pl.INT64],
    ):
        with pl.spmd(blocks, name_hint="boundary_probe_producer"):
            block = pl.tile.get_block_idx()
            pl.system.wait_pre_task_end()
            # Retain the diagnostic buffer in the outlined signature. Runtime
            # instrumentation overwrites this row after genuine completion.
            pl.tensor.write(a_trace, [block, 0], pl.cast(block, pl.INT64))
            if cube:
                a_l1 = pl.load(x, [block * tile_rows, 0], [tile_rows, 128], target_memory=pl.MemorySpace.Mat)
                b_l1 = pl.load(w, [0, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                a = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
                b = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
                acc = pl.matmul(a, b)
                for _ in pl.range(64 + (block % 4) * 32):
                    acc = pl.matmul_acc(acc, a, b)
                pl.store(acc, [block * tile_rows, 0], a_out)
            else:
                value = pl.cast(pl.load(x, [block * tile_rows, 0], [tile_rows, 256]), pl.FP32)
                for _ in pl.range(1024 + (block % 4) * 512):
                    value = pl.add(value, 1.0)
                pl.store(value, [block * tile_rows, 0], a_out)
            pl.system.set_next_task_start()
        with pl.spmd(blocks, name_hint="boundary_probe_successor"):
            block = pl.tile.get_block_idx()
            core = pl.system.physical_core_id()
            begin = pl.system.read_clock()
            state = seed + pl.cast(block, pl.INT64) + begin % clock_modulus
            for _ in pl.range(steps):
                state = (state * 17 + 3) % 65521
            if branch_wait:
                if block % 2 == 0:
                    state = state * 3 + 1
                    end = pl.system.read_clock(state)
                    pl.system.wait_pre_task_end()
                    after_wait = pl.system.read_clock()
                    observed = pl.tensor.read(y, [block * tile_rows, 0])
                else:
                    state = state * 5 + 2
                    end = pl.system.read_clock(state)
                    pl.system.wait_pre_task_end()
                    after_wait = pl.system.read_clock()
                    observed = pl.tensor.read(y, [block * tile_rows, 0])
            else:
                end = pl.system.read_clock(state)
                pl.system.wait_pre_task_end()
                after_wait = pl.system.read_clock()
                observed = pl.tensor.read(y, [block * tile_rows, 0])
            pl.tensor.write(b_trace, [block, 0], core)
            pl.tensor.write(b_trace, [block, 1], begin)
            pl.tensor.write(b_trace, [block, 2], end)
            pl.tensor.write(b_trace, [block, 3], after_wait)
            pl.tensor.write(b_trace, [block, 4], state)
            pl.tensor.write(b_trace, [block, 5], pl.cast(observed, pl.INT64))
            pl.tensor.write(b_trace, [block, 6], pl.cast(1, pl.INT64))
            if cube:
                a_l1 = pl.load(y, [block * tile_rows, 0], [tile_rows, 128], target_memory=pl.MemorySpace.Mat)
                b_l1 = pl.load(w, [0, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
                a = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
                b = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
                value = pl.matmul(a, b)
                pl.store(value, [block * tile_rows, 0], b_out)
            else:
                value = pl.cast(pl.load(y, [block * tile_rows, 0], [tile_rows, 256]), pl.FP32)
                pl.store(pl.add(value, pl.cast(state, pl.FP32)), [block * tile_rows, 0], b_out)
        return a_out, b_out

    return boundary_probe


def _trace_sample(a, b, repeat: int):
    overlaps = []
    for ai, row in enumerate(a):
        assert 0 < row[1] <= row[2] <= row[5] <= row[7] <= row[3] and row[4] == 7
        for bi, successor in enumerate(b):
            assert 0 < successor[1] <= successor[2] <= successor[3]
            if row[0] != successor[0] or row[2] > successor[1]:
                continue
            assert successor[3] >= row[3], "successor reused its core before the old tail drained"
            if successor[2] <= row[5] and row[7] - row[5] > 5:
                overlaps.append({"predecessor_block": ai, "successor_block": bi})
    return {
        "repeat": repeat,
        "a": a,
        "b": b,
        "overlaps": overlaps,
        "successors_before_last_a_complete": sum(row[3] < max(v[3] for v in a) for row in b),
    }


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("cube", [False, True], ids=["aiv", "aic"])
@pytest.mark.parametrize("branch_wait", [False, True], ids=["flat_wait", "branch_wait"])
def test_local_boundary_overlap(
    test_config: RunConfig, tmp_path: Path, cube: bool, branch_wait: bool, monkeypatch
):
    blocks = 24 if cube else 48
    tile_rows = 128 if cube else 64
    rows = blocks * tile_rows + 16
    name = "local_aic" if cube else "local_aiv"
    suffix = "_branch" if branch_wait else ""
    output = Path(test_config.save_kernels_dir or tmp_path) / f"{name}{suffix}"
    original_wrapper = pto_backend._generate_kernel_wrapper
    traced = []

    def instrument_wrapper(func, *args, **kwargs):
        if func.name == "boundary_probe_producer":
            trace = next(p for p in func.params if p.name_hint.startswith("a_trace"))
            func = ir.Function(
                func.name,
                list(zip(func.params, func.param_directions)),
                func.return_types,
                func.body,
                func.span,
                type=func.func_type,
                level=func.level,
                role=func.role,
                attrs={**dict(func.attrs), "async_return_trace": trace},
                requires_runtime_binding=func.requires_runtime_binding,
                ir_stage=func.ir_stage,
            )
            traced.append(func.name)
        return original_wrapper(func, *args, **kwargs)

    # Only the test harness selects a diagnostic buffer. Task capabilities
    # still come entirely from the local Set/Wait operations.
    monkeypatch.setattr(pto_backend, "_generate_kernel_wrapper", instrument_wrapper)
    compiled = _program(cube, branch_wait).compile(
        config=replace(test_config, save_kernels_dir=str(output)),
        runtime=RuntimeKind.TENSORMAP_AND_RINGBUFFER,
    )
    assert traced == ["boundary_probe_producer"]
    compiled.load()
    if test_config.codegen_only:
        return
    if os.environ.get("TASK_DEVICE") is not None:
        assert str(test_config.device_id) == os.environ["TASK_DEVICE"], "use the queue-assigned card"
    dtype = torch.float16
    x = (torch.arange(rows * 256) % 7 - 3).to(dtype).reshape(rows, 256)
    w = (torch.arange(128 * 256) % 5 - 2).to(torch.float16).reshape(128, 256)
    hosts = [x.clone(), x.clone(), w, torch.empty((rows, 256)), torch.empty((rows, 256))]
    hosts.extend(torch.empty((blocks + 1, 8), dtype=torch.int64) for _ in range(2))
    samples = []
    with ChipWorker(config=test_config, runtime=compiled.runtime_name) as worker:
        tensors = [worker.alloc_tensor(t.shape, t.dtype) for t in hosts]
        try:
            handle = worker.register(compiled)
            for repeat in range(2):
                hosts[0].copy_(x + repeat)
                hosts[1].copy_(x - repeat - 2)
                for host in hosts[3:]:
                    host.fill_(float("nan") if host.dtype == torch.float32 else -1)
                for host, device in zip(hosts, tensors):
                    worker.copy_to(device.data_ptr, host.data_ptr(), device.nbytes)
                seed, steps = 7 + repeat, 64
                handle(*tensors, seed, steps, 1, config=test_config)
                for host, device in zip(hosts[3:], tensors[3:]):
                    worker.copy_from(host.data_ptr(), device.data_ptr, device.nbytes)
                for block in range(blocks):
                    region = slice(block * tile_rows, (block + 1) * tile_rows)
                    state = seed + block
                    for _ in range(steps):
                        state = (state * 17 + 3) % 65521
                    if branch_wait:
                        state = state * (3 if block % 2 == 0 else 5) + (1 if block % 2 == 0 else 2)
                    if cube:
                        expected_a = (hosts[0][region, :128].float() @ w.float()) * (65 + block % 4 * 32)
                        expected_b = hosts[1][region, :128].float() @ w.float()
                    else:
                        expected_a = hosts[0][region].float() + 1024 + block % 4 * 512
                        expected_b = hosts[1][region].float() + state
                    assert torch.equal(hosts[3][region], expected_a)
                    assert torch.equal(hosts[4][region], expected_b)
                    assert hosts[6][block, 4:].tolist() == [state, int(hosts[1][block * tile_rows, 0]), 1, -1]
                for host in hosts[3:5]:
                    assert torch.isnan(host[blocks * tile_rows :]).all(), "output guard was overwritten"
                for host in hosts[5:]:
                    assert torch.equal(host[-1], torch.full((8,), -1, dtype=torch.int64))
                a, b = hosts[5][:-1].tolist(), hosts[6][:-1].tolist()
                samples.append(_trace_sample(a, b, repeat))
        finally:
            for tensor in reversed(tensors):
                worker.free_tensor(tensor)
    (output / "ready_results.json").write_text(
        json.dumps(
            {
                "device_id": test_config.device_id,
                "cube": cube,
                "branch_wait": branch_wait,
                "completion_scope": "all_local_pipelines",
                "samples": samples,
            },
            indent=2,
        )
        + "\n"
    )
    assert any(s["overlaps"] for s in samples), "correct outputs, but no ready-successor pipeline overlap"
    assert any(s["successors_before_last_a_complete"] for s in samples), "independent tasks waited for all A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
