# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""A gated AIV prefix overlaps unrelated local work while waiting for an AIC input."""

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


def _program(local_work: bool, sync_start: bool):
    @pl.jit
    def remote_dependency_probe(  # noqa: PLR0913 -- Independent data and trace buffers expose cross-core ordering.
        x: pl.Tensor[[3088, 256], pl.FP16],
        w: pl.Tensor[[128, 256], pl.FP16],
        local_out: pl.Out[pl.Tensor[[3088, 256], pl.FP32]],
        remote_out: pl.Out[pl.Tensor[[128, 256], pl.FP32]],
        out: pl.Out[pl.Tensor[[3088, 256], pl.FP32]],
        local_trace: pl.Out[pl.Tensor[[49, 8], pl.INT64]],
        remote_trace: pl.Out[pl.Tensor[[2, 8], pl.INT64]],
        trace: pl.Out[pl.Tensor[[49, 8], pl.INT64]],
        seed: pl.Scalar[pl.INT64],
        steps: pl.Scalar[pl.INT64],
        clock_modulus: pl.Scalar[pl.INT64],
    ):
        if local_work:
            with pl.spmd(48, name_hint="unrelated_local_task", allow_early_resolve=True):
                block = pl.tile.get_block_idx()
                pl.system.wait_pre_task_end()
                pl.tensor.write(local_trace, [block, 0], pl.cast(block, pl.INT64))
                value = pl.cast(pl.load(x, [block * 64, 0], [64, 256]), pl.FP32)
                for _ in pl.range(1024 + (block % 4) * 512):
                    value = pl.add(value, 1.0)
                pl.store(value, [block * 64, 0], local_out)
                pl.system.set_next_task_start()
        with pl.spmd(1, name_hint="remote_data_producer", allow_early_resolve=True):
            block = pl.tile.get_block_idx()
            pl.system.wait_pre_task_end()
            pl.tensor.write(remote_trace, [block, 0], pl.cast(block, pl.INT64))
            lhs_l1 = pl.load(x, [block * 128, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            rhs_l1 = pl.load(w, [0, 0], [128, 256], target_memory=pl.MemorySpace.Mat)
            lhs = pl.move(lhs_l1, target_memory=pl.MemorySpace.Left)
            rhs = pl.move(rhs_l1, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(lhs, rhs)
            for _ in pl.range(4096):
                acc = pl.matmul_acc(acc, lhs, rhs)
            pl.store(acc, [0, 0], remote_out)
            pl.system.set_next_task_start()
        with pl.spmd(48, name_hint="remote_data_consumer", allow_early_resolve=True, sync_start=sync_start):
            block = pl.tile.get_block_idx()
            core = pl.system.physical_core_id()
            begin = pl.system.read_clock()
            state = seed + pl.cast(block, pl.INT64) + begin % clock_modulus
            for _ in pl.range(steps):
                state = (state * 17 + 3) % 65521
            end = pl.system.read_clock(state)
            pl.system.wait_pre_task_end()
            after_wait = pl.system.read_clock()
            value = pl.load(remote_out, [0, 0], [64, 256])
            pl.store(pl.add(value, pl.cast(state, pl.FP32)), [block * 64, 0], out)
            pl.tensor.write(trace, [block, 0], core)
            pl.tensor.write(trace, [block, 1], begin)
            pl.tensor.write(trace, [block, 2], end)
            pl.tensor.write(trace, [block, 3], after_wait)
            pl.tensor.write(trace, [block, 4], state)
        return local_out, remote_out, out

    return remote_dependency_probe


def _validate_local_handoffs(local, consumer, remote, sync_start: bool):
    overlaps = []
    local_predecessor_ends = []
    for prior_block, prior in enumerate(local):
        assert 0 < prior[1] <= prior[2] <= prior[5] <= prior[7] <= prior[3]
        # The low byte identifies the tail; bits 8/9 record notification and late handoff.
        assert prior[4] & 0xFF == 7
        assert prior[4] >> 8 in (0, 1, 2, 3)
        for block, row in enumerate(consumer):
            if prior[0] == row[0] and prior[2] <= row[1]:
                local_predecessor_ends.append(prior[3])
                assert row[3] >= prior[3], "consumer reused its core before the local tail drained"
                if row[2] <= prior[5] and prior[7] - prior[5] > 5 and row[1] < remote[3]:
                    overlaps.append({"local_block": prior_block, "consumer_block": block})
    if sync_start and local_predecessor_ends:
        # X and B have no data edge. An unrelated X block can be scheduled
        # after B; sync_start waits for the predecessors on B's selected cores,
        # not every block of every unrelated task in the graph.
        assert len({row[0] for row in consumer}) == len(consumer)
        assert min(row[3] for row in consumer) >= max(local_predecessor_ends), (
            "sync_start computation began before its selected cores drained"
        )
    return overlaps


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("local_work", [True, False], ids=["pending", "idle"])
@pytest.mark.parametrize("sync_start", [False, True], ids=["regular", "sync_start"])
def test_gated_prefix_with_unrelated_local_task(
    test_config: RunConfig, tmp_path: Path, monkeypatch, local_work: bool, sync_start: bool
):
    case = f"remote_dependency_local{int(local_work)}_sync{int(sync_start)}"
    output = Path(test_config.save_kernels_dir or tmp_path) / case
    original_wrapper = pto_backend._generate_kernel_wrapper
    trace_params = {"unrelated_local_task": "local_trace", "remote_data_producer": "remote_trace"}
    if not local_work:
        del trace_params["unrelated_local_task"]
    traced = []

    def instrument_wrapper(func, *args, **kwargs):
        if func.name in trace_params:
            trace = next(p for p in func.params if p.name_hint.startswith(trace_params[func.name]))
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

    monkeypatch.setattr(pto_backend, "_generate_kernel_wrapper", instrument_wrapper)
    compiled = _program(local_work, sync_start).compile(
        config=replace(test_config, save_kernels_dir=str(output)),
        runtime=RuntimeKind.TENSORMAP_AND_RINGBUFFER,
    )
    assert set(traced) == set(trace_params)
    compiled.load()
    if test_config.codegen_only:
        return
    assert os.environ.get("TASK_DEVICE", str(test_config.device_id)) == str(test_config.device_id), (
        "use the queue-assigned card"
    )

    source = (torch.arange(3088 * 256) % 7 - 3).to(torch.float16).reshape(3088, 256)
    weight = (torch.arange(128 * 256) % 5 - 2).to(torch.float16).reshape(128, 256)
    hosts = [
        source.clone(),
        weight,
        torch.empty((3088, 256)),
        torch.empty((128, 256)),
        torch.empty((3088, 256)),
        torch.empty((49, 8), dtype=torch.int64),
        torch.empty((2, 8), dtype=torch.int64),
        torch.empty((49, 8), dtype=torch.int64),
    ]
    samples = []
    with ChipWorker(config=test_config, runtime=compiled.runtime_name) as worker:
        tensors = [worker.alloc_tensor(t.shape, t.dtype) for t in hosts]
        try:
            handle = worker.register(compiled)
            for repeat in range(2):
                hosts[0].copy_(source + repeat)
                for host in hosts[2:]:
                    host.fill_(float("nan") if host.dtype == torch.float32 else -1)
                for host, device in zip(hosts, tensors):
                    worker.copy_to(device.data_ptr, host.data_ptr(), device.nbytes)
                seed, steps = 7 + repeat, 64
                handle(*tensors, seed, steps, 1, config=test_config)
                for host, device in zip(hosts[2:], tensors[2:]):
                    worker.copy_from(host.data_ptr(), device.data_ptr, device.nbytes)
                expected_remote = (hosts[0][:128, :128].float() @ weight.float()) * 4097
                assert torch.equal(hosts[3], expected_remote)
                local = hosts[5][:-1].tolist()
                remote = hosts[6][0].tolist()
                consumer = hosts[7][:-1].tolist()
                assert 0 < remote[1] <= remote[2] <= remote[5] <= remote[7] <= remote[3]
                assert remote[4] & 0xFF == 7
                assert remote[4] >> 8 in (0, 1, 2, 3)
                for block, row in enumerate(consumer):
                    region = slice(block * 64, (block + 1) * 64)
                    state = seed + block
                    for _ in range(steps):
                        state = (state * 17 + 3) % 65521
                    if local_work:
                        assert torch.equal(
                            hosts[2][region], hosts[0][region].float() + 1024 + block % 4 * 512
                        )
                    assert torch.equal(hosts[4][region], expected_remote[:64] + state)
                    assert row[4:] == [state, -1, -1, -1]
                    assert 0 < row[1] <= row[2] <= row[3]
                    assert row[3] >= remote[3], "consumer entered computation before its AIC input was ready"
                if not local_work:
                    assert torch.isnan(hosts[2]).all()
                    assert torch.equal(hosts[5], torch.full_like(hosts[5], -1))
                # Keep the full timing evidence even if an ordering assertion
                # fails; pytest's abbreviated locals cannot identify the cohort.
                (output / f"remote_dependency_round{repeat}.json").write_text(
                    json.dumps({"local": local, "remote": remote, "consumer": consumer}, indent=2) + "\n"
                )
                overlaps = _validate_local_handoffs(local if local_work else [], consumer, remote, sync_start)
                early_prefixes = sum(row[2] < remote[3] for row in consumer)
                for host in (hosts[2], hosts[4]):
                    assert torch.isnan(host[3072:]).all(), "output guard was overwritten"
                for host in hosts[5:]:
                    assert torch.equal(host[-1], torch.full((8,), -1, dtype=torch.int64))
                samples.append(
                    {
                        "repeat": repeat,
                        "local": local,
                        "remote": remote,
                        "consumer": consumer,
                        "unrelated_local_overlaps": overlaps,
                        "prefixes_before_remote_completion": early_prefixes,
                    }
                )
        finally:
            for tensor in reversed(tensors):
                worker.free_tensor(tensor)
    (output / "remote_dependency_results.json").write_text(
        json.dumps(
            {
                "device_id": test_config.device_id,
                "graph": "AIC A -> AIV B; optional unrelated AIV X",
                "local_work": local_work,
                "sync_start": sync_start,
                "time_unit": "raw device timer ticks",
                "samples": samples,
            },
            indent=2,
        )
        + "\n"
    )
    assert all(s["prefixes_before_remote_completion"] > 0 for s in samples), (
        "no checked prefix executed ahead of the full-fanin gate"
    )
    if local_work:
        assert any(s["unrelated_local_overlaps"] for s in samples), (
            "no gated prefix overlapped unrelated local work"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
