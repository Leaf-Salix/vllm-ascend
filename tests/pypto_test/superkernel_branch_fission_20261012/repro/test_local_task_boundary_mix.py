# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Local boundary propagation through a mixed Cube/Vector task."""

import json
import re
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.backend import pto_backend
from pypto.pypto_core.passes import RuntimeKind
from pypto.runtime import ChipWorker, RunConfig

from .task_boundary_explicit_event_probe import explicit_sync


@pl.jit
def mixed_boundary_probe(
    x: pl.Tensor[[128, 128], pl.FP16],
    w: pl.Tensor[[128, 128], pl.FP16],
    bias: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.spmd(
        1,
        name_hint="mixed_boundary",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        allow_early_resolve=True,
    ):
        block = pl.tile.get_block_idx()
        pl.system.wait_pre_task_end()
        acc = pl.matmul(x, w, out_dtype=pl.FP32)
        out[:] = pl.add(acc, pl.add(bias, pl.cast(pl.cast(block, pl.INT64), pl.FP32)))
        pl.system.set_next_task_start()
    return out


@pytest.mark.platforms("a2a3")
def test_mixed_local_boundaries(test_config: RunConfig, tmp_path: Path):
    output = Path(test_config.save_kernels_dir or tmp_path) / "mixed_boundary"
    compiled = mixed_boundary_probe.compile(
        config=replace(test_config, save_kernels_dir=str(output)),
        runtime=RuntimeKind.TENSORMAP_AND_RINGBUFFER,
    )
    compiled.load()
    if test_config.codegen_only:
        return
    x = (torch.arange(128 * 128) % 3 - 1).to(torch.float16).reshape(128, 128)
    w = (torch.arange(128 * 128) % 5 - 2).to(torch.float16).reshape(128, 128)
    bias = torch.ones((128, 128), dtype=torch.float32)
    out = torch.full_like(bias, float("nan"))
    hosts = [x, w, bias, out]
    with ChipWorker(config=test_config, runtime=compiled.runtime_name) as worker:
        tensors = [worker.alloc_tensor(t.shape, t.dtype) for t in hosts]
        try:
            handle = worker.register(compiled)
            for host, device in zip(hosts, tensors):
                worker.copy_to(device.data_ptr, host.data_ptr(), device.nbytes)
            handle(*tensors, config=test_config)
            worker.copy_from(out.data_ptr(), tensors[-1].data_ptr, tensors[-1].nbytes)
        finally:
            for tensor in reversed(tensors):
                worker.free_tensor(tensor)
    assert torch.equal(out, x.float() @ w.float() + bias)


@pl.jit
def mixed_chain_probe(
    x: pl.Tensor[[3200, 128], pl.FP16],
    w: pl.Tensor[[128, 256], pl.FP16],
    middle: pl.Out[pl.Tensor[[3200, 128], pl.FP16]],
    out: pl.Out[pl.Tensor[[3200, 128], pl.FP32]],
    ac_trace: pl.Tensor[[49, 8], pl.INT64],
    av_trace: pl.Tensor[[73, 8], pl.INT64],
    bc_trace: pl.Tensor[[49, 8], pl.INT64],
    bv_trace: pl.Tensor[[73, 8], pl.INT64],
    seed: pl.Scalar[pl.INT64],
    steps: pl.Scalar[pl.INT64],
):
    with pl.spmd(
        24,
        name_hint="mixed_producer",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        allow_early_resolve=True,
    ):
        block = pl.tile.get_block_idx()
        pl.system.wait_pre_task_end()
        # Read untouched guard rows to retain diagnostic pointers in both
        # outlined member signatures. Each sum is zero in the device test.
        guard = pl.tensor.read(ac_trace, [24 + block, 0]) + pl.tensor.read(av_trace, [48 + block, 0]) + 2
        row = block * 128 + pl.cast(guard, pl.INDEX)
        lhs = pl.load(x, [row, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        rhs = pl.load(w, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        acc = pl.matmul(
            pl.move(lhs, target_memory=pl.MemorySpace.Left), pl.move(rhs, target_memory=pl.MemorySpace.Right)
        )
        value = pl.add(acc, 0.0)
        for _ in pl.range(1024 + (block % 3) * 256):
            value = pl.add(value, 1.0)
        pl.store(pl.cast(value, pl.FP16), [block * 128, 0], middle)
        pl.system.set_next_task_start()
    with pl.spmd(
        24,
        name_hint="mixed_consumer",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        allow_early_resolve=True,
        sync_start=True,
    ):
        block = pl.tile.get_block_idx()
        state = seed + pl.cast(block, pl.INT64)
        for _ in pl.range(steps):
            state = (state * 17 + 3) % 65521
        if state % 2 == 0:
            col = pl.cast(0, pl.INDEX)
            pl.system.wait_pre_task_end()
            guard = pl.tensor.read(bc_trace, [24 + block, 0]) + pl.tensor.read(bv_trace, [48 + block, 0]) + 2
        else:
            col = pl.cast(128, pl.INDEX)
            pl.system.wait_pre_task_end()
            guard = pl.tensor.read(bc_trace, [24 + block, 0]) + pl.tensor.read(bv_trace, [48 + block, 0]) + 2
        lhs = pl.load(
            middle, [block * 128 + pl.cast(guard, pl.INDEX), 0], [128, 128], target_memory=pl.MemorySpace.Mat
        )
        rhs = pl.load(w, [0, col], [128, 128], target_memory=pl.MemorySpace.Mat)
        acc = pl.matmul(
            pl.move(lhs, target_memory=pl.MemorySpace.Left), pl.move(rhs, target_memory=pl.MemorySpace.Right)
        )
        value = pl.add(acc, pl.cast(state, pl.FP32))
        pl.store(value, [block * 128, 0], out)
        pl.system.set_next_task_start()
    return middle, out


def _exhaust_direct_notification_pair(explicit):
    # This pair is unused by the fixture's original scheduled body. Adding
    # eight distinct balanced events cannot consume its existing live flags.
    assert not re.search(r"pto\.(?:set|wait)_flag(?:_dyn)?[^\n]*PIPE_MTE2[^\n]*PIPE_MTE3", explicit)
    events = "\n".join(
        f"    pto.{op}_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID{event}>]"
        for event in range(8)
        for op in ("set", "wait")
    )
    explicit, count = re.subn(
        r"^[ \t]*(?:func\.)?call @pypto_set_next_task_start\(",
        lambda match: events + "\n" + match[0],
        explicit,
        flags=re.M,
    )
    assert count == 2, "both MIX physical member functions must receive the explicit flags"
    return explicit


@pytest.mark.platforms("a2a3")
@pytest.mark.parametrize("relay_events", [False, True], ids=["default_events", "relay_events"])
def test_mixed_members_enter_prefix_before_group_completion(
    test_config: RunConfig, tmp_path: Path, monkeypatch, relay_events: bool
):
    output = Path(test_config.save_kernels_dir or tmp_path) / (
        "mixed_chain_relay" if relay_events else "mixed_chain"
    )
    original_wrapper = pto_backend._generate_kernel_wrapper
    names = {
        "mixed_producer_aic": "ac_trace",
        "mixed_producer_aiv": "av_trace",
        "mixed_consumer_aic": "bc_trace",
        "mixed_consumer_aiv": "bv_trace",
    }
    traced = set()

    def instrument(func, *args, **kwargs):
        if func.name not in names:
            return original_wrapper(func, *args, **kwargs)
        trace = next(p for p in func.params if p.name_hint.startswith(names[func.name]))
        traced.add(func.name)
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
        code = original_wrapper(func, *args, **kwargs)
        if relay_events:
            assert f"// PYPTO_TASK_BOUNDARY_RELAY_V1 {func.name} " in code
            assert "#define PYPTO_TASK_BOUNDARY_RELAY_V1 1" in code
        if func.name.endswith("_aiv"):
            code = code.replace(
                "const int32_t __async_trace_block = get_block_idx(args);",
                "const int32_t __async_trace_block = 2 * get_block_idx(args) + get_sub_block_id(args);",
            )
        return code

    monkeypatch.setattr(pto_backend, "_generate_kernel_wrapper", instrument)
    synchronization = (
        explicit_sync(("mixed_producer", "mixed_consumer"), _exhaust_direct_notification_pair)
        if relay_events
        else nullcontext()
    )
    with synchronization:
        compiled = mixed_chain_probe.compile(
            config=replace(test_config, save_kernels_dir=str(output)),
            runtime=RuntimeKind.TENSORMAP_AND_RINGBUFFER,
        )
    assert traced == set(names)
    compiled.load()
    if test_config.codegen_only:
        return
    source = (torch.arange(3200 * 128) % 3 - 1).to(torch.float16).reshape(3200, 128)
    weight = torch.cat((torch.eye(128), 2 * torch.eye(128)), dim=1).half()
    hosts = [
        source.clone(),
        weight,
        torch.empty_like(source),
        torch.empty(source.shape),
        torch.empty((49, 8), dtype=torch.int64),
        torch.empty((73, 8), dtype=torch.int64),
        torch.empty((49, 8), dtype=torch.int64),
        torch.empty((73, 8), dtype=torch.int64),
    ]
    samples = []
    with ChipWorker(config=test_config, runtime=compiled.runtime_name) as worker:
        tensors = [worker.alloc_tensor(t.shape, t.dtype) for t in hosts]
        try:
            handle = worker.register(compiled)
            for repeat in range(2):
                hosts[0].copy_(source + repeat)
                for host in hosts[2:]:
                    host.fill_(-1 if host.dtype == torch.int64 else float("nan"))
                for host, device in zip(hosts, tensors):
                    worker.copy_to(device.data_ptr, host.data_ptr(), device.nbytes)
                seed, steps = 7 + repeat, 4
                handle(*tensors, seed, steps, config=test_config)
                for host, device in zip(hosts[2:], tensors[2:]):
                    worker.copy_from(host.data_ptr(), device.data_ptr, device.nbytes)
                for block in range(24):
                    region = slice(block * 128, (block + 1) * 128)
                    expected = (hosts[0][region].float() + 1024 + (block % 3) * 256).half()
                    state = seed + block
                    for _ in range(steps):
                        state = (state * 17 + 3) % 65521
                    assert torch.equal(hosts[2][region], expected)
                    assert torch.equal(hosts[3][region], expected.float() * (1 + state % 2) + state)
                assert torch.isnan(hosts[2][3072:]).all() and torch.isnan(hosts[3][3072:]).all()
                traces = [h[:n].tolist() for h, n in zip(hosts[4:], (24, 48, 24, 48))]
                for host, size in zip(hosts[4:], (24, 48, 24, 48)):
                    assert torch.equal(host[size:], torch.full_like(host[size:], -1))
                for rows in traces:
                    assert all(0 < r[1] <= r[2] <= r[5] <= r[7] <= r[3] and r[4] & 255 == 7 for r in rows)
                producer_end = max(r[3] for rows in traces[:2] for r in rows)
                early = [sum(r[1] < producer_end for r in rows) for rows in traces[2:]]
                samples.append(
                    {
                        "repeat": repeat,
                        "early_members_aic_aiv": early,
                        "notifications_by_member": [sum(bool(r[4] & 256) for r in rows) for rows in traces],
                        "traces": dict(
                            zip(("producer_aic", "producer_aiv", "consumer_aic", "consumer_aiv"), traces)
                        ),
                    }
                )
        finally:
            for tensor in reversed(tensors):
                worker.free_tensor(tensor)
    (output / "mixed_chain_results.json").write_text(json.dumps({"samples": samples}, indent=2) + "\n")
    assert all(all(count > 0 for count in s["early_members_aic_aiv"]) for s in samples), (
        "both physical member types must enter a checked prefix before the producer group completes"
    )
    if relay_events:
        assert all(any(s["notifications_by_member"][core::2]) for core in (0, 1) for s in samples), (
            "each physical core type must actually execute its relay callback"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
