# SPDX-License-Identifier: Apache-2.0
"""临时普通 GM notify/wait 的 24 组 MIX 数据交接小例；不是 CSA 精度测试。"""

import argparse
import json
import os
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto.language as pl
from local_signal_adapter import csa_signal_notify, csa_signal_wait


@pl.jit.incore
def cube(a: pl.Tensor[[384, 16], pl.BF16], b: pl.Tensor[[16, 16], pl.BF16],
         signal: pl.InOut[pl.Tensor[[9216], pl.INT32]],
         scalar_data: pl.InOut[pl.Tensor[[72, 128], pl.INT32]],
         mm: pl.InOut[pl.Tensor[[384, 16], pl.FP32]],
         vec: pl.InOut[pl.Tensor[[48, 128], pl.FP32]],
         checked: pl.InOut[pl.Tensor[[72, 256], pl.FP32]]):
    group = pl.tile.get_block_idx()
    own = group * 128
    pl.system.cacheinvalid(signal, [1], [own])
    pl.system.fence()
    epoch = pl.cast(pl.read(signal, [own]), pl.INDEX) + 1
    left = pl.load(a, [group * 16, 0], [16, 16], target_memory=pl.Mem.Mat)
    right = pl.load(b, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    value = pl.matmul(left, right)
    mm = pl.store(value, [group * 16, 0], mm)
    # Scalar payload 跨 8 条缓存行，只有本 AIC 写自己的整行。
    for i in pl.range(128):
        scalar_data = pl.write(scalar_data, [group, i], pl.cast(epoch * 10000 + group * 100 + i, pl.INT32))
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    signal = csa_signal_notify(signal, own, pl.cast(epoch, pl.INT32))
    # 等另一个核组的 AIV1；前面已经通知，不形成等待环。
    peer = (group + 1) % 24
    signal = csa_signal_wait(signal, (24 + peer * 2 + 1) * 128, pl.cast(epoch, pl.INT32))
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    for i in pl.range(128):
        item = pl.read(vec, [peer * 2 + 1, i])
        checked = pl.write(checked, [group, i], item)
        checked = pl.write(checked, [group, i + 128], item)
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signal, scalar_data, mm, vec, checked


@pl.jit.incore
def vector(signal: pl.InOut[pl.Tensor[[9216], pl.INT32]],
           scalar_data: pl.InOut[pl.Tensor[[72, 128], pl.INT32]],
           mm: pl.InOut[pl.Tensor[[384, 16], pl.FP32]],
           vec: pl.InOut[pl.Tensor[[48, 128], pl.FP32]],
           checked: pl.InOut[pl.Tensor[[72, 256], pl.FP32]]):
    pl.func_attr({"split_aiv": True})
    group = pl.tile.get_block_idx()
    lane = pl.tile.get_subblock_idx()
    core = 24 + group * 2 + lane
    own = core * 128
    pl.system.cacheinvalid(signal, [1], [own])
    pl.system.fence()
    epoch = pl.cast(pl.read(signal, [own]), pl.INDEX) + 1
    peer = (group + 1) % 24
    # 同一个 AIC 信号有两个 AIV 消费者；等待不清除该信号。
    signal = csa_signal_wait(signal, peer * 128, pl.cast(epoch, pl.INT32))
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    tile = pl.load(mm, [peer * 16 + lane * 8, 0], [8, 16])
    incremented = pl.add(tile, 1.0)
    flat = pl.reshape(incremented, [1, 128])
    vec = pl.store(flat, [group * 2 + lane, 0], vec)
    scalar_tile = pl.load(scalar_data, [peer, 0], [1, 128])
    checked = pl.store(pl.cast(scalar_tile, pl.FP32), [core, 0], checked)
    checked = pl.store(flat, [core, 128], checked)
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    signal = csa_signal_notify(signal, own, pl.cast(epoch, pl.INT32))
    if lane == 1:
        # 额外覆盖不同核组的 AIV0 → AIV1 数据交接。
        vector_peer = (group + 2) % 24
        signal = csa_signal_wait(signal, (24 + vector_peer * 2) * 128, pl.cast(epoch, pl.INT32))
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
        other = pl.load(vec, [vector_peer * 2, 0], [1, 128])
        checked = pl.store(other, [core, 128], checked)
    return signal, scalar_data, mm, vec, checked


@pl.jit.inline
def group_wrapper(a: pl.Tensor[[384, 16], pl.BF16], b: pl.Tensor[[16, 16], pl.BF16],
                  signal: pl.InOut[pl.Tensor[[9216], pl.INT32]],
                  scalar_data: pl.InOut[pl.Tensor[[72, 128], pl.INT32]],
                  mm: pl.InOut[pl.Tensor[[384, 16], pl.FP32]],
                  vec: pl.InOut[pl.Tensor[[48, 128], pl.FP32]],
                  checked: pl.InOut[pl.Tensor[[72, 256], pl.FP32]]):
    with pl.cluster(name_hint="local_signal_payload"):
        signal, scalar_data, mm, vec, checked = cube(a, b, signal, scalar_data, mm, vec, checked)
        signal, scalar_data, mm, vec, checked = vector(signal, scalar_data, mm, vec, checked)
    return signal, scalar_data, mm, vec, checked


@pl.jit
def entry(a: pl.Tensor[[384, 16], pl.BF16], b: pl.Tensor[[16, 16], pl.BF16],
          signal: pl.InOut[pl.Tensor[[9216], pl.INT32]],
          scalar_data: pl.InOut[pl.Tensor[[72, 128], pl.INT32]],
          mm: pl.InOut[pl.Tensor[[384, 16], pl.FP32]],
          vec: pl.InOut[pl.Tensor[[48, 128], pl.FP32]],
          checked: pl.InOut[pl.Tensor[[72, 256], pl.FP32]]):
    with pl.spmd(24, sync_start=True):
        signal, scalar_data, mm, vec, checked = group_wrapper(a, b, signal, scalar_data, mm, vec, checked)
    return signal, scalar_data, mm, vec, checked


def main():
    import torch
    import torch_npu
    import pypto.torch
    from pypto.runtime import CompileOptions
    from test_static_jit_device import verify_isolation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device_id = int(os.environ["TASK_DEVICE"])
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "scope": "普通 GM 信号与 Scalar/MTE payload；非 CSA", "launches": []}
    try:
        report["environment"] = verify_isolation()
        torch.npu.set_device(device_id)
        pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer")
        device = torch.device(f"npu:{device_id}")
        stream = torch.npu.Stream(device=device)
        allocations = []

        def guarded(shape, dtype, initial):
            count = 1
            for size in shape:
                count *= size
            backing = torch.full((count + 256,), -12345, dtype=dtype, device=device)
            view = backing[128:-128].view(shape)
            view.fill_(initial)
            assert view.data_ptr() % 512 == 0
            allocations.append(backing)
            return view

        config = CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"))
        with torch.npu.stream(stream):
            signal = guarded((9216,), torch.int32, 0)
            scalar_data = guarded((72, 128), torch.int32, -999)
            mm = guarded((384, 16), torch.float32, -999)
            vec = guarded((48, 128), torch.float32, -999)
            checked = guarded((72, 256), torch.float32, -999)
            a = torch.empty((384, 16), dtype=torch.bfloat16, device=device)
            b = torch.eye(16, dtype=torch.bfloat16).to(device)
            for epoch in (1, 2, 3):
                a_cpu = (torch.arange(24) + epoch * 10).repeat_interleave(16)[:, None].expand(384, 16).to(torch.bfloat16)
                a.copy_(a_cpu)
                returned = entry(a, b, signal, scalar_data, mm, vec, checked, config=config)
                assert all(x is y for x, y in zip(returned, (signal, scalar_data, mm, vec, checked)))
                stream.synchronize()
                expected = torch.empty((72, 256), dtype=torch.float32)
                expected_vec = torch.empty((48, 128), dtype=torch.float32)
                for group in range(24):
                    expected[group].fill_((group + 2) % 24 + epoch * 10 + 1)
                    for lane in (0, 1):
                        core = 24 + group * 2 + lane
                        expected[core, :128] = epoch * 10000 + ((group + 1) % 24) * 100 + torch.arange(128)
                        expected[core, 128:].fill_((group + (1 if lane == 0 else 3)) % 24 + epoch * 10 + 1)
                        expected_vec[group * 2 + lane].fill_((group + 1) % 24 + epoch * 10 + 1)
                torch.testing.assert_close(checked.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(vec.cpu(), expected_vec, rtol=0, atol=0)
                torch.testing.assert_close(mm.cpu(), a_cpu.float(), rtol=0, atol=0)
                expected_signal = torch.zeros(9216, dtype=torch.int32)
                expected_signal[::128] = epoch
                torch.testing.assert_close(signal.cpu(), expected_signal, rtol=0, atol=0)
                for allocation in allocations:
                    assert bool((allocation[:128].cpu() == -12345).all())
                    assert bool((allocation[-128:].cpu() == -12345).all())
                report["launches"].append({"epoch": epoch, "signals_payload_and_guards_passed": True})
        report.update(passed=True, device_execution=True, groups=24, aic=24, aiv=48,
                      shared_install_modified=False, full_csa_validated=False)
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
