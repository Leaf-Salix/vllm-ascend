# SPDX-License-Identifier: Apache-2.0
"""设备验证本轮组描述符：单写者发布、不同AIV读取、同址图内metadata更新。

仅执行正式分组helper及正式notify/wait，不重新实现CSA数学。独立CPU枚举
请求/位置生成期望描述符；G2/G6各在同一图内依次使用A→B→A→全dummy→A。
所有metadata地址保持固定，只有内容改变。输出每组占64B，避免验证脚本自身
的多Scalar写者共享缓存行；signal沿用每核512B槽。此探针不测性能。
"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
import torch

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_group_stages import (
    OFFSET_ROWS_DYN,
    QUERY_BOUNDS_DYN,
    REQUESTS_DYN,
    TOKENS_DYN,
    indexer_decode_group,
    indexer_group_plan_stage,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.phase_sync import (
    SIGNAL_WORDS,
    begin_invocation,
    publish_phase,
    wait_producer,
)

PROBE_WORKERS = 8
PROBE_PLANNER = 7


@pl.jit.incore
def group_probe_core(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    plan: pl.InOut[pl.Tensor[[8], pl.INT32]],
    output: pl.InOut[pl.Tensor[[TOKENS_DYN, 16], pl.INT32]],
    signal: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    GROUP_SIZE: pl.constexpr,
):
    worker = pl.tile.get_block_idx()
    signal, generation = begin_invocation(signal, worker)
    if worker == PROBE_PLANNER:
        offsets, plan = indexer_group_plan_stage(
            bounds, lengths, positions, offsets, plan, pl.tensor.dim(positions, 0), 0, GROUP_SIZE, True, 1
        )
        signal = publish_phase(signal, 1, worker, PROBE_WORKERS, generation)
    signal = wait_producer(signal, 1, PROBE_PLANNER, PROBE_WORKERS, generation)
    groups = pl.cast(pl.read(plan, [0]), pl.INDEX)
    for group in pl.range(worker, groups, PROBE_WORKERS):
        request, first, live, visible = indexer_decode_group(bounds, lengths, positions, offsets, group, GROUP_SIZE)
        row = pl.tile.full([1, 16], dtype=pl.INT32, value=37)
        row = pl.tile.write(row, [0, 0], pl.cast(request, pl.INT32))
        row = pl.tile.write(row, [0, 1], pl.cast(first, pl.INT32))
        row = pl.tile.write(row, [0, 2], pl.cast(live, pl.INT32))
        row = pl.tile.write(row, [0, 3], pl.cast(visible, pl.INT32))
        output = pl.store(row, [group, 0], output)
    return offsets, plan, output, signal


@pl.jit
def group_probe(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    plan: pl.InOut[pl.Tensor[[8], pl.INT32]],
    output: pl.InOut[pl.Tensor[[TOKENS_DYN, 16], pl.INT32]],
    signal: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    GROUP_SIZE: pl.constexpr,
):
    with pl.spmd(PROBE_WORKERS, sync_start=True, name_hint="dynamic_group_metadata_probe"):
        offsets, plan, output, signal = group_probe_core(
            bounds, lengths, positions, offsets, plan, output, signal, GROUP_SIZE
        )
    return offsets, plan, output, signal


def expected(q, lengths, positions, group_size):
    descriptors, prefix = [], []
    for request, length in enumerate(lengths.tolist()):
        prefix.append(len(descriptors))
        begin, end = map(int, q[request : request + 2])
        if length <= 0 or not (0 <= begin < end <= len(positions)) or end - begin > length:
            continue
        history = length - end + begin
        queries = list(range(begin, end))
        for first_index in range(0, len(queries), group_size):
            chunk = queries[first_index : first_index + group_size]
            visible = [
                min(length // 4, (int(positions[t]) + 1) // 4, 262144)
                for t in chunk
                if int(positions[t]) == history + t - begin
            ]
            descriptors.append([request, chunk[0], len(chunk), max([0, *visible])])
    prefix.append(len(descriptors))
    result = torch.full((len(positions), 16), 37, dtype=torch.int32)
    if descriptors:
        result[: len(descriptors), :4] = torch.tensor(descriptors, dtype=torch.int32)
    return result, prefix


def validate(device, config, output_path):
    cases = []
    for name, bounds, lengths in (
        ("A", [0, 6, 9, 13], [131078, 135, 204]),
        ("B", [0, 2, 6, 13], [130, 2, 120]),
        ("dummy", [0, 2, 6, 13], [0, 0, 0]),
    ):
        q = torch.tensor(bounds, dtype=torch.int32)
        lens = torch.tensor(lengths, dtype=torch.int32)
        pos = torch.full((13,), -1, dtype=torch.int64)
        for r, length in enumerate(lengths):
            begin, end = bounds[r : r + 2]
            if length >= end - begin:
                pos[begin:end] = torch.arange(length - end + begin, length, dtype=torch.int64)
        pos[8] = -13  # 合法请求内部的陈旧位置，不允许扩大该组visible。
        cases.append((name, q, lens, pos))
    recorded = []
    for group_size in (2, 6):
        q, lens, pos = (x.to(device) for x in cases[0][1:])
        desc_base = 16
        guarded_offsets = torch.full((desc_base + 4 * 13 + 32,), -77, dtype=torch.int32, device=device)
        offsets = guarded_offsets[16:-16]
        plan = torch.full((8,), -77, dtype=torch.int32, device=device)
        guarded_output = torch.full((15, 16), 37, dtype=torch.int32, device=device)
        out = guarded_output[1:-1]
        signal = torch.zeros((2 * PROBE_WORKERS * 128,), dtype=torch.int32, device=device)
        tensors = (q, lens, pos, offsets, plan, out, signal)
        addresses = tuple(t.data_ptr() for t in tensors)
        group_probe(*tensors, GROUP_SIZE=group_size, config=config)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            group_probe(*tensors, GROUP_SIZE=group_size, config=config)
        for which in (0, 1, 0, 2, 0):
            name, cq, cl, cp = cases[which]
            q.copy_(cq)
            lens.copy_(cl)
            pos.copy_(cp)
            out.fill_(37)
            graph.replay()
            torch.npu.synchronize()
            wanted, prefix = expected(cq, cl, cp, group_size)
            assert tuple(t.data_ptr() for t in tensors) == addresses
            assert torch.equal(out.cpu(), wanted), (group_size, name)
            assert offsets[:4].cpu().tolist() == prefix
            assert int(plan[0].cpu()) == prefix[-1]
            assert bool((guarded_offsets[:16].cpu() == -77).all())
            assert bool((guarded_offsets[-16:].cpu() == -77).all())
            assert bool((guarded_output[[0, -1]].cpu() == 37).all())
            recorded.append({"group_size": group_size, "case": name, "groups": prefix[-1], "passed": True})
        generations = signal.cpu().view(2, PROBE_WORKERS, 128)[0, :, 0]
        assert bool((generations == generations[0]).all())
    result = {"passed": True, "same_addresses": True, "cases": recorded, "performance_measured": False}
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    from pypto.runtime import CompileOptions

    parser = argparse.ArgumentParser(description="仅CPU编译描述符图重放probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    group_probe._resolve_kernel_artifact(
        (),
        {"GROUP_SIZE": 6, "config": CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"))},
        allow_signature_mode=True,
    )
    (args.output / "result.json").write_text('{"compiled": true, "device_execution": false}\n')
