# SPDX-License-Identifier: Apache-2.0
"""正式scale页计划与提交的设备回归：同页RMW、页owner碰撞、无效slot及同址图。

只验证地址、FP16 scale更新和缓存发布；FP32量化值由测试输入提供，不另写量化。
每次重放用独立CPU枚举关闭token生成预期cache，比较完整allocation的所有字节。
普通GM信号使用正式发布/等待实现，不用torch事件冒充核内跨AIV可见性。
"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
import torch

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_cache_stages import (
    CACHE_BYTES_DYN,
    CACHE_PAGES_DYN,
    SCALE_ROWS_DYN,
    SLOT_ROWS_DYN,
    T_DYN,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_group_stages import OFFSET_ROWS_DYN
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_scale_plan_stages import (
    indexer_scale_page_commit_stage,
    indexer_scale_page_plan_stage,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.metadata_stages import compact_offsets_and_count_stage
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.phase_sync import (
    SIGNAL_WORDS,
    begin_invocation,
    publish_phase,
    wait_producer,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN

SCALE_TEST_WORKERS = 8
SCALE_TEST_PLANNER = 7


@pl.jit.incore
def scale_page_probe_core(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    offsets: pl.InOut[pl.Tensor[[REQUESTS_DYN], pl.INT32]],
    count: pl.InOut[pl.Tensor[[1], pl.INT32]],
    slots: pl.Tensor[[SLOT_ROWS_DYN, 2], pl.INT32],
    scales: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    workspace: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    cache: pl.InOut[pl.Tensor[[CACHE_PAGES_DYN, CACHE_BYTES_DYN], pl.INT8]],
    signal: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
):
    worker = pl.tile.get_block_idx()
    signal, generation = begin_invocation(signal, worker)
    tokens = pl.tensor.dim(positions, 0)
    if worker == SCALE_TEST_PLANNER:
        offsets, count = compact_offsets_and_count_stage(bounds, lengths, offsets, count, tokens, 0)
        workspace = indexer_scale_page_plan_stage(
            bounds, lengths, positions, offsets, slots, cache, workspace, pl.tensor.dim(scales, 0), SCALE_TEST_WORKERS
        )
        signal = publish_phase(signal, 1, worker, SCALE_TEST_WORKERS, generation)
    signal = wait_producer(signal, 1, SCALE_TEST_PLANNER, SCALE_TEST_WORKERS, generation)
    cache = indexer_scale_page_commit_stage(
        scales, workspace, cache, tokens, pl.tensor.dim(lengths, 0), worker, SCALE_TEST_WORKERS
    )
    signal = publish_phase(signal, 2, worker, SCALE_TEST_WORKERS, generation)
    return offsets, count, workspace, cache, signal


@pl.jit
def scale_page_probe(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    offsets: pl.InOut[pl.Tensor[[REQUESTS_DYN], pl.INT32]],
    count: pl.InOut[pl.Tensor[[1], pl.INT32]],
    slots: pl.Tensor[[SLOT_ROWS_DYN, 2], pl.INT32],
    scales: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    workspace: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    cache: pl.InOut[pl.Tensor[[CACHE_PAGES_DYN, CACHE_BYTES_DYN], pl.INT8]],
    signal: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
):
    with pl.spmd(SCALE_TEST_WORKERS, sync_start=True, name_hint="native_scale_page_owners_probe"):
        offsets, count, workspace, cache, signal = scale_page_probe_core(
            bounds, lengths, positions, offsets, count, slots, scales, workspace, cache, signal
        )
    return offsets, count, workspace, cache, signal


def expected_cache(initial, q, lengths, positions, slots, scales):
    expected = initial.clone()
    compact = 0
    for request, length in enumerate(lengths.tolist()):
        begin, end = map(int, q[request : request + 2])
        if length <= 0 or not 0 <= begin < end <= len(positions) or end - begin > length:
            continue
        history = length - end + begin
        # 直接枚举token的绝对位置是否关闭C4，不复写helper的first_token公式。
        for token in range(begin, end):
            position = history + token - begin
            if (position + 1) % 4:
                continue
            page, offset = map(int, slots[compact])
            if int(positions[token]) == position and 0 <= page < len(initial) and 0 <= offset < 32:
                value = scales[compact].to(torch.float16).view(torch.int8)
                expected[page, 4096 + offset * 2 : 4098 + offset * 2] = value
            compact += 1
    return expected


def validate(device, config, output_path):
    q_cpu = torch.tensor([0, 7, 12, 17], dtype=torch.int32)
    lengths_cpu = torch.tensor([136, 137, 204], dtype=torch.int32)
    pos_cpu = torch.cat((torch.arange(129, 136), torch.arange(132, 137), torch.arange(199, 204))).to(torch.int64)
    scales_cpu = torch.linspace(0.00011, 0.0027, 17, dtype=torch.float32).view(17, 1)
    # 8B dtype对齐之外，cache按64B页布局；保护前后整页，比较全部历史key/scale字节。
    initial = torch.arange(62 * 4160, dtype=torch.int64).to(torch.int8).view(62, 4160)
    guarded_cache = initial.to(device)
    cache = guarded_cache[1:-1]
    q, lengths, pos, scales = (x.to(device) for x in (q_cpu, lengths_cpu, pos_cpu, scales_cpu))
    offsets = torch.empty((3,), dtype=torch.int32, device=device)
    count = torch.empty((1,), dtype=torch.int32, device=device)
    slots = torch.full((17, 2), -1, dtype=torch.int32, device=device)
    group_end = 16 + 4 * 17
    base = (group_end + 15) // 16 * 16
    guarded_workspace = torch.full((base + SCALE_TEST_WORKERS + 4 * 17 + 32,), -71, dtype=torch.int32, device=device)
    workspace = guarded_workspace[16:-16]
    signal = torch.zeros((3 * SCALE_TEST_WORKERS * 128,), dtype=torch.int32, device=device)
    tensors = (q, lengths, pos, offsets, count, slots, scales, workspace, cache, signal)
    addresses = tuple(t.data_ptr() for t in tensors)
    # 预编译/预热时没有合法slot；不把预热的cache状态作为预期。
    scale_page_probe(*tensors, config=config)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        scale_page_probe(*tensors, config=config)
    reports = []
    for name in ("shared_page", "distinct_pages", "shared_page", "stale_and_invalid", "dummy", "shared_page"):
        cp = pos_cpu.clone()
        cl = lengths_cpu.clone()
        slot_cpu = torch.full((17, 2), -1, dtype=torch.int32)
        slot_cpu[:5, 0] = 5
        slot_cpu[:5, 1] = torch.tensor([0, 1, 31, 2, 3], dtype=torch.int32)
        if name == "distinct_pages":
            slot_cpu[:5, 0] = torch.tensor([1, 9, 2, 10, 3], dtype=torch.int32)
        elif name == "stale_and_invalid":
            cp[2] = -1
            slot_cpu[1, 0] = -1
            slot_cpu[2, 0] = 60
            slot_cpu[3, 1] = 32
        elif name == "dummy":
            cl.zero_()
        lengths.copy_(cl)
        pos.copy_(cp)
        slots.copy_(slot_cpu)
        guarded_cache.copy_(initial)
        graph.replay()
        torch.npu.synchronize()
        expected = initial.clone()
        expected[1:-1] = expected_cache(initial[1:-1], q_cpu, cl, cp, slot_cpu, scales_cpu)
        assert tuple(t.data_ptr() for t in tensors) == addresses
        assert torch.equal(guarded_cache.cpu(), expected), name
        assert bool((guarded_workspace[:16].cpu() == -71).all() and (guarded_workspace[-16:].cpu() == -71).all())
        reports.append({"case": name, "passed": True, "full_cache_and_guards_exact": True})
    result = {"passed": True, "same_addresses": True, "cases": reports, "performance_measured": False}
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    from pypto.runtime import CompileOptions

    parser = argparse.ArgumentParser(description="仅CPU编译scale页所有权probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scale_page_probe._resolve_kernel_artifact(
        (),
        {"config": CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"))},
        allow_signature_mode=True,
    )
    (args.output / "result.json").write_text('{"compiled": true, "device_execution": false}\n')
