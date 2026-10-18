# SPDX-License-Identifier: Apache-2.0
"""页检查快速路径与回退的设备回归：异构Q、dummy、尾块、空洞和短页表。

直接调用正式attention plan helper，独立CPU逐候选oracle检查indices/bias/mask。
同一作业内复用已有设备，不测性能，不修改完整CSA的输入/缓存或源码。
"""

import importlib.util
import json
from pathlib import Path

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.attention_plan_stages import (
    COMPRESSED_COLUMNS_DYN,
    ORIGINAL_COLUMNS_DYN,
    PLAN_CAP_DYN,
    QUERY_BOUNDS_DYN,
    REQUESTS_DYN,
    T_DYN,
    attention_slot_plan_stage,
    compressed_pages_all_valid,
)

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.phase_sync import SIGNAL_WORDS


@pl.jit.incore
def page_validity_core(
    table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    flags: pl.InOut[pl.Tensor[[REQUESTS_DYN, 16], pl.INT32]],
):
    """直接核对快速路径判定，防止错误回退仍因最终结果正确而漏检。

    单AIV依次覆盖不同尾列，之前的UB内容不能决定下一个合法前缀的结果。
    所有结果由同一Scalar写入，显式发布；每行其余15项必须保留填毒值。
    """
    # 这里混合Scalar控制和Vector Tile，必须显式声明incore边界；仅把循环
    # 放在spmd编排体会让当前outliner把Tile留在没有片上内存的Spmd函数。
    for request in pl.range(pl.tensor.dim(lengths, 0)):
        length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
        valid = compressed_pages_all_valid(table, request, length, 19)
        pl.write(flags, [request, 0], valid)
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return flags


@pl.jit
def page_validity_probe(
    table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    flags: pl.InOut[pl.Tensor[[REQUESTS_DYN, 16], pl.INT32]],
):
    with pl.spmd(1, name_hint="page_prefix_valid_columns"):
        flags = page_validity_core(table, lengths, flags)
    return flags


@pl.jit
def plan_probe(
    topk: pl.Tensor[[T_DYN, 512], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    original_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    compressed_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    indices: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 512], pl.INT32]],
    bias: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32]],
    mask: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32]],
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
):
    with pl.spmd(48, name_hint="attention_page_plan_regression"):
        worker = pl.tile.get_block_idx()
        indices, bias, mask, signals = attention_slot_plan_stage(
            topk,
            positions,
            bounds,
            lengths,
            original_table,
            compressed_table,
            indices,
            bias,
            mask,
            signals,
            pl.cast(1, pl.INT32),
            23,
            19,
            worker,
            48,
            True,
        )
    return indices, bias, mask


def validate(device, compile_options, output):
    import numpy as np
    import torch

    pipeline = next(p for p in Path(__file__).resolve().parents if p.name == "dynamic_full_pipeline_20261017")
    path = pipeline.parent / "dynamic_attention_plan_20261017/check_oracle.py"
    spec = importlib.util.spec_from_file_location("attention_plan_cpu_expected", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    # 含65/66列尾块、1024整块和1025跨块；前缀之后故意填非法页，不能参与归约。
    # hole仍必须返回0，短表也必须拒绝；不允许用“总是1”修复慢路径。
    lengths_test = np.array([4, 128, 132, 4096, 4224, 8198, 8448, 131072, 131078, 8198, 131078], np.int32)
    table_test = np.full((len(lengths_test), 1050), -1, np.int32)
    expected_flags = np.ones(len(lengths_test), np.int32)
    for row, length in enumerate(lengths_test):
        needed = (int(length) // 4 + 31) // 32
        table_test[row, :needed] = np.arange(needed, dtype=np.int32) % 19
    table_test[-2, 64] = -1
    table_test[-1, 1024] = 19
    expected_flags[-2:] = 0
    validity_checks = []
    for columns in (1050, 64):
        active_table = np.ascontiguousarray(table_test[:, :columns])
        flag_tensor = torch.full((len(lengths_test), 16), -777, dtype=torch.int32, device=device)
        page_validity_probe(
            torch.from_numpy(active_table).to(device),
            torch.from_numpy(lengths_test).to(device),
            flag_tensor,
            config=compile_options,
        )
        torch.npu.synchronize()
        actual_flags = flag_tensor.cpu().numpy()
        required = (lengths_test // 4 + 31) // 32
        expected = np.where(required <= columns, expected_flags, 0)
        np.testing.assert_array_equal(actual_flags[:, 0], expected)
        assert np.all(actual_flags[:, 1:] == -777)
        validity_checks.append(dict(columns=columns, flags=actual_flags[:, 0].tolist(), passed=True))
    q = np.array([0, 6, 9, 13], dtype=np.int32)
    lengths = np.array([131078, 135, 0], dtype=np.int32)
    positions = np.array([131072 + i for i in range(6)] + [132, 133, 134] + [-1] * 4, dtype=np.int64)
    topk = np.tile(np.arange(512, dtype=np.int32), (13, 1))
    topk[:, :8] = [-1, 0, 31, 32, 32767, 32768, 32769, 2147483647]
    original = np.arange(3 * 4097, dtype=np.int32).reshape(3, 4097) % 23
    compressed = np.arange(3 * 1025, dtype=np.int32).reshape(3, 1025) % 19
    # 第二请求实际只需两列；后面的-1不能让当前长度无条件走慢路径或误屏蔽合法候选。
    compressed[1, 2:] = -1
    records = []
    for name in ("valid_prefix_tail1", "holes_and_oob", "short_tables", "stale_positions"):
        ot, ct = original.copy(), compressed.copy()
        active_positions = positions.copy()
        if name == "holes_and_oob":
            ct[0, 1] = -1
            ct[1, 0] = 19
            ot[0, 4093] = -1
        elif name == "short_tables":
            ct = np.ascontiguousarray(ct[:, :1])
            ot = np.ascontiguousarray(ot[:, :2])
        elif name == "stale_positions":
            active_positions[4] += 2
            active_positions[8] = -1
        expected_i, expected_b, expected_m, owners = oracle.expected(topk, active_positions, q, lengths, ot, ct, 23, 19)
        geometry = np.tile(np.array([-1, 0], np.int32), (13, 1))
        for token, owner in enumerate(owners):
            if owner is not None:
                request, begin, end = owner
                if active_positions[token] == int(lengths[request]) - (end - begin) + token - begin:
                    geometry[token] = (request, active_positions[token])
        inputs = [torch.from_numpy(a).to(device) for a in (topk, active_positions, q, lengths, ot, ct)]
        indices = torch.full((24, 512), -777, dtype=torch.int32, device=device)
        bias = torch.full((24, 640), 777.0, dtype=torch.float32, device=device)
        mask = torch.full((24, 16), -777, dtype=torch.int32, device=device)
        # 输入Top-K已由Host准备完成；只给plan所需的阶段状态置当前代际，
        # 探针不模拟真实生产者、不测同步性能。普通页检查使用后置几何默认值。
        signals = torch.ones((SIGNAL_WORDS,), dtype=torch.int32, device=device)
        plan_probe(*inputs, indices, bias, mask, signals, config=compile_options)
        torch.npu.synchronize()
        actual_i, actual_b, actual_m = (t.cpu().numpy() for t in (indices, bias, mask))
        np.testing.assert_array_equal(actual_i[:13], expected_i)
        np.testing.assert_array_equal(actual_b[:13], expected_b)
        np.testing.assert_array_equal(actual_m[:13, :5], expected_m)
        np.testing.assert_array_equal(actual_m[:13, 5:7], geometry)
        assert np.all(actual_i[13:] == -777) and np.all(actual_b[13:] == 777)
        assert np.all(actual_m[13:] == -777) and np.all(actual_m[:13, 7:] == -777)
        records.append(
            {
                "case": name,
                "passed": True,
                "tokens": 13,
                "query_bounds": q.tolist(),
                "dummy_request": 2,
                "compressed_columns": ct.shape[1],
                "tail_and_mask_padding_preserved": True,
                "query_geometry_exact": True,
            }
        )
    report = {
        "passed": True, "performance_measured": False, "cases": records,
        "page_prefix_validity": validity_checks,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
