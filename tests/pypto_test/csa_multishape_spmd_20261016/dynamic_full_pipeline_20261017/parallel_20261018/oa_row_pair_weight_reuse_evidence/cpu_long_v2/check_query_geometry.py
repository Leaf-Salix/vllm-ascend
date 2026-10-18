# SPDX-License-Identifier: Apache-2.0
"""解释正式计划/消费helper，检查几何复用及未写区；不模拟硬件缓存。

复用既有NumPy基础操作解释器，期望请求归属独立从Q/L区间求交生成。
候选计划与原关闭分支的indices/bias/块标志必须相同；第5/6列必须精确
保存实际INT32位置，尤其不能经FP32中转。消费开启分支只允许读plan，
不能在代码中仍扫描Q/L而把复用写成性能结论。此检查不代替真机验证。
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    package = here.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    oracle_path = here.parent / "dynamic_attention_plan_20261017/check_oracle.py"
    spec = importlib.util.spec_from_file_location("attention_plan_cpu_oracle", oracle_path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    accesses = []
    ns, names, allowed_rows, writes = oracle.source_namespace(package, accesses)
    cases = (
        ("same_T_222", [0, 2, 4, 6], [132, 135, 139], 6, False),
        ("same_T_42", [0, 4, 6], [134, 139], 6, False),
        ("mixed_dummy_tail", [0, 6, 9, 13], [131078, 135, 0], 13, False),
        ("stale_positions", [0, 3, 9], [134, 139], 11, True),
        ("overlap", [0, 6, 2, 8], [132, 133, 134], 9, False),
        ("all_dummy", [99, -3, 27], [0, -1], 9, False),
        ("fp32_position_boundary", [0, 2], [16777219], 2, False),
        ("int32_position_boundary", [0, 2], [2147483647], 2, False),
    )
    records = []
    for number, (label, q, lengths, tokens, stale) in enumerate(cases):
        q, lengths = np.asarray(q, np.int32), np.asarray(lengths, np.int32)
        topk = np.tile(np.arange(512, dtype=np.int32), (tokens, 1))
        positions = np.full(tokens, -77, np.int64)
        owners = oracle.geometry(q, lengths, tokens)
        for token, owner in enumerate(owners):
            if owner is not None:
                request, begin, end = owner
                positions[token] = int(lengths[request]) - (end - begin) + token - begin
        if stale:
            positions[0] += 3
            positions[7] = -1
        original = np.tile(np.arange(8, dtype=np.int32), (len(lengths), 1))
        compressed = original.copy()
        expected = np.tile(np.array([-1, 0], np.int32), (tokens, 1))
        for token, owner in enumerate(owners):
            if owner is not None:
                request, begin, end = owner
                if positions[token] == int(lengths[request]) - (end - begin) + token - begin:
                    expected[token] = (request, positions[token])
        allowed_rows.clear()
        allowed_rows[id(positions)] = {token for token, owner in enumerate(owners) if owner is not None}
        outputs = {}
        for enabled in (False, True):
            indices = np.full((tokens + 7, 512), -777, np.int32)
            bias = np.full((tokens + 7, 640), 777, np.float32)
            mask = np.full((tokens + 7, 16), -777, np.int32)
            # 数组在用例间释放后Python可复用id，不能让旧mask标签归到新indices。
            names.clear()
            names.update({id(mask): "plan", id(q): "bounds", id(lengths): "lengths", id(positions): "positions"})
            workers = (1, 3, 48)[number % 3]
            writes.clear()
            for worker in range(workers):
                ns["attention_slot_plan_stage"](
                    topk, positions, q, lengths, original, compressed,
                    indices, bias, mask, 8, 8, worker, workers, enabled,
                )
            stop = 7 if enabled else 5
            assert np.all(mask[:tokens, stop:] == -777) and np.all(mask[tokens:] == -777)
            assert np.all(indices[tokens:] == -777) and np.all(bias[tokens:] == 777)
            if enabled:
                np.testing.assert_array_equal(mask[:tokens, 5:7], expected)
                counts = np.zeros(mask.shape, np.int32)
                for name, offset, shape in writes:
                    if name == "plan":
                        counts[tuple(slice(start, start + size) for start, size in zip(offset, shape))] += 1
                assert np.all(counts[:tokens, :7] == 1) and np.all(counts[tokens:] == 0), (label, counts[:tokens, :7])
            outputs[enabled] = (indices, bias, mask[:, :5].copy())
            for token in range(tokens):
                accesses.clear()
                request, position, active = ns["attention_query_geometry"](
                    positions, q, lengths, mask, token, enabled
                )
                want = expected[token]
                assert (request, position, active) == (want[0], want[1], int(want[0] >= 0))
                if enabled:
                    assert len(accesses) == 2 and all(name == "plan" for name, *_ in accesses)
        for old, new in zip(outputs[False], outputs[True]):
            np.testing.assert_array_equal(old, new)
        records.append({"case": label, "tokens": tokens, "workers": workers, "passed": True})
    result = {
        "passed": True,
        "cases": records,
        "plan_math_identical": True,
        "one_writer_per_plan_element": True,
        "cached_consumer_only_two_plan_reads": True,
        "device_execution": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
