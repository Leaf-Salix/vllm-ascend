# SPDX-License-Identifier: Apache-2.0
"""执行正式源码的Python控制流，独立枚举关闭位置/目标cache行做CPU地址检查。

只去掉DSL导入、装饰器与类型；不另抄被测算法。不模拟硬件浮点、流水或
DCache，不能据此认定NPU正确。所有期望值逐位置枚举，包含dummy与尾块。
"""

import ast
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


class PlainPython(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node.returns = None
        for argument in node.args.args:
            argument.annotation = None
        return self.generic_visit(node)


def interpreted_source():
    reads = []

    def read(tensor, indices):
        assert len(indices) == tensor.ndim
        assert all(0 <= index < extent for index, extent in zip(indices, tensor.shape)), (indices, tensor.shape)
        reads.append((id(tensor), tuple(indices)))
        return tensor[tuple(indices)].item()

    def write(tensor, indices, value):
        assert all(0 <= index < extent for index, extent in zip(indices, tensor.shape)), (indices, tensor.shape)
        tensor[tuple(indices)] = value

    def gather_row(dest, source, dst, src, shape):
        for start, extent, capacity in zip(src, shape, source.shape):
            assert start >= 0 and start + extent <= capacity
        for start, extent, capacity in zip(dst, shape, dest.shape):
            assert start >= 0 and start + extent <= capacity
        result = dest.copy()
        result[dst[0]:dst[0] + shape[0], dst[1]:dst[1] + shape[1]] = source[
            src[0]:src[0] + shape[0], src[1]:src[1] + shape[1]
        ]
        return result

    language = SimpleNamespace(dynamic=lambda name: name, cast=lambda value, dtype: int(value),
                               INDEX="INDEX", INT32="INT32", FP32="FP32", BF16="BF16",
                               range=range, min=min, read=read, write=write, gather_row=gather_row,
                               full=lambda shape, dtype, value: np.full(shape, value, dtype=np.float32),
                               tensor=SimpleNamespace(read=read, dim=lambda value, axis: value.shape[axis]))
    language.tile = SimpleNamespace(full=language.full)
    namespace = {"pl": language}
    for name in ("request_geometry.py", "metadata_stages.py", "cache_stages.py"):
        path = SOURCE / name
        tree = ast.fix_missing_locations(PlainPython().visit(ast.parse(path.read_text())))
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace, reads


def metadata_cases():
    rng = random.Random(20261017)
    cases = [([2, 2, 2], [128, 256, 64]), ([4, 2], [128, 256]), ([1, 13, 9], [0, 18, 17]),
             ([3, 2, 6], [127, -1, 257]), ([1, 1], [0, 1]), ([6, 6], [-1, -1]), ([], [])]
    for _ in range(12):
        widths = [rng.randrange(1, 35) for _ in range(rng.randrange(1, 8))]
        histories = [rng.randrange(0, 1024) if rng.randrange(4) else -1 for _ in widths]
        cases.append((widths, histories))
    return cases


def check_metadata(helpers, reads):
    count = 0
    for widths, histories in metadata_cases():
        bounds = np.array([0, *np.cumsum(widths).tolist()], dtype=np.int32)
        lengths = np.array([width + history if history >= 0 else 0 for width, history in zip(widths, histories)],
                           dtype=np.int32)
        tokens = int(bounds[-1])
        positions = np.full(tokens, 127, dtype=np.int64)
        closed = []
        expected_offsets = []
        for request, (width, history) in enumerate(zip(widths, histories)):
            if history < 0:
                expected_offsets.append(0)
                continue
            expected_offsets.append(len(closed) - len(range(3, history, 4)) - 1)
            positions[bounds[request]:bounds[request + 1]] = np.arange(history, history + width)
            closed.extend((request, local, pos) for local, pos in enumerate(range(history, history + width))
                          if (pos + 1) % 4 == 0)
        offsets = np.full(len(widths), 777, dtype=np.int32)
        helpers["compact_offsets_stage"](bounds, lengths, offsets, tokens, 1)
        assert np.all(offsets == 777), "非owner不能写前缀"
        helpers["compact_offsets_stage"](bounds, lengths, offsets, tokens, 0)
        assert offsets.tolist() == expected_offsets
        cos = np.arange(len(closed) * 64, dtype=np.float32).reshape(len(closed), 64) + 10
        sin = -cos
        for request, width in enumerate(widths):
            for local in range(0, width, 16):
                rows = min(16, width - local)
                first = int(bounds[request]) + local
                reads.clear()
                actual = helpers["load_request_compact_rope_rows"](
                    cos, sin, positions, bounds, lengths, offsets, request, first, rows
                )
                expected = (np.ones((16, 64), np.float32), np.zeros((16, 64), np.float32))
                for compact, (owner, token_local, _) in enumerate(closed):
                    if owner == request and local <= token_local < local + rows:
                        expected[0][token_local - local] = cos[compact]
                        expected[1][token_local - local] = sin[compact]
                assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
                if histories[request] < 0:
                    assert not any(identity == id(positions) for identity, _ in reads), "dummy读取了残留position"
                count += 1
        # 非法compact偏移不能负下标读取；空表也必须返回单位旋转。
        if widths and histories[0] >= 0:
            rows = min(16, widths[0])
            for invalid_offset in (-10000, 10000):
                bad = offsets.copy()
                bad[0] = invalid_offset
                a, b = helpers["load_request_compact_rope_rows"](
                    cos, sin, positions, bounds, lengths, bad, 0, 0, rows
                )
                assert np.all(a == 1) and np.all(b == 0)
                count += 1
            a, b = helpers["load_request_compact_rope_rows"](
                cos[:0], sin[:0], positions, bounds, lengths, offsets, 0, 0, rows
            )
            assert np.all(a == 1) and np.all(b == 0)
            count += 1
    return count


def check_cache(helpers):
    count = 0
    for tokens in (0, 1, 6, 7, 8, 9, 24, 48, 96, 144, 168, 169, 192, 257):
        pages = (tokens + 31) // 32 + 1
        cache_rows = pages * 32
        kv = np.arange(tokens * 512, dtype=np.float32).reshape(tokens, 512)
        slots = np.array([(token // 32, token % 32) for token in range(tokens)], np.int32).reshape(tokens, 2)
        # -1 dummy、负offset、页越界和页内越界分别检查；其余合法槽保持唯一。
        for token in range(0, tokens, 11):
            slots[token] = ((-1, 0), (0, -1), (pages, 0), (0, 32))[(token // 11) % 4]
        expected = np.full((cache_rows, 512), -333, np.float32)
        for token, (page, offset) in enumerate(slots):
            if page in range(pages) and offset in range(32):
                expected[int(page) * 32 + int(offset)] = kv[token]
        for workers in (1, 3, 8, 24, 48):
            guarded = np.full((cache_rows + 2, 512), -333, np.float32)
            cache = guarded[1:-1]
            for worker in reversed(range(workers)):
                helpers["original_kv_writeback_stage"](kv, slots, cache, worker, workers)
            assert np.array_equal(cache, expected), (tokens, workers)
            assert np.all(guarded[[0, -1]] == -333), "改写cache保护区"
            count += 1
    return count


def main():
    helpers, reads = interpreted_source()
    report = {"source_ast_interpreted": True, "device_execution": False, "lowered_cpp_executed": False,
              "metadata_checks": check_metadata(helpers, reads), "cache_cases": check_cache(helpers), "passed": True}
    (HERE / "oracle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
