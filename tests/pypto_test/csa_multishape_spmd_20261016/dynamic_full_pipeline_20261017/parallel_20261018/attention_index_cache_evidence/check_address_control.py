# SPDX-License-Identifier: Apache-2.0
"""执行候选源码的地址分支，独立核对合法行与禁止访问的坏地址。

只模拟整数控制、UB索引计算和DMA地址，不模拟浮点数学、设备缓存或流水。
从实际函数AST抽取分支，避免另写一份候选实现作为测试对象。
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def check():
    here = Path(__file__).resolve().parent
    source = here / "source_candidate/attention_stream_stages.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "attention_qk_pv_vector_lane_stage"
    )
    branches = [
        node for node in ast.walk(function)
        if isinstance(node, ast.If) and "CACHE_INDEX_PAGES" in ast.unparse(node.test)
    ]
    assert len(branches) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=branches, type_ignores=[])), str(source), "exec")
    accesses = []

    def load(tensor, offset, shape, valid_shape=None):
        valid = shape if valid_shape is None else valid_shape
        r, c = offset
        rows, cols = valid
        assert 0 <= r <= tensor.shape[0] - rows
        assert 0 <= c <= tensor.shape[1] - cols
        tile = np.full(shape, np.iinfo(np.int32).min, dtype=np.int32)
        tile[:rows, :cols] = tensor[r:r + rows, c:c + cols]
        return tile

    def gather(table, indices, destination):
        assert np.all(indices >= 0) and np.all(indices < actual_columns)
        destination[:] = table[0, indices]
        return destination

    def gather_row(destination, tensor, target, origin, shape):
        assert shape == [1, 512]
        assert target[1] == 0 and origin[1] == 0
        assert 0 <= origin[0] < tensor.shape[0]
        destination[target[0]] = origin[0]
        accesses.append(origin[0])
        return destination

    def read(tensor, position):
        return int(tensor[tuple(position)])

    tile = SimpleNamespace(
        maximums=np.maximum, minimums=np.minimum,
        shrs=np.right_shift, shls=np.left_shift,
        sub=np.subtract, add=np.add, gather=gather, read=read,
    )
    pl = SimpleNamespace(
        tile=tile, INDEX=int, INT32=np.int32,
        tensor=SimpleNamespace(dim=lambda x, axis: x.shape[axis]),
        load=load, cast=lambda value, dtype: int(value),
        create_tile=lambda shape, dtype: np.zeros(shape, dtype=dtype),
        gather_row=gather_row, range=range, read=read,
    )
    random = np.random.default_rng(20261018)
    cases = 0
    enabled_cases = 0
    for actual_columns in (0, 1, 2, 31, 66, 1026, 2048, 2049, 8193):
        for physical_rows in (0, 31, 32, 33, 4096, (1 << 31) - 1, (1 << 31) + 32):
            page_limit = physical_rows // 32
            table = random.integers(0, max(page_limit, 1), size=(2, actual_columns), dtype=np.int32)
            if actual_columns:
                table[1, 0] = -1
            if actual_columns > 1:
                table[1, 1] = max(page_limit - 1, 0)
            if actual_columns > 2:
                table[1, 2] = page_limit
            if actual_columns > 3:
                table[1, 3] = np.iinfo(np.int32).max
            if actual_columns > 4:
                table[1, 4] = np.iinfo(np.int32).min
            indices = random.integers(-64, max(actual_columns * 32 + 64, 65), size=(1, 512), dtype=np.int32)
            boundary = [-1, 0, 31, 32, 63, 64, 96, 128, actual_columns * 32 - 1,
                        actual_columns * 32, np.iinfo(np.int32).max, np.iinfo(np.int32).min]
            for offset in range(0, 512, 64):
                indices[0, offset:offset + len(boundary)] = boundary
            for enabled in (False, True):
                for block in (1, 2, 3, 4):
                    for lane in (0, 64):
                        first = (block - 1) * 128 + lane
                        expected = np.full(64, -1, dtype=np.int64)
                        for row, raw in enumerate(indices[0, first:first + 64]):
                            index = int(raw)
                            if 0 <= index < actual_columns * 32:
                                page = int(table[1, index // 32])
                                if 0 <= page < page_limit:
                                    expected[row] = page * 32 + index % 32
                        scope = {
                            "pl": pl, "CACHE_INDEX_PAGES": enabled,
                            "INDEX_PAGE_CACHE_COLS": 2048, "INDEX_ROW_INT32_MAX": (1 << 31) - 1,
                            "ATTN_K_TILE": 128, "CMP_TOPK": 512, "BLOCK_SIZE": 32, "HEAD_DIM": 512,
                            "qk_page_columns": actual_columns, "qk_physical_rows": physical_rows,
                            "qk_sb": block, "qk_lane_kv": lane, "qk_t": 0, "qk_b": 1,
                            "cmp_block_table": table, "cmp_sparse_indices": indices,
                            "cmp_kv_flat": SimpleNamespace(shape=(physical_rows, 512)),
                            "qk_kv_half": np.full(64, -1, dtype=np.int64),
                        }
                        accesses.clear()
                        exec(code, scope)
                        np.testing.assert_array_equal(scope["qk_kv_half"], expected)
                        assert accesses == expected[expected >= 0].tolist()
                        cases += 1
                        enabled_cases += int(enabled and 0 < actual_columns <= 2048 and physical_rows <= (1 << 31) - 1)
    report = {
        "passed": True, "source": str(source), "cases": cases,
        "vector_branch_cases": enabled_cases,
        "scope": "实际源码的整数地址分支与独立逐行oracle；不执行NPU数学/缓存/FFTS",
        "edge_cases": ["负索引", "逻辑索引列越界", "负页", "物理页上界", "页号INT32极值",
                       "非整页物理容量", "空页表", "2048列预算边界", "INT32物理行边界", "四压缩块及双lane"],
    }
    (here / "ADDRESS_CONTROL_CHECK.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    check()
