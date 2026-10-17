# SPDX-License-Identifier: Apache-2.0
"""主压缩后处理有界CPU审计：原数学表达式、请求块所有权及副作用边界。

不导入PyPTO，不编译或运行设备。模型记录实际允许的读写地址，覆盖相同T的
不同Q、非16对齐请求、C=0、无关闭组、陈旧position和非法slot等场景。数学
AST核对排除动态访存改写，只比原归约/归一化/RoPE表达式；设备数值仍待验证。
"""

import ast
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/compressor_output_stages.py"


def assignments(tree):
    return {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }


def expression(tree, name):
    value = copy.deepcopy(assignments(tree)[name])
    if name == "square_sum":
        # Tile接口显式scratch替代原Tensor接口自动scratch，不改变row_sum输入。
        assert isinstance(value, ast.Call) and ast.unparse(value.func) == "pl.row_sum"
        value.args = value.args[:1]
    if name in ("rope_ones", "rope_index") and isinstance(value, ast.Call):
        # Tile层索引常量与原Tensor层常量生成值相同，显式层级避免混合类型。
        operation = ast.unparse(value.func)
        if operation in ("pl.tile.full", "pl.tile.arange"):
            value.func = ast.parse(operation.replace("pl.tile.", "pl.")).body[0].value
    return ast.dump(value, include_attributes=False)


def source_math():
    source = ast.parse(SOURCE.read_text())
    reference = ast.parse((HERE / "reference_decode_compressor_ratio4.py").read_text())
    old = next(
        f for f in reference.body if isinstance(f, ast.FunctionDef) and f.name == "compressor_ratio4_cache_write"
    )
    new = next(f for f in source.body if isinstance(f, ast.FunctionDef))
    names = [
        "folded4",
        "folded2",
        "folded1",
        "square_sum",
        "variance",
        "inv_rms",
        "normed_chunk",
        "rope_normed",
        "rope_ones",
        "rope_index",
        "rope_index_f",
        "rope_col",
        "rope_dup_f",
        "rope_lane",
        "rope_swap_idx",
        "sin_signed",
        "rope_rot",
    ]
    equal = {name: expression(old, name) == expression(new, name) for name in names}
    assert all(equal.values()), equal
    new_assign = assignments(new)
    assert ast.unparse(new_assign["swapped"].func) == "pl.tile.gather"
    assert ast.unparse(new_assign["rope_row_offsets"]) == (
        "pl.reshape(pl.tile.muls(pl.tile.ci(0, [1, RMS_PAD_TILE], dtype=pl.INT32), ROPE_HEAD_DIM), [RMS_PAD_TILE, 1])"
    )
    # 原Tensor的逐行j^1与Tile绝对元素索引在全部16种有效行数下逐元素一致。
    for rows in range(1, 17):
        for row in range(rows):
            for column in range(64):
                absolute = row * 64 + column + 1 - 2 * (column % 2)
                assert divmod(absolute, 64) == (row, column ^ 1)
    equal["gather_index_equivalence_all_valid_rows"] = True
    imports = [ast.unparse(node) for node in source.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("tests" in line or "static" in line for line in imports)
    calls = [ast.unparse(node.func) for node in ast.walk(new) if isinstance(node, ast.Call)]
    assert not any("spmd" in call or "submit" in call or "task_dummy" in call for call in calls)
    assert not {"S", "B", "BS_PAD", "DECODE_SEQ", "safe_row"} & {
        node.id for node in ast.walk(source) if isinstance(node, ast.Name)
    }
    assert "pl.InOut[pl.Tensor[[NORMED_ROWS_DYN, HEAD_DIM], pl.FP32]]" in SOURCE.read_text()
    load_calls = [node for node in ast.walk(new) if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.load"]
    pooled = [node for node in load_calls if ast.unparse(node.args[0]) == "pooled_kv"]
    assert len(pooled) == 4 and all("valid_shape" in [kw.arg for kw in node.keywords] for node in pooled)
    parents = {}
    for node in ast.walk(new):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    reads = [
        node
        for node in ast.walk(new)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "pl.read"
        and ast.unparse(node.args[0]) == "compact_slots"
    ]
    assert len(reads) == 2
    for read in reads:
        guards = []
        node = read
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                guards.append(ast.unparse(node.test))
        assert any("compact_row >= 0" in g and "pl.tensor.dim(compact_slots, 0)" in g for g in guards)
        assert any("position == history_start + token - begin" in g and "% COMPRESS_RATIO == 0" in g for g in guards)
        assert "active != 0" in guards
    return equal


def bounds(q, lengths, request, capacity):
    if request + 1 < len(q) and lengths[request] > 0:
        begin, end = q[request : request + 2]
        if 0 <= begin < end <= capacity and end - begin <= lengths[request]:
            return begin, end, True
    return 0, 0, False


def model(queries, histories, workers, variant="normal"):
    q = [0]
    for count in queries:
        q.append(q[-1] + count)
    tokens = q[-1] + 3  # Native padding始终留三行，必须完全不读写。
    lengths = [h + s for h, s in zip(histories, queries)]
    if variant == "inactive":
        lengths[-1] = 0
    if variant == "short_Q":
        q = q[:-1]
    positions = [-991] * tokens
    offsets, prefix, valid_ranges = [], 0, []
    for request in range(len(lengths)):
        begin, end, active = bounds(q, lengths, request, tokens)
        offset = 0
        if active:
            h = lengths[request] - (end - begin)
            valid_ranges.append((request, begin, end))
            positions[begin:end] = list(range(h, h + end - begin))
            offset = prefix - h // 4 - 1
            prefix += lengths[request] // 4 - h // 4
        offsets.append(offset)
    capacity = 0 if variant == "zero_C" else prefix
    slots = [(row // 32, row % 32) for row in range(capacity)]
    cache_pages = max(1, (capacity + 31) // 32)
    if slots and variant == "bad_slots":
        slots = [(-1, 0), (cache_pages, 0), (0, -1), (0, 32)] * ((len(slots) + 3) // 4)
        slots = slots[:capacity]
    if valid_ranges and variant == "stale_positions":
        for _, begin, end in valid_ranges:
            positions[begin:end] = [127] * (end - begin)
    if variant == "negative_compact":
        offsets = [-10_000] * len(lengths)
    loads, stores, slot_reads, side_effects, blocks = [], [], [], [], []
    for worker in range(workers):
        for request in range(worker, len(lengths), workers):
            begin, end, active = bounds(q, lengths, request, tokens)
            if active:
                history = lengths[request] - (end - begin)
                for first in range(begin, end, 16):
                    rows = min(16, end - first)
                    assert begin <= first < first + rows <= end
                    blocks.append((request, first, rows, worker))
                    loads.extend(range(first, first + rows))
                    stores.extend(range(first, first + rows))
                    for token in range(first, first + rows):
                        position = positions[token]
                        if position == history + token - begin and (position + 1) % 4 == 0:
                            row = offsets[request] + (position + 1) // 4
                            if 0 <= row < capacity:
                                slot_reads.append(row)
                                page, offset = slots[row]
                                if 0 <= page < cache_pages and 0 <= offset < 32:
                                    side_effects.append((token, page * 32 + offset))
    active_rows = {token for _, begin, end in valid_ranges for token in range(begin, end)}
    assert len(loads) == len(set(loads)) and set(loads) == active_rows
    assert len(stores) == len(set(stores)) and set(stores) == active_rows
    assert all(0 <= row < capacity for row in slot_reads)
    assert len(side_effects) == len(set(side_effects))
    if variant in ("zero_C", "negative_compact", "bad_slots"):
        assert not side_effects
    if variant in ("zero_C", "negative_compact"):
        assert not slot_reads
    if variant == "normal":
        expected = [
            (token, offsets[request] + (positions[token] + 1) // 4)
            for request, begin, end in valid_ranges
            for token in range(begin, end)
            if (positions[token] + 1) % 4 == 0
        ]
        assert sorted(side_effects) == sorted(expected)
    return {
        "queries": queries,
        "histories": histories,
        "workers": workers,
        "variant": variant,
        "active_rows": len(active_rows),
        "blocks": blocks,
        "compact_capacity": capacity,
        "slot_reads": slot_reads,
        "side_effects": side_effects,
    }


def main():
    math = source_math()
    shapes = [
        ([2, 2, 2], [0, 6, 9]),
        ([4, 2], [6, 1]),
        ([1, 17, 33], [0, 3, 127]),
        ([6] * 28, [131072] * 28),
        ([6] * 32, [8192] * 32),
        ([0, 1, 0, 2], [0, 0, 4, 1]),
    ]
    variants = ["normal", "inactive", "short_Q", "zero_C", "bad_slots", "stale_positions", "negative_compact"]
    cases = [model(q, h, w, v) for q, h in shapes for w in (1, 8, 48) for v in variants]
    report = {
        "passed": True,
        "mathematical_expressions_equal": math,
        "cases": len(cases),
        "coverage": (
            "请求2/2/2与4/2、非对齐长请求、B28/B32、多wave、零query/关闭组/C、"
            "inactive、短Q、非法slot、陈旧position"
        ),
        "source_guard_before_slot_reads": True,
        "request_local_valid_shape": True,
        "device_execution": False,
        "device_numerical_equivalence": False,
        "examples": [c for c in cases if c["workers"] == 48 and len(c["queries"]) < 4 and c["variant"] == "normal"],
    }
    (HERE / "source_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "math_expressions": len(math), "cases": len(cases)}))


if __name__ == "__main__":
    main()
