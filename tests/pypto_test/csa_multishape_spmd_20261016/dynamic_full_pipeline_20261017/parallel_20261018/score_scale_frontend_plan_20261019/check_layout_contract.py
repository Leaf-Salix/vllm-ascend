# SPDX-License-Identifier: Apache-2.0
"""只读 CPU 合同检查：容量分区、正逆序/碎片判定及全部叶地址范围；不是设备精度证据。"""

import ast
import json
import random
import struct
from pathlib import Path


def fp32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def proof(table, pages, physical_pages):
    """模拟候选的整数/FP32域，和直接整数逐项判断这一独立 oracle 比较。"""
    if not 0 < pages <= len(table) or not 0 < physical_pages <= 2**24:
        return 0, 0
    first, last = table[0], table[pages - 1]
    direction = -1 if last < first else 1
    expected_last = first + direction * (pages - 1)
    if last != expected_last or min(first, expected_last) < 0 or max(first, expected_last) >= physical_pages:
        return 0, 0
    for block in range(0, pages, 1024):
        for lane in range(min(1024, pages - block)):
            expected = fp32(fp32(direction * lane) + fp32(first + direction * block))
            if abs(fp32(table[block + lane]) - expected) != 0:
                return 0, 0
    return first, direction


def main():
    here = Path(__file__).resolve().parent
    source = ast.parse((here / "source_candidate/binding.py").read_text())
    expression = next(
        value.elts[0].elts[0]
        for node in ast.walk(source) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "group_offsets"
    )
    capacity = compile(ast.Expression(expression), "<binding group_offsets capacity>", "eval")
    pad = lambda x, n: (x + n - 1) // n * n
    checked_layouts = 0
    for b in (1, 2, 4, 8, 15, 16, 24, 28, 32, 47, 48, 64):
        for t in (b, b * 5, b * 6, b * 7 + 3, b * 9 + 1):
            old_end = pad(pad(b + 1, 16) + 4 * t, 16) + 48 + 4 * t
            got = eval(capacity, {"_pad": pad}, {"b": b, "t": t})
            assert got == old_end + 2 * b
            writes = [old_end + 2 * request + field for request in range(b) for field in (0, 1)]
            assert len(set(writes)) == 2 * b
            assert min(writes) == old_end and max(writes) == got - 1
            checked_layouts += 1

    rng = random.Random(20261019)
    checked_proofs = 0
    checked_leaf_addresses = 0
    for pages in (1, 2, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4096):
        for direction in (1, -1):
            for low in (0, 1, 17, 2**24 - pages):
                first = low if direction > 0 else low + pages - 1
                original = [first + direction * i for i in range(pages)]
                variants = [original, original + [-123, 2**24 + 5]]
                if pages >= 3:
                    swapped = original[:]
                    a = rng.randrange(1, pages - 1)
                    swapped[a], swapped[(a + 1) % pages] = swapped[(a + 1) % pages], swapped[a]
                    variants.append(swapped)
                    repeated = original[:]
                    repeated[a] = repeated[a - 1]
                    variants.append(repeated)
                for table in variants:
                    actual = proof(table, pages, 2**24)
                    oracle_direction = (direction if pages > 1 else 1) if table[:pages] == original else 0
                    assert actual == ((first, oracle_direction) if oracle_direction else (0, 0))
                    checked_proofs += 1
                    if actual[1]:
                        # Score叶起点为32页的倍数；覆盖全部尾页数及正/逆序。
                        for begin in range(0, pages, 32):
                            run_pages = min(32, pages - begin)
                            run_first = actual[0] + direction * begin
                            lowest = run_first if direction > 0 else run_first - run_pages + 1
                            loaded = list(range(lowest, lowest + run_pages))
                            logical = loaded if direction > 0 else loaded[::-1]
                            assert logical == table[begin:begin + run_pages]
                            assert 0 <= lowest and lowest + run_pages <= 2**24
                            checked_leaf_addresses += 1
                assert proof(original, pages, 2**24 + 1) == (0, 0)
                assert proof(original, pages + 1, 2**24) == (0, 0)
    result = dict(
        passed=True, layouts=checked_layouts, proofs=checked_proofs,
        leaf_read_ranges=checked_leaf_addresses,
        physical_capacity_boundary=2**24,
        device_execution=False, device_precision_or_performance=False,
    )
    (here / "LAYOUT_CONTRACT_CHECK.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
