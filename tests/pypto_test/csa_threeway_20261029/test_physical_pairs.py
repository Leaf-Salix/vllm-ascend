# SPDX-License-Identifier: Apache-2.0
"""Scalar Native页验证/换对 vs 物理plan+整数向量mask的CPU契约；只在227运行。"""

import random
import struct
import unittest


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def scalar_reference(pair, table, limit, pages):
    checked = []
    for raw in pair:
        good = 0 <= raw < limit and raw // 32 < len(table)
        if good:
            good = 0 <= table[raw // 32] < pages
        checked.append(raw if good else -1)
    swap = False
    if all(i >= 0 and i + 1 < limit for i in checked):
        first, second = [table[i // 32] * 32 + i % 32 for i in checked]
        swap = first > second and (first - second - 1) * 1024 < 2147483647
    if swap:
        checked.reverse()
    return [table[i // 32] * 32 + i % 32 if i >= 0 else -1 for i in checked]


def vector_contract(pair, table, limit, pages):
    # 原physical plan有效域中的索引均<limit<=2**22，raw+1再减1不损失合法地址。
    physical = [
        table[r // 32] * 32 + r % 32 if 0 <= r < limit and r // 32 < len(table) and 0 <= table[r // 32] < pages else -1
        for r in pair
    ]
    p0, p1 = physical
    r0, r1 = [max(r, 0) for r in pair]

    def mask(value):
        return min(max(value, 0), 1)

    difference = p0 - p1
    swap = (
        mask(p0 + 1)
        * mask(p1 + 1)
        * mask((limit - 1) - r0)
        * mask((limit - 1) - r1)
        * mask(difference)
        * mask(2097153 - difference)
    )
    assert swap in (0.0, 1.0)
    # 只用mask选整数下标，不把物理地址经浮点再转换回去。
    return [physical[i + int(swap) * (1 - 2 * (i % 2))] for i in range(2)]


class PhysicalPairTests(unittest.TestCase):
    def test_random_pages_and_sentinels(self):
        rng = random.Random(1024)
        for _ in range(5000):
            pages = rng.choice([1, 32, 65538, 524288, 16777216])
            table = [rng.choice([-1, 0, pages - 1, pages, rng.randrange(pages)]) for _ in range(12)]
            limit = rng.randrange(1, 384)
            edge = [-2147483648, -1, 0, 1, limit - 2, limit - 1, limit, 2147483647]
            pair = [rng.choice(edge + [rng.randrange(512)]) for _ in range(2)]
            self.assertEqual(vector_contract(pair, table, limit, pages), scalar_reference(pair, table, limit, pages))

    def test_plan_upper_bound_in_guarded_domain(self):
        for pos in [0, 1, 3, 4, 16383, 1048575, 16777215]:
            self.assertEqual(int(f32(f32(f32(pos) + 1) * 0.25)), (pos + 1) // 4)

    def test_gap_and_last_key_boundaries(self):
        for difference in [1, 2097151, 2097152, 2097153, 16777183, 536870879]:
            table = [difference // 32, 0]
            pair = [difference % 32, 32]
            for limit in [33, 34, 35, 128]:
                self.assertEqual(
                    vector_contract(pair, table, limit, 16777216), scalar_reference(pair, table, limit, 16777216)
                )
        self.assertEqual(scalar_reference([64, 0], [0, 0, 3], 66, 8), [0, 96])
        self.assertEqual(scalar_reference([64, 0], [0, 0, 3], 65, 8), [96, 0])


if __name__ == "__main__":
    unittest.main()
