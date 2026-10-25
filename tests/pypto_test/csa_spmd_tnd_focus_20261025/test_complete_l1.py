# SPDX-License-Identifier: Apache-2.0
"""计时不能漏掉初始化或依赖间隙，也不能把缺核窗口记为有效样本。"""

import copy
import unittest

from complete_l1 import summarize


class CompleteL1Test(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "metadata": {"clock_freq_hz": 50_000_000},
            "chip_swimlane_level": 1,
            "aicore_tasks": [[i, 10, 0, 100, 200, 0] for i in range(24, 72)]
            + [[i, 11, 0, 250, 1000, 0] for i in range(72)],
        }

    def test_full_span_preserves_init_and_gap(self):
        result = summarize(self.raw)
        self.assertEqual(result["incore_span_us"], 18)
        self.assertEqual(result["main_mix_us"], 15)
        self.assertEqual(result["initialization_us"], 2)
        self.assertEqual(result["initialization_to_mix_gap_us"], 1)

    def test_missing_init_or_mix_core_is_rejected(self):
        for rows in (self.raw["aicore_tasks"][48:], self.raw["aicore_tasks"][:-1]):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                summarize({**self.raw, "aicore_tasks": rows})

    def test_duplicate_role_is_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["aicore_tasks"][-1][0] = 0
        with self.assertRaises(ValueError):
            summarize(raw)

    def test_extra_task_is_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["aicore_tasks"].append([0, 12, 0, 1100, 1200, 0])
        with self.assertRaises(ValueError):
            summarize(raw)

    def test_broken_initialization_dependency_is_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["aicore_tasks"][0][4] = 260
        with self.assertRaises(ValueError):
            summarize(raw)


if __name__ == "__main__":
    unittest.main()
