# SPDX-License-Identifier: Apache-2.0
"""只接受明确核内清零的完整MIX，不能把丢失清零记录误认作优化。"""

import copy
import unittest

from complete_l1_v2 import summarize


class FusedL1Test(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "metadata": {
                "clock_freq_hz": 50_000_000,
                "num_cores": 72,
                "core_types": ["aic"] * 24 + ["aiv"] * 48,
                "run_boundaries": [[0, 90]],
                "dropped_run_boundaries": 0,
            },
            "chip_swimlane_level": 1,
            "aicore_tasks": [[i, 11, 0, 100 + i, 1000 + i, 0] for i in range(72)],
        }

    def test_fused_span_includes_all_role_entry_and_exit(self):
        result = summarize(self.raw, init_layout="fused")
        self.assertEqual(result["incore_span_us"], 19.42)
        self.assertIsNone(result["initialization_us"])

    def test_separate_reference_cannot_lose_initialization(self):
        with self.assertRaises(ValueError):
            summarize(self.raw)

    def test_fused_cannot_drop_extra_task(self):
        raw = copy.deepcopy(self.raw)
        raw["aicore_tasks"].extend([[i, 10, 0, 10, 90, 0] for i in range(24, 72)])
        with self.assertRaises(ValueError):
            summarize(raw, init_layout="fused")

    def test_full_occupancy_and_single_window_are_required(self):
        for changes in (
            {"num_cores": 80},
            {"core_types": ["aiv"] * 72},
            {"dropped_run_boundaries": 1},
            {"run_boundaries": []},
        ):
            raw = copy.deepcopy(self.raw)
            raw["metadata"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                summarize(raw, init_layout="fused")

    def test_every_role_required(self):
        self.raw["aicore_tasks"][-1][0] = 0
        with self.assertRaises(ValueError):
            summarize(self.raw, init_layout="fused")

    def test_malformed_or_stale_boundary_is_rejected(self):
        for boundaries in ([[]], [[0, 2000]], [[0, float("nan")]], [[0, -1]]):
            raw = copy.deepcopy(self.raw)
            raw["metadata"]["run_boundaries"] = boundaries
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                summarize(raw, init_layout="fused")

    def test_nonfinite_or_noninteger_records_are_rejected(self):
        for value in (float("nan"), float("inf"), 2.5, True):
            raw = copy.deepcopy(self.raw)
            raw["aicore_tasks"][0][4] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(raw, init_layout="fused")


if __name__ == "__main__":
    unittest.main()
