# SPDX-License-Identifier: Apache-2.0
"""已知数值差异可继续性能采集，但不改精度记录或真实生成。"""

import json
from types import SimpleNamespace

from csa_threeway_20261029.nal_performance import NalPerformanceScheduler


def test_nal_mismatch_is_recorded_without_stop(tmp_path):
    probe = object.__new__(NalPerformanceScheduler)
    probe._threeway_output = tmp_path
    probe.parallel_config = SimpleNamespace(data_parallel_rank=3)
    probe._fail("token", "q02", 0, 3648, 9854)
    path = tmp_path / "performance_difference_rank3.json"
    original = path.read_bytes()
    assert json.loads(original)["actual"] == 3648
    probe._fail("output_length", "q02", 190, 190, 192)
    assert path.read_bytes() == original and not (tmp_path / "STOP").exists()
