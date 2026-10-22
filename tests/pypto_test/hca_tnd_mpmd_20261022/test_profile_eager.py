# SPDX-License-Identifier: Apache-2.0
"""验证 eager trace 窗口隔离；不需要 NPU，不伪造任何性能结论。"""

import json

import pytest

from profile_eager import extract_windows


def test_all_streams_keep_internal_copies_but_exclude_reset(tmp_path):
    events = [{"ph": "M", "name": "process_name", "pid": 7, "args": {"name": "Ascend Hardware"}}]
    for i, start in enumerate((100, 200)):
        events.extend([
            {"ph": "X", "name": f"test/{i}", "pid": 1, "tid": 1, "ts": start, "dur": 50},
            {"ph": "X", "name": "same_copy", "pid": 7, "tid": 2, "ts": start-8, "dur": 4},
            {"ph": "X", "name": "same_copy", "pid": 7, "tid": 2, "ts": start+2, "dur": 4},
            {"ph": "X", "name": "aicpu", "pid": 7, "tid": 3, "ts": start+5, "dur": 35},
            {"ph": "X", "name": "aicore", "pid": 7, "tid": 4, "ts": start+8, "dur": 20},
        ])
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(events))
    result = extract_windows(path, "test", 2)
    assert result["samples_us"] == [38, 38]
    assert result["busy_union_median_us"] == 38
    assert all(window["events"] == 3 for window in result["windows"])
    assert result["aclgraph_replay"] is False
    events[-1]["dur"] = 100
    path.write_text(json.dumps(events))
    with pytest.raises(ValueError, match="完整包住"):
        extract_windows(path, "test", 2)
