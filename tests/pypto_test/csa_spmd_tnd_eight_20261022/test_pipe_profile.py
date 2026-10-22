# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete/invalid hardware evidence before making utilization claims."""

import csv
import json

import pytest
from profile_current import REQUIRED, summarize_csv


def rows():
    return [
        dict(
            {"Accelerator Core": "MIX_AIC", "Step Id": str(i), "Block Num": "24", "Mix Block Num": "48"},
            **{name: str(1 + i) for name in REQUIRED},
        )
        for i in range(3)
    ]


def write(tmp_path, records):
    path = tmp_path / "kernel_details.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


def test_retains_all_samples(tmp_path):
    result = summarize_csv(write(tmp_path, rows()))
    assert len(result["raw_rows"]) == 3
    assert result["summary"]["aic_mac_time(us)"] == {"samples": [1.0, 2.0, 3.0], "median": 2.0}


@pytest.mark.parametrize("damage", ["missing", "duplicate_step", "half_vectors", "nan", "inactive_counter"])
def test_rejects_invalid_capture(tmp_path, damage):
    records = rows()
    if damage == "missing":
        records.pop()
    elif damage == "duplicate_step":
        records[2]["Step Id"] = "1"
    elif damage == "half_vectors":
        records[1]["Mix Block Num"] = "24"
    elif damage == "nan":
        records[1]["aic_mte2_time(us)"] = "nan"
    elif damage == "inactive_counter":
        records[1]["aiv_vec_time(us)"] = "0"
    with pytest.raises(ValueError):
        summarize_csv(write(tmp_path, records))


@pytest.mark.parametrize("damage", [None, "missing_marker", "outside_marker", "overlap"])
def test_current_csv_requires_unambiguous_trace_markers(tmp_path, damage):
    records = rows()
    markers = []
    for index, row in enumerate(records):
        row.pop("Step Id")
        row.update({"Name": "one_mix_kernel", "Start Time(us)": str(100 + 10 * index), "Duration(us)": "2"})
        markers.append({"name": "CSA_TND_CURRENT_PMU", "ts": 99 + 10 * index, "dur": 5})
    if damage == "missing_marker":
        markers.pop()
    elif damage == "outside_marker":
        markers[1]["dur"] = 1
    elif damage == "overlap":
        markers[0]["dur"] = 15
    path = write(tmp_path, records)
    path.with_name("trace_view.json").write_text(json.dumps(markers))
    if damage:
        with pytest.raises(ValueError):
            summarize_csv(path)
    else:
        result = summarize_csv(path)
        assert "trace markers" in result["sample_binding"]
        assert len(result["raw_rows"]) == 3
