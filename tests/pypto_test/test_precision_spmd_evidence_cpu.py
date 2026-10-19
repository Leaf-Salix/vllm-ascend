# SPDX-License-Identifier: Apache-2.0
"""绑定真实设备切片结果、源码与原始计时，不在CPU重跑候选。"""

import gzip
import hashlib
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd_precision_spmd"
EVIDENCE = Path(__file__).parent / "evidence_tnd_precision/spmd_slice_20261019"


def test_spmd_slice_archive_identity():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert len(manifest) == 6
    for name, expected in manifest.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    compiled = json.loads(gzip.decompress((EVIDENCE / "compile-slice.json.gz").read_bytes()))
    assert compiled["compiled"] is True and compiled["kernel_mode_abi_pass"] is True
    for name, expected in compiled["sources"].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize(
    ("tokens", "name", "checks"),
    [
        (1, "t1-native-result.json.gz", 53),
        (21, "t21-native-result.json.gz", 53),
        (240, "t240-native-control-spmd-result.json.gz", 89),
    ],
)
def test_native_norm_wkv_and_replay_evidence(tokens, name, checks):
    result = json.loads(gzip.decompress((EVIDENCE / name).read_bytes()))
    assert result["complete"] is True and result["pass"] is True
    assert result["tokens"] == tokens and "not vLLM FULL" in result["scope"]
    assert len(result["comparisons"]) == checks
    assert all(
        value["bit_equal"] is True and value["different_elements"] == 0 and value["max_abs"] == 0
        for value in result["comparisons"].values()
    )
    for name, expected in result["candidate_sources"].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected
    for phase in ("A", "B", "A-restored"):
        for stage in ("control-norm-vs-native", "control-wkv-vs-native"):
            assert result["comparisons"][phase + "/" + stage]["bit_equal"]
        for repeat in range(3):
            for stage in ("norm", "rstd", "wkv", "input"):
                assert result["comparisons"][f"{phase}/{repeat}/{stage}"]["bit_equal"]
    for stage in ("norm", "rstd", "wkv", "input", "readonly/gamma", "readonly/weight", "readonly/order"):
        assert result["comparisons"]["after-timing/" + stage]["bit_equal"]
    if tokens == 240:
        timings = result["graph_replay_us"]
        assert set(timings) == {"native", "control", "spmd"}
        for values in timings.values():
            samples = values["samples"]
            assert len(samples) == 6 and all(sample > 0 for sample in samples)
            assert values["median"] == statistics.median(samples)
            assert values["mean"] == statistics.mean(samples)
            assert values["min"] == min(samples) and values["max"] == max(samples)
        assert (
            result["timing_order"]
            == [["native", "control", "spmd"], ["spmd", "native", "control"], ["control", "spmd", "native"]] * 2
        )
