# SPDX-License-Identifier: Apache-2.0
"""CPU 反例：统计量相同、非有限值、广播和整数转换都不能绕过数值门禁。"""

import json
import sys

import pytest
import torch

from dsv4_csa_validation import compare_tensor, compare_topk, validate_outputs


def test_topk_order_is_separate_from_candidate_selection():
    expected = torch.tensor([[0, 1, 2], [0, 1, 2], [0, -1, -1]])
    actual = torch.tensor([[2, 0, 1], [0, 1, 3], [0, -1, -1]])
    report = compare_topk(actual, expected, torch.tensor([4, 4, 1]))
    assert report["status"] == "MEASURED"
    assert report["order_only_rows"] == 1 and report["different_set_rows"] == 1
    assert report["replaced_indices"] == 1 and report["invalid_rows"] == 0


def test_topk_rejects_duplicates_out_of_bounds_and_missing_candidates():
    expected = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    actual = torch.tensor([[0, 0, 2], [0, 1, 3], [0, 1, -1]])
    report = compare_topk(actual, expected, torch.tensor([3, 3, 3]))
    assert report["status"] == "FAIL" and report["invalid_rows"] == 3


def test_permutation_with_same_statistics_fails():
    actual = torch.tensor([46.5, 1.0, -1.0, 0.0])
    expected = torch.tensor([46.5, -1.0, 1.0, 0.0])
    assert actual.mean() == expected.mean()
    assert actual.abs().max() == expected.abs().max()
    report = compare_tensor(actual, expected, 0.01, 0.01)
    assert report["status"] == "FAIL"
    assert report["max_abs"] == 2
    assert report["mismatches"] == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_fails_even_when_both_sides_match(value):
    a = torch.tensor([value])
    assert compare_tensor(a, a, 0, 0)["status"] == "FAIL"
    assert validate_outputs({"x": a}, ["x"])["status"] == "FAIL"


def test_no_broadcast_or_implicit_cast():
    a = torch.ones(2, dtype=torch.bfloat16)
    assert compare_tensor(a, a[:1], 0, 0)["status"] == "FAIL"
    assert compare_tensor(a, a.float(), 0, 0)["status"] == "FAIL"


def test_integer_low_bits_are_compared_before_float_conversion():
    a, b = torch.tensor([2**55]), torch.tensor([2**55 + 1])
    report = compare_tensor(a, b, 0, 0)
    assert report["status"] == "FAIL" and report["mismatches"] == 1
    assert compare_tensor(a, a, 1, 0)["status"] == "FAIL"


def test_reference_and_tolerances_are_required_for_pass():
    actual = {"x": torch.tensor([1.001]), "slots": torch.tensor([2])}
    reference = {"x": torch.tensor([1.0]), "slots": torch.tensor([2])}
    limits = {"x": {"atol": 0.002, "rtol": 0}}
    assert validate_outputs(actual, actual)["status"] == "MEASURED"
    assert validate_outputs(actual, actual, reference)["status"] == "FAIL"
    assert validate_outputs(actual, actual, reference, limits)["status"] == "PASS"
    assert validate_outputs(actual, actual, {"x": reference["x"]}, limits)["status"] == "FAIL"
    assert validate_outputs({"x": actual["x"]}, actual, reference, limits)["status"] == "FAIL"
    assert validate_outputs(actual, actual, reference, {"x": {"atol": 0, "rtol": 0}})["status"] == "FAIL"


def test_benchmark_records_setup_failure_and_raises(monkeypatch, tmp_path):
    import dsv4_csa_single_card_bench as bench

    def fail(args, report):
        raise ValueError("invalid snapshot")

    monkeypatch.setattr(bench, "_run_benchmark", fail)
    monkeypatch.setattr(sys, "argv", ["bench", "--args-dir", str(tmp_path), "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="invalid snapshot"):
        bench.main()
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "FAIL"


@pytest.mark.parametrize("count,nonfinite,expected", [(2, 0, "MEASURED"), (1, 0, "FAIL"), (2, 1, "FAIL")])
def test_observer_requires_samples_and_finite_values(count, nonfinite, expected):
    from offline_pd.observer import OfflineCSAObserver

    observer = OfflineCSAObserver()
    runtime = type("Runtime", (), {})
    # 有限算术差异只作诊断；样本不足或非有限值必须失败。
    sample = {"bit_equal": False, "elementwise": {"status": "FAIL", "nonfinite": nonfinite}}
    state = {"samples": [sample] * count, "max_samples": 2}
    observer._offline_bitcompare = (state, lambda: None, runtime)
    assert observer.offline_end_bitcompare()["status"] == expected
