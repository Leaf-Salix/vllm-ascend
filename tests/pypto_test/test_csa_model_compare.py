# SPDX-License-Identifier: Apache-2.0
"""模型结果门禁的 CPU 反例：token 相同仍须比较接受分布，缺项不能通过。"""

import json

from offline_pd.compare import compare_decode


def make_results(tmp_path):
    plan = {"prefill": {"dp": 1}, "decode": {"speculative_tokens": 2},
            "cases": [{"key": "h8192_v0", "p_dp_rank": 0}]}
    for backend in ("native", "pto"):
        directory = tmp_path / backend
        directory.mkdir()
        payload = {"role": "decode", "backend": backend, "rank": 0, "batch": 1, "cases": [{
            "key": "h8192_v0", "submitted": 1, "output_token_ids": [[7, 8, 9]],
            "spec_decode": {"sufficient": True, "counters": {
                "num_drafts": 3, "num_draft_tokens": 6, "num_accepted_tokens": 2},
                "num_accepted_tokens_per_pos": [2, 0]}}]}
        (directory / "rank0.json").write_text(json.dumps(payload))
    return plan


def result(tmp_path, plan, ranks=1):
    return compare_decode(tmp_path / "native", tmp_path / "pto", plan, 1, 3, ranks)


def test_token_permutation_fails_even_with_equal_statistics(tmp_path):
    plan = make_results(tmp_path)
    assert result(tmp_path, plan)["status"] == "PASS"
    path = tmp_path / "pto/rank0.json"
    data = json.loads(path.read_text())
    data["cases"][0]["output_token_ids"] = [[8, 7, 9]]
    path.write_text(json.dumps(data))
    report = result(tmp_path, plan)
    assert report["status"] == "FAIL" and report["token_mismatches"] == 2
    assert report["cases"][0]["first_mismatches"][0]["position"] == 0


def test_equal_tokens_and_accepted_totals_do_not_hide_per_position_difference(tmp_path):
    plan = make_results(tmp_path)
    path = tmp_path / "pto/rank0.json"
    data = json.loads(path.read_text())
    data["cases"][0]["spec_decode"]["num_accepted_tokens_per_pos"] = [1, 1]
    path.write_text(json.dumps(data))
    report = result(tmp_path, plan)
    assert report["status"] == "FAIL" and report["token_mismatches"] == 0
    assert report["spec_decode_mismatched_cases"] == 1


def test_missing_rank_cannot_shrink_expected_coverage(tmp_path):
    plan = make_results(tmp_path)
    report = result(tmp_path, plan, ranks=2)
    assert report["status"] == "FAIL" and report["compared_ranks"] == 1
    assert len(report["errors"]) == 2


def test_empty_or_partial_metrics_never_pass(tmp_path):
    plan = make_results(tmp_path)
    path = tmp_path / "pto/rank0.json"
    data = json.loads(path.read_text())
    data["cases"][0]["spec_decode"] = {"sufficient": True, "counters": {"num_drafts": 3}}
    path.write_text(json.dumps(data))
    report = result(tmp_path, plan)
    assert report["status"] == "FAIL" and report["compared_ranks"] == 0
    assert "DSpark" in report["errors"][0]
