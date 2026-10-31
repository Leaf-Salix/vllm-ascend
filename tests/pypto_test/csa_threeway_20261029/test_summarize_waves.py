# SPDX-License-Identifier: Apache-2.0
"""离线汇总必须拒绝入场轨迹不同的性能，同时独立报告token结果。"""

import json

from csa_threeway_20261029.summarize_waves import summarize


def test_performance_requires_matched_actual_inputs(tmp_path):
    native, ours = tmp_path / "native", tmp_path / "ours"
    for root in (native, ours):
        root.mkdir()
        (root / "experiment.json").write_text(
            json.dumps({"arm": root.name, "csa_source": "test", "wave_measurement": {"hash": "test"}})
        )
        (root / "postflight.json").write_text(
            json.dumps(
                {
                    "passed": root == native,
                    "coverage_errors": [],
                    "acceptance": {"total_preemptions": 0, "mean_accepted": 3.8, "qualified": True},
                }
            )
        )
        for rank in range(8):
            samples = [
                {
                    "actual_requests": 8,
                    "actual_tokens": 48,
                    "padded_tokens": 240,
                    "positions": [i],
                    "forward_us": 100 + i,
                }
                for i in range(20)
            ]
            data = {
                "requests": [{"key": str(i), "output_token_ids": [i] * 192, "finished": True} for i in range(40)],
                "observation": [{"threeway_timing": {"complete": True, "samples": samples}}],
                "generation_waves": [{"offset": i * 8, "elapsed_s": 2, "output_tokens": 8 * 192} for i in range(5)],
            }
            (root / f"rank{rank}.mixed.json").write_text(json.dumps(data))
            (root / f"request_stats_rank{rank}.jsonl").write_text(json.dumps({"key": "0", "events": [[5, 4]]}))
    actual = summarize(ours, native)
    assert actual["performance_qualified"] and actual["tokens"]["tokens_equal"]
    assert actual["target_forward_us"]["mean"] == 109.5
    assert actual["client_waves"]["tokens_per_s"] == 6144
    path = ours / "rank0.mixed.json"
    data = json.loads(path.read_text())
    data["observation"][0]["threeway_timing"]["samples"][0]["positions"] = [99]
    path.write_text(json.dumps(data))
    actual = summarize(ours, native)
    assert actual["tokens"]["tokens_equal"] and not actual["performance_qualified"]
    path.unlink()
    actual = summarize(ours, native)
    assert not actual["complete"] and not actual["performance_qualified"]
