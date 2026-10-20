# SPDX-License-Identifier: Apache-2.0
"""读取已验证的真实group计划核对短尾工作量，不测性能或冒充新算子。"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent


def main():
    reference = CASE / "source_after_score_split/indexer_group_stages.py"
    captured_source = CASE / "wave_current_math/source_candidate/indexer_group_stages.py"
    assert reference.read_bytes() == captured_source.read_bytes()
    reports = {}
    for batch in (16, 28):
        path = CASE / f"wave_current_math/device_b{batch}_v1/result.json"
        root = json.loads(path.read_text())
        assert root["passed"] and root["batch"] == batch and root["history"] == 131072
        assert root["query_tokens"] == 6
        p = root["launches"][0]["wave_protocol"]
        assert p["passed"]
        groups, _, leaves, leaf_tiles, extra = p["plan"][:5]
        base = (p["requests"] + 16) // 16 * 16
        full_panels = [0] * 24
        short_panels = [0] * 24
        short_steps = [0] * 24
        for item in range(groups * leaves):
            group, leaf = ((item % groups, item // groups) if groups < 24
                           else (item // leaves, item % leaves))
            live, visible = p["offsets"][base + group * 4 + 2:base + group * 4 + 4]
            begin = (leaf * leaf_tiles + min(leaf, extra)) * 1024
            capacity = (leaf_tiles + int(leaf < extra)) * 1024
            valid = max(min(visible - begin, capacity), 0)
            if not live or not valid:
                continue
            owner = item % 24
            for step in range((valid + 1023) // 1024):
                remaining = min(valid - step * 1024, 1024)
                full_panels[owner] += 16
                short_panels[owner] += 1 if remaining <= 64 else 16
                short_steps[owner] += int(remaining <= 64)
        reports[str(batch)] = {
            "captured_metadata": str(path), "group_plan_source_equal_current": True,
            "reference_panels_per_core": full_panels,
            "prospective_short_tail_panels_per_core": short_panels,
            "short_steps_per_core": short_steps,
            "reference_max_panels": max(full_panels), "prospective_max_panels": max(short_panels),
            "all_core_panel_count_saved": sum(full_panels) - sum(short_panels),
            "scope": "旧已通过用例的实际metadata、相同group计划规则；不是本轮新回读或性能",
            "candidate_implemented": False,
            "caveat": "最大工作量不含阶段起点、同步、资源竞争；不能折算微秒或与列池收益相加",
        }
    (HERE / "NEXT_TAIL_WORK_MODEL.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    for batch, row in reports.items():
        print(batch, row["reference_max_panels"], "→", row["prospective_max_panels"],
              "all-core saved", row["all_core_panel_count_saved"])


if __name__ == "__main__":
    main()
