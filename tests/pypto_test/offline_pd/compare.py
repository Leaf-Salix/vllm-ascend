# SPDX-License-Identifier: Apache-2.0
"""在 CPU 比较已保存的 decode token 与 DSpark 统计，不代替层误差或性能验收。"""

import argparse
import json
import math
from pathlib import Path

COUNTERS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")


def _count(value):
    return (type(value) in (int, float) and math.isfinite(value)
            and value >= 0 and value == int(value))


def _spec_stats(value, num_spec_tokens):
    if not isinstance(value, dict) or value.get("sufficient") is not True:
        raise ValueError("缺少有效 DSpark 统计")
    counters = value.get("counters", {})
    if not isinstance(counters, dict) or any(not _count(counters.get(key)) for key in COUNTERS):
        raise ValueError("DSpark 计数缺失或不是有限非负整数")
    counts = {key: int(counters[key]) for key in COUNTERS}
    drafts, proposed, accepted = (counts[key] for key in COUNTERS)
    per_pos = value.get("num_accepted_tokens_per_pos")
    if (not isinstance(per_pos, list) or len(per_pos) != num_spec_tokens
            or any(not _count(x) for x in per_pos)):
        raise ValueError("DSpark 逐位置接受计数缺失或长度错误")
    if (drafts == 0 or proposed == 0 or accepted > proposed
            or proposed > drafts * num_spec_tokens or sum(per_pos) != accepted
            or any(x > drafts for x in per_pos)
            or any(a < b for a, b in zip(per_pos, per_pos[1:]))):
        raise ValueError("DSpark 总计数与逐位置计数不一致")
    counts["num_accepted_tokens_per_pos"] = [int(x) for x in per_pos]
    return counts


def _load_rank(directory, backend, rank, keys, batch, tokens, num_spec_tokens):
    data = json.loads((directory / f"rank{rank}.json").read_text())
    if (not isinstance(data, dict) or data.get("role") != "decode" or data.get("backend") != backend
            or data.get("rank") != rank or data.get("batch") != batch):
        raise ValueError("rank/后端/命令/batch 与本次对照不符")
    cases = data.get("cases")
    if (not isinstance(cases, list) or not cases or not all(isinstance(c, dict) for c in cases)
            or [c.get("key") for c in cases] != keys):
        raise ValueError("case 缺失、重复或顺序与 bank 不符")
    for case in cases:
        rows = case.get("output_token_ids")
        if (case.get("submitted") != batch or not isinstance(rows, list) or len(rows) != batch
                or any(not isinstance(row, list) or len(row) != tokens for row in rows)
                or any(type(x) is not int or x < 0 for row in rows for x in row)):
            raise ValueError(f"{case['key']}: 请求数、token 数或 token 类型不正确")
        case["validated_spec_decode"] = _spec_stats(case.get("spec_decode"), num_spec_tokens)
    return cases


def compare_decode(native, pto, plan, batch, tokens, ranks=16):
    """按显式声明的 rank/请求/token 数比较，不能由已有文件反推预期覆盖。"""
    if min(batch, tokens, ranks) <= 0:
        raise ValueError("预期 rank、batch 和 token 数必须为正")
    report = {
        "status": "FAIL",
        "scope": "已保存 decode 结果的逐 token 与 DSpark 计数对照；不含层误差、保护区或性能验收",
        "expected": {"ranks": ranks, "batch_per_rank": batch, "decode_tokens": tokens},
        "errors": [], "cases": [], "token_mismatches": 0, "spec_decode_mismatched_cases": 0,
        "compared_ranks": 0, "compared_tokens": 0,
    }
    num_spec_tokens = plan["decode"]["speculative_tokens"]
    p_dp = plan["prefill"]["dp"]
    for rank in range(ranks):
        keys = [c["key"] for c in plan["cases"] if c["p_dp_rank"] == rank % p_dp]
        if not keys or len(keys) != len(set(keys)):
            report["errors"].append(f"rank{rank}: bank case 缺失或重复")
            continue
        sides = {}
        for backend, directory in (("native", native), ("pto", pto)):
            try:
                sides[backend] = _load_rank(
                    directory, backend, rank, keys, batch, tokens, num_spec_tokens)
            except (OSError, ValueError, TypeError) as exc:
                report["errors"].append(f"{backend}/rank{rank}: {exc}")
        if len(sides) != 2:
            continue
        report["compared_ranks"] += 1
        for left, right in zip(sides["native"], sides["pto"]):
            mismatches, examples = 0, []
            for request, (a, b) in enumerate(zip(left["output_token_ids"], right["output_token_ids"])):
                for position, (expected, actual) in enumerate(zip(a, b)):
                    if expected != actual:
                        mismatches += 1
                        if len(examples) < 8:
                            examples.append({"request": request, "position": position,
                                             "native": expected, "pto": actual})
            stats = {side: case["validated_spec_decode"]
                     for side, case in (("native", left), ("pto", right))}
            changed = [key for key in stats["native"] if stats["native"][key] != stats["pto"][key]]
            report["cases"].append({"rank": rank, "key": left["key"], "token_mismatches": mismatches,
                                    "first_mismatches": examples, "spec_decode": stats,
                                    "spec_decode_different_fields": changed})
            report["compared_tokens"] += batch * tokens
            report["token_mismatches"] += mismatches
            report["spec_decode_mismatched_cases"] += bool(changed)
    if (not report["errors"] and not report["token_mismatches"]
            and not report["spec_decode_mismatched_cases"] and report["compared_ranks"] == ranks):
        report["status"] = "PASS"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--pto", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, required=True)
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.bank / "plan.json").read_text())
    report = compare_decode(args.native, args.pto, plan, args.batch, args.decode_tokens, args.ranks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{report['status']}: ranks={report['compared_ranks']}/{args.ranks}, "
          f"tokens={report['compared_tokens']}, token_mismatches={report['token_mismatches']}, "
          f"DSpark_mismatched_cases={report['spec_decode_mismatched_cases']}, "
          f"errors={len(report['errors'])}")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
