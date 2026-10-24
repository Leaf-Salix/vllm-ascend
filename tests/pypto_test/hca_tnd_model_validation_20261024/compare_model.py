# SPDX-License-Identifier: Apache-2.0
"""对照Native与HCA整模型输出和DSpark事件；报告差异，不自动放宽token验收。"""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("pto", type=Path)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native_config, config = read(args.native / "configuration.json"), read(args.pto / "configuration.json")
    audit = read(args.source_audit)
    assert audit["status"] == "PASS" and audit["native_path_unchanged"]
    assert audit["native_source_commit"] == native_config["source_commit"]
    assert audit["pto_source_commit"] == config["source_commit"]
    for key in ("bank", "batch", "decode_tokens", "selected_cases", "gpu_memory_utilization", "TP", "DP", "EP", "NZ",
                "EPLB", "static_kernel", "superkernel", "graph_mode", "deterministic_level", "cann", "native_extension"):
        assert native_config[key] == config[key], (key, native_config[key], config[key])
    assert native_config.get("hccl_buffer_mb", 1800) == config["hccl_buffer_mb"]
    assert read(args.native / "functional_check.json")["status"] == "PASS"
    assert read(args.pto / "functional_check.json")["status"] == "PASS"
    from tokenizers import Tokenizer
    model = Path(read(Path(config["bank"]) / "plan.json")["model"])
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    rows = []
    for rank in range(16):
        left, right = read(args.native / f"rank{rank}.mixed.json"), read(args.pto / f"rank{rank}.mixed.json")
        assert all(r["implementation"] == "native" for group in left["attention_implementations"] for r in group)
        def events(root):
            return {r["key"]: r["events"] for r in (json.loads(line) for line in
                    (root / f"request_stats_rank{rank}.jsonl").read_text().splitlines())}
        le, re = events(args.native), events(args.pto)
        assert len(left["requests"]) == len(right["requests"]) == config["batch"]
        for a, b in zip(left["requests"], right["requests"]):
            assert a["key"] == b["key"]
            at, bt = a["output_token_ids"], b["output_token_ids"]
            assert len(at) == len(bt) == config["decode_tokens"]
            differences = [i for i, (x, y) in enumerate(zip(at, bt)) if x != y]
            rows.append(dict(rank=rank, key=a["key"], kind=a["kind"], token_equal=not differences,
                             mismatches=len(differences), first_difference=differences[0] if differences else None,
                             dspark_events_equal=le[a["key"]] == re[b["key"]],
                             native_tokens=at, hca_tokens=bt,
                             native_text=tokenizer.decode(at, skip_special_tokens=False),
                             hca_text=tokenizer.decode(bt, skip_special_tokens=False)))
    simple = [r for r in rows if r["kind"] == "easy_repeat"]
    report = dict(status="COMPARED", performance_comparison=False, source_audit=str(args.source_audit),
                  native=str(args.native), hca=str(args.pto), requests=len(rows),
                  all_tokens_equal=all(r["token_equal"] for r in rows),
                  token_equal_requests=sum(r["token_equal"] for r in rows),
                  simple_token_equal_requests=sum(r["token_equal"] for r in simple), simple_requests=len(simple),
                  all_dspark_events_equal=all(r["dspark_events_equal"] for r in rows), rows=rows)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
