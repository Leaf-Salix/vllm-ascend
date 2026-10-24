# SPDX-License-Identifier: Apache-2.0
"""比较三侧全部输出/接受事件，同时检查有明确答案的重复句子请求。"""

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

from tokenizers import Tokenizer


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("spmd", type=Path)
    parser.add_argument("mpmd", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-native-reference",
        action="store_true",
        help="仅正确性对照：允许复用不执行PyPTO的Native历史输出，显式记录PTO工具链/heap差异",
    )
    parser.add_argument(
        "--native-source-audit",
        type=Path,
        help="Native参考与PTO基础提交不同时，必须提供明确版本及Native路径未变的审计记录",
    )
    args = parser.parse_args()
    roots = {"native": args.native, "spmd_tnd": args.spmd, "mpmd_tnd": args.mpmd}
    configs = {mode: read(root / "configuration.json") for mode, root in roots.items()}
    reference = configs["native"]
    pto_reference = configs["spmd_tnd"]
    assert configs["mpmd_tnd"]["source_commit"] == pto_reference["source_commit"], "两种PTO必须同一基础源码"
    source_audit = read(args.native_source_audit) if args.native_source_audit else None
    if source_audit is not None:
        assert args.reuse_native_reference
        assert source_audit["status"] == "PASS" and source_audit["native_path_unchanged"] is True
        assert source_audit["native_source_commit"] == reference["source_commit"]
        assert source_audit["pto_source_commit"] == pto_reference["source_commit"]
    assert configs["mpmd_tnd"]["toolchain"] == pto_reference["toolchain"], "两种PTO必须使用同一SDK"
    assert configs["mpmd_tnd"].get("pto_ring_config") == pto_reference.get("pto_ring_config"), "两种PTO的heap必须一致"
    for mode, config in configs.items():
        if config["source_commit"] != reference["source_commit"]:
            assert source_audit is not None, "源码版本不同，须先审核Native实际路径，不能只凭历史结果复用"
            assert config["native_extension"] == reference["native_extension"], "Native扩展来源必须一致"
        if not args.reuse_native_reference:
            assert config.get("pto_ring_config") == reference.get("pto_ring_config"), "运行时容量配置必须记录并对齐"
            assert config["toolchain"] == reference["toolchain"], "工具链必须一致"
        assert config.get("hccl_buffer_mb", 1800) == reference.get("hccl_buffer_mb", 1800)
        for key in (
            "bank",
            "batch",
            "decode_tokens",
            "gpu_memory_utilization",
            "selected_cases",
            "TP",
            "DP",
            "EP",
            "HCA",
            "NZ",
            "EPLB",
            "static_kernel",
            "superkernel",
            "cann",
            "graph_mode",
            "deterministic_level",
        ):
            assert config[key] == reference[key], (key, config[key], reference[key])
    bank = Path(reference["bank"])
    plan = read(bank / "plan.json")
    questions = {c["key"]: c for c in read(bank.parent / "mixed128k_questions.json")["cases"]}
    tokenizer = Tokenizer.from_file(str(Path(plan["model"]) / "tokenizer.json"))
    data, stats, modes = {}, {}, {}
    for mode, root in roots.items():
        functional = read(root / "functional_check.json")
        assert functional["status"] == "PASS" and functional["variant"] == mode
        data[mode], stats[mode] = [], []
        for rank in range(16):
            data[mode].append(read(root / f"rank{rank}.mixed.json"))
            if mode == "native" and args.reuse_native_reference:
                assert all(
                    layer["implementation"] == "native"
                    for group in data[mode][-1]["attention_implementations"]
                    for layer in group
                ), "复用的Native参考不能包含PTO执行"
            events = [json.loads(line) for line in (root / f"request_stats_rank{rank}.jsonl").read_text().splitlines()]
            stats[mode].append({e["key"]: e["events"] for e in events})
        simple, replicas = [], []
        for rank, item in enumerate(data[mode]):
            differences = 0
            for a, b in zip(data[mode][0]["requests"], item["requests"]):
                assert a["key"] == b["key"]
                differences += sum(x != y for x, y in zip(a["output_token_ids"], b["output_token_ids"]))
                if b["kind"] != "easy_repeat":
                    continue
                question = questions[b["key"]]["question"]
                count = int(re.search(r"重复下面这句话(\d+)次", question).group(1))
                expected = (question.splitlines()[-1] + "\n") * count
                actual = tokenizer.decode(b["output_token_ids"], skip_special_tokens=False)
                simple.append({"rank": rank, "key": b["key"], "pass": bool(actual) and expected.startswith(actual)})
            replicas.append({"rank": rank, "token_mismatches_vs_rank0": differences})
        assert simple, "必须覆盖带明确答案的简单请求"
        modes[mode] = {
            "functional": "PASS",
            "mean_accepted": functional["acceptance"]["mean_accepted"],
            "accepted_length_histogram": functional["acceptance"]["accepted_length_histogram"],
            "simple_answer_status": "PASS" if all(r["pass"] for r in simple) else "FAIL",
            "simple_answers": simple,
            "cross_dp": replicas,
        }
    pairs = {}
    for left, right in combinations(roots, 2):
        rows = []
        for rank in range(16):
            a, b = data[left][rank], data[right][rank]
            for key in (
                "batch",
                "decode_tokens",
                "atomic_add",
                "pto_runtime",
                "deterministic",
                "hccl_deterministic",
                "pto_attention",
                "worker_runtime_config",
            ):
                assert a[key] == b[key], (left, right, rank, key)
            for aa, bb in zip(a["requests"], b["requests"]):
                assert aa["key"] == bb["key"]
                ta, tb = aa["output_token_ids"], bb["output_token_ids"]
                assert len(ta) == len(tb) == reference["decode_tokens"]
                different = [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y]
                key = aa["key"]
                rows.append(
                    {
                        "rank": rank,
                        "key": key,
                        "kind": aa["kind"],
                        "token_mismatches": len(different),
                        "common_prefix_tokens": different[0] if different else len(ta),
                        "draft_events_equal": stats[left][rank][key] == stats[right][rank][key],
                        "accepted_histograms": {
                            mode: [sum(a == n for _, a in stats[mode][rank][key]) for n in range(6)]
                            for mode in (left, right)
                        },
                    }
                )
        pairs[f"{left}_vs_{right}"] = {
            "requests": len(rows),
            "tokens": len(rows) * reference["decode_tokens"],
            "matched_requests": sum(r["token_mismatches"] == 0 for r in rows),
            "token_mismatches": sum(r["token_mismatches"] for r in rows),
            "matched_draft_event_requests": sum(r["draft_events_equal"] for r in rows),
            "by_kind": {
                kind: {
                    "requests": sum(r["kind"] == kind for r in rows),
                    "matched_requests": sum(r["kind"] == kind and r["token_mismatches"] == 0 for r in rows),
                }
                for kind in sorted({r["kind"] for r in rows})
            },
            "details": rows,
        }
    report = {
        "batch": reference["batch"],
        "scope": (
            "复用Native历史正确性参考；两种PTO同SDK，问题/KV/推理配置一致；不用于跨版本性能比较"
            if args.reuse_native_reference
            else "同源码/工具链/问题/KV/配置，完整256或384请求"
        ),
        "native_reference_reused": args.reuse_native_reference,
        "native_source_audit": source_audit,
        "configuration_provenance": {
            mode: {
                "results": str(root.resolve()),
                "source_commit": configs[mode]["source_commit"],
                "toolchain": configs[mode]["toolchain"],
                "pto_ring_config": configs[mode].get("pto_ring_config"),
            }
            for mode, root in roots.items()
        },
        "modes": modes,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    text = []
    for index, request in enumerate(data["native"][0]["requests"]):
        text += [f"## {request['key']} ({request['kind']})", questions[request["key"]]["question"]]
        for mode in roots:
            text += [
                f"### {mode}",
                tokenizer.decode(data[mode][0]["requests"][index]["output_token_ids"], skip_special_tokens=False),
            ]
    args.output.with_suffix(".texts.md").write_text("\n\n".join(text) + "\n")
    print(
        json.dumps(
            {
                "modes": {
                    m: {k: v for k, v in r.items() if k not in ("simple_answers", "cross_dp")} for m, r in modes.items()
                },
                "pairs": {p: {k: v for k, v in r.items() if k != "details"} for p, r in pairs.items()},
            },
            ensure_ascii=False,
        )
    )
    assert all(r["simple_answer_status"] == "PASS" for r in modes.values())
    assert all(
        pair["by_kind"]["easy_repeat"]["matched_requests"] == pair["by_kind"]["easy_repeat"]["requests"]
        for pair in pairs.values()
    ), "简单问题必须逐token一致，不能只比较解码后的文本"


if __name__ == "__main__":
    main()
