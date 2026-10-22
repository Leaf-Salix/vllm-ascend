# SPDX-License-Identifier: Apache-2.0
"""检查真实16卡路由/混合接受率，并逐请求比较冻结SPMD与MPMD。"""

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def collect(root):
    config = read_json(root / "configuration.json")
    variant = config["variant"]
    package = "deepseek_v4_flash_csa_tnd" if variant == "mpmd_tnd" else "deepseek_v4_flash_csa_spmd_tnd"
    expected_runtime = "vllm_ascend.ops.pypto." + package + ".service"
    acceptance = read_json(root / "acceptance.json")
    errors, records, rank_summaries = [], {}, []
    if not acceptance["qualified"]:
        errors.append("接受率/边界/Prometheus检查未通过")
    expected_keys = [c["key"] for c in config["selected_cases"]]
    for rank in range(16):
        data = read_json(root / f"rank{rank}.mixed.json")
        requests = data["requests"]
        if data["variant"] != variant or [r["key"] for r in requests] != expected_keys:
            errors.append(f"rank{rank}的实现或请求清单不符")
        csa = [r for r in data["attention_implementations"][0] if r["kind"] == "csa"]
        hca = [r for r in data["attention_implementations"][0] if r["kind"] == "hca"]
        if len(csa) != 21 or any(r["implementation"] != "pto" or r["runtime_class"] != expected_runtime for r in csa):
            errors.append(f"rank{rank}未安装全部21个预期TND CSA")
        if len(hca) != 20 or any(r["implementation"] != "native" for r in hca):
            errors.append(f"rank{rank}的Native HCA配置不符")
        if len(data["observation"]) != 1:
            raise ValueError("TP1每个DP应有一份worker观测")
        observed = data["observation"][0]
        before, after = observed["tnd_before"], observed["tnd_after"]
        if len(after) != 21 or set(before) != set(after):
            errors.append(f"rank{rank}缺少C4层运行时观测")
        layers = {}
        for layer, last in after.items():
            first = before[layer]
            submit_delta = last["submissions"] - first["submissions"]
            checks = {
                kind: last["graph_contract_checks"].get(kind, 0) - first["graph_contract_checks"].get(kind, 0)
                for kind in ("allowed", "rejected")
            }
            protocol_ran = any(v["min"] > 0 for v in last["generation"].values())
            if (
                last["runtime_class"] != expected_runtime
                or not protocol_ran
                or not (submit_delta > 0 or checks["allowed"] > 0)
            ):
                errors.append(f"rank{rank}/layer{layer}缺少生成阶段实际TND执行证据")
            layers[layer] = {"submissions": submit_delta, "graph": checks, "generation": last["generation"]}
        # ACLGraph合同计数可以证明真实重放；没有计数的其他backend不能由标签冒充。
        graph_proven = bool(layers) and all(v["graph"]["allowed"] > 0 for v in layers.values())
        rank_summaries.append(
            {
                "rank": rank,
                "layers": layers,
                "all_c4_graph_replay_proven": graph_proven,
                "model_forward_counts": observed["model_forward_counts"],
            }
        )
        stats = [json.loads(line) for line in (root / f"request_stats_rank{rank}.jsonl").read_text().splitlines()]
        by_key = {s["key"]: s for s in stats}
        if len(stats) != len(expected_keys) or set(by_key) != set(expected_keys):
            raise ValueError(f"rank{rank}的逐请求事件缺失或重复")
        for request in requests:
            key = request["key"]
            tokens = request["output_token_ids"]
            if len(tokens) != config["decode_tokens"]:
                errors.append(f"rank{rank}/{key}输出token数量错误")
            records[rank, key] = {
                "tokens": tokens,
                "events": by_key[key]["events"],
                "preemptions": by_key[key].get("preemptions"),
            }
    cross_replica = all(records[rank, key] == records[0, key] for rank in range(16) for key in expected_keys)
    report = {
        "root": str(root),
        "variant": variant,
        "route_and_acceptance_passed": not errors,
        "all_c4_graph_replay_proven": all(r["all_c4_graph_replay_proven"] for r in rank_summaries),
        "cross_replica_equal": cross_replica,
        "errors": errors,
        "ranks": rank_summaries,
        "acceptance": {k: v for k, v in acceptance.items() if k != "requests"},
        "state_scope": "实际协议代际推进；不将其冒充全部cache内容bit一致",
    }
    return config, records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, actual, report = collect(args.candidate)
    result = {"candidate": report, "paired": False, "passed": False}
    if args.reference:
        other, expected, baseline = collect(args.reference)
        for name in (
            "source_commit",
            "reference_math_commit",
            "toolchain",
            "bank",
            "selected_cases",
            "batch",
            "decode_tokens",
        ):
            if config[name] != other[name]:
                raise ValueError(f"对照条件不一致：{name}")
        if config["variant"] != "mpmd_tnd" or other["variant"] != "spmd_tnd":
            raise ValueError("要求MPMD对冻结SPMD")
        if actual.keys() != expected.keys():
            raise ValueError("两侧请求不一致")
        token_rows, event_rows = [], []
        mismatched_tokens = 0
        for key in actual:
            a, e = actual[key], expected[key]
            positions = [i for i, (x, y) in enumerate(zip(a["tokens"], e["tokens"])) if x != y]
            mismatched_tokens += len(positions)
            if positions:
                token_rows.append(
                    {"rank": key[0], "key": key[1], "first_position": positions[0], "positions": len(positions)}
                )
            if a["events"] != e["events"]:
                event_rows.append({"rank": key[0], "key": key[1]})
        result.update(
            reference=baseline,
            paired=True,
            mismatched_tokens=mismatched_tokens,
            token_different_requests=token_rows,
            acceptance_different_requests=event_rows,
        )
        result["passed"] = (
            report["route_and_acceptance_passed"]
            and baseline["route_and_acceptance_passed"]
            and report["all_c4_graph_replay_proven"]
            and baseline["all_c4_graph_replay_proven"]
            and report["cross_replica_equal"]
            and baseline["cross_replica_equal"]
            and not token_rows
            and not event_rows
        )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("candidate", "reference")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
