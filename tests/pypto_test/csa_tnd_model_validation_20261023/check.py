# SPDX-License-Identifier: Apache-2.0
"""检查真实16卡decode完成、S6路由及部署配置；数值对照另行报告。"""

import json
import sys
from collections import Counter
from pathlib import Path


def check_completion(request, limit, respect_eos):
    """区分EOS完成与上限截断；正常停止本身不证明答案正确。"""
    count = len(request["output_token_ids"])
    if not respect_eos:
        assert count == limit
        return "fixed_length"
    assert request["finished"] and 0 < count <= limit
    reason = request["finish_reason"]
    assert reason in ("stop", "length"), reason
    assert reason != "length" or count == limit
    assert isinstance(request["output_text"], str)
    return reason


def check(root):
    config = json.loads((root / "configuration.json").read_text())
    variant, batch = config["variant"], config["batch"]
    tokens = config["decode_tokens"]
    respect_eos = not config.get("ignore_eos", True)
    plan = json.loads((Path(config["bank"]) / "plan.json").read_text())
    vocab = json.loads((Path(plan["model"]) / "config.json").read_text())["vocab_size"]
    package = {"mpmd_tnd": "deepseek_v4_flash_csa_tnd", "spmd_tnd": "deepseek_v4_flash_csa_spmd_tnd"}.get(variant)
    acceptance = json.loads((root / "acceptance.json").read_text())
    assert acceptance["qualified"], acceptance
    assert acceptance["total_preemptions"] == 0, "不能用抢占缩减实际并发来冒充B16/B24"
    rows = []
    total_output_tokens = 0
    completion = Counter()
    for rank in range(16):
        data = json.loads((root / f"rank{rank}.mixed.json").read_text())
        assert data["rank"] == rank and data["batch"] == batch and data["decode_tokens"] == tokens
        assert data["backend"] == ("pto" if package else "native")
        assert not package or data["variant"] == variant
        assert len(data["requests"]) == batch
        reasons = Counter(check_completion(r, tokens, respect_eos) for r in data["requests"])
        completion.update(reasons)
        total_output_tokens += sum(len(r["output_token_ids"]) for r in data["requests"])
        if respect_eos:
            assert data["sampling"] == {"temperature": 0.0, "ignore_eos": False, "max_tokens": tokens}
        assert all(type(t) is int and 0 <= t < vocab for r in data["requests"] for t in r["output_token_ids"])
        (implementations,) = data["attention_implementations"]
        csa = [r for r in implementations if r["kind"] == "csa"]
        hca = [r for r in implementations if r["kind"] == "hca"]
        assert len(csa) == 21 and len(hca) == 20
        assert all(r["implementation"] == "native" for r in hca)
        if package:
            assert all(
                r["implementation"] == "pto" and r["runtime_class"] == f"vllm_ascend.ops.pypto.{package}.service"
                for r in csa
            )
        else:
            assert all(r["implementation"] == "native" for r in csa)
        (runtime,) = data["worker_runtime_config"]
        assert runtime["runtime_environment"]["HCCL_BUFFSIZE"] == str(config.get("hccl_buffer_mb", 1800))
        assert runtime["deterministic_level"] == 1 and not runtime["dynamic_eplb"]
        assert runtime["cann_event_work_mode"] == 1
        compilation = runtime["decode_optimizations"]["ascend_compilation_config"]
        assert not compilation["enable_static_kernel"] and compilation["enable_npugraph_ex"]
        (observation,) = data["observation"]
        occupancy = observation.get("decode_occupancy", [])
        full_s6_steps = sum(
            r["steps"] for r in occupancy if r["actual_requests"] == batch and r["actual_tokens"] == batch * 6
        )
        if batch == 24 or occupancy:
            assert full_s6_steps > 0, (rank, "没有有效B×S6，图padding容量不能代替真实并发", occupancy)
        key = f"FULL_tokens{batch * 6}"
        assert observation["model_forward_counts"].get(key, 0) > 0, (rank, observation["model_forward_counts"])
        if package:
            layers = observation["target_layer_names"]
            assert len(layers) == 21
            assert all(
                observation["capture_time_selection"].get(layer, {}).get(f"pto_tokens{batch * 6}", 0) > 0
                for layer in layers
            )
            assert len(observation["tnd_bindings"]) == 21
            assert all(
                r["decode_submissions"] > 0 or r["decode_graph_checks"]["allowed"] > 0
                for r in observation["tnd_bindings"]
            ), "生成阶段缺少实际PTO提交或重放证据"
            if config.get("internal_workspace"):
                assert observation["tnd_physical_workspace_bytes"] == 0, "仍有Python侧TND临时工作区"
                assert all(
                    binding["workspace_bytes"] == 0
                    for layer in observation["tnd_bindings"]
                    for binding in layer["bindings"]
                ), "内部工作区版本不能保留外部scratch"
            if config.get("require_tnd_graph_replay"):
                routes = observation.get("tnd_execution_routes", [])
                real_decode = [
                    row
                    for row in routes
                    if row["requested_mode"] == "FULL" and row["actual_tokens"] > 0 and row["max_seq_len"] >= 8192
                ]
                assert real_decode, (rank, "缺少真实长历史图执行证据")
                assert any(row["actual_requests"] == batch for row in real_decode), (rank, "满批未覆盖")
                assert any(row["actual_requests"] < batch for row in real_decode), (rank, "尾批未覆盖")
                assert any(row["actual_tokens"] < row["padded_tokens"] for row in real_decode), (rank, "padding未覆盖")
                assert all(row["all_csa_graph_replayed"] and not row["python_resubmitted"] for row in real_decode), (
                    rank,
                    "真实decode仍有eager重派发",
                    real_decode,
                )
        rows.append(
            {
                "rank": rank,
                "completion": dict(reasons),
                # FULL是调用时的图模式标签，合同拒绝后仍可能实际执行eager。
                "full_s6_graph_mode_forwards": observation["model_forward_counts"][key],
                "actual_full_s6_steps": full_s6_steps if occupancy else None,
                "decode_occupancy": occupancy,
                "memory": observation["memory"],
                "physical_workspace_bytes": observation.get("tnd_physical_workspace_bytes"),
                "all_csa_graph_replay_proven": bool(package)
                and all(r["decode_graph_checks"]["allowed"] > 0 for r in observation["tnd_bindings"]),
                "execution_routes": observation.get("tnd_execution_routes", []),
                "workspace_bytes": sum(
                    b["workspace_bytes"] for r in observation["tnd_bindings"] for b in r["bindings"]
                ),
            }
        )
    report = {
        "status": "PASS",
        "scope": "功能及实际S6路由通过；不代替输出正确性或整模型图重放验收；FULL模式计数不证明实际重放",
        "variant": variant,
        "batch": batch,
        "requests": 16 * batch,
        "output_tokens": total_output_tokens,
        "sampling": {"ignore_eos": not respect_eos, "max_tokens": tokens},
        "completion": dict(completion),
        "acceptance": {k: v for k, v in acceptance.items() if k != "requests"},
        "ranks": rows,
    }
    (root / "functional_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "ranks"}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    check(Path(sys.argv[1]))
