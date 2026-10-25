# SPDX-License-Identifier: Apache-2.0
"""CPU-only harness control-flow and fault-injection tests; no CSA simulation."""

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "runs/internal_state_harness_v1"


def load_validator():
    scenarios = types.ModuleType("scenarios")
    scenarios.make_fixture = None
    with patch.dict(sys.modules, scenarios=scenarios):
        spec = importlib.util.spec_from_file_location("internal_state_test_validator", HARNESS / "state_replay.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def exercise(fault=None, requests=16):
    module = load_validator()
    capacity = requests * 6
    allocation = torch.zeros(128, dtype=torch.uint8)
    view = allocation.view(16, 8)
    root = {
        "x_hc": torch.zeros((capacity, 4, 4), dtype=torch.bfloat16),
        "x_out": torch.zeros((capacity, 4), dtype=torch.float32),
        "idx_topk": torch.zeros((capacity, 2), dtype=torch.int32),
        "idx_topk_scores": torch.zeros((capacity, 2), dtype=torch.float32),
        "kv_cache": view,
        "readonly_weight": torch.arange(8, dtype=torch.float32),
    }
    for name in module.METADATA_NAMES:
        root[name] = torch.zeros((capacity, 2), dtype=torch.int32)
    root["cmp_block_table"] = torch.zeros((requests, 17), dtype=torch.int32)
    group = {"allocation": allocation, "views": [view], "initial": allocation.clone(), "prefix": "cache"}
    fixture = {
        "tokens": capacity,
        "request_capacity": requests,
        "groups": {"cache": group},
        "hidden": root["x_hc"],
        "readonly": {"positions": (root["position_ids"], root["position_ids"].clone())},
    }
    metadata_calls, policies = [], []
    last = {}

    def make_fixture(*args, **kwargs):
        queries, histories = kwargs["query_lengths"], kwargs["history_lengths"]
        metadata_calls.append((list(queries), list(histories), kwargs["page_owners"], kwargs["indexer_page_pattern"]))
        last["stamp"] = sum(queries)
        allowed = torch.zeros_like(allocation, dtype=torch.bool)
        allowed[0] = True
        request = types.SimpleNamespace(decode=types.SimpleNamespace(slot_mapping=torch.tensor([[0, 0]])))
        return {
            "metadata": {"cache": request},
            "groups": {"cache": {**group, "allowed": allowed}},
            "positions": root["position_ids"],
            "compact": {},
            "query_lengths": queries,
            "history_lengths": histories,
        }

    module.make_fixture = make_fixture

    def bind(*args, **kwargs):
        return {name: torch.full_like(root[name], last["stamp"]) for name in module.METADATA_NAMES}

    policy_module = types.ModuleType("internal_state_fake_policy")
    Policy = type("Policy", (), {"__module__": policy_module.__name__})
    policy = Policy()

    def select_policy(**kwargs):
        policies.append(kwargs)
        assert kwargs["compressed_table_columns"] == 17
        return policy

    policy_module.select_policy = select_policy
    entries = {
        side: types.SimpleNamespace(buffers={"x": root["x_hc"]}, allocations={}, policy=policy)
        for side in ("reference", "candidate")
    }
    replay_count = {side: 0 for side in entries}

    def replay(side):
        replay_count[side] += 1
        next_value = (int(allocation[0]) + last["stamp"]) % 251
        allocation[0] = next_value
        root["x_out"].fill_(next_value)
        root["idx_topk"].fill_(last["stamp"])
        root["idx_topk_scores"].fill_(next_value)
        if side == "candidate" and fault and replay_count[side] == 1:
            if fault == "output":
                root["x_out"][0, 0] += 1
            elif fault == "readonly":
                root["readonly_weight"][0] += 1
            elif fault == "metadata":
                root["position_ids"][0, 0] += 1
            elif fault == "cache_guard":
                allocation[1] = 9
            elif fault == "cache_bytes":
                allocation[0] = (next_value + 1) % 251
            elif fault == "address":
                entries[side].buffers["x"] = root["x_hc"].clone()
            elif fault == "missing_bit_equal":
                pass
            else:
                raise AssertionError(fault)

    graphs = {side: types.SimpleNamespace(replay=lambda side=side: replay(side)) for side in entries}

    def compare(values, expected):
        checked = {
            name: {
                "status": "PASS" if torch.equal(value, expected[name]) else "FAIL",
                "bit_equal": torch.equal(value.view(torch.uint8), expected[name].view(torch.uint8)),
            }
            for name, value in values.items()
        }
        if fault == "missing_bit_equal":
            checked["output"].pop("bit_equal")
        return checked

    layer = types.SimpleNamespace(
        self_attn=types.SimpleNamespace(
            dsa_attn=types.SimpleNamespace(dsa_attn=types.SimpleNamespace(layer_name="stub"))
        )
    )
    # Tensor.cpu() normally allocates independent storage for device tensors.
    # Clone CPU tensors here to model that snapshot contract without any device.
    with (
        patch.dict(sys.modules, {policy_module.__name__: policy_module}),
        patch.object(torch, "npu", types.SimpleNamespace(synchronize=lambda: None), create=True),
        patch.object(torch.Tensor, "cpu", lambda tensor: tensor.detach().clone()),
    ):
        result = module.validate(
            config=None,
            layer=layer,
            fixture=fixture,
            root=root,
            native_bind=bind,
            entries=entries,
            graphs=graphs,
            guards={},
            compare=compare,
            case={"history_label": 131072, "table_history": 131200, "seed": 1},
            report={},
            save=lambda *args: None,
        )
    assert result["passed"] is (fault is None), fault
    assert not result["internal_observation"]["protocol_pass_claim"]
    assert all(
        v == "not_observed_internal_workspace"
        for k, v in result["internal_observation"].items()
        if k != "protocol_pass_claim"
    )
    if fault is None:
        assert len(result["metadata_cases"]) == 12 and len(result["continuous_steps"]) == 4
        assert replay_count == {"reference": 16, "candidate": 16}
        assert len(policies) == 32 and len(metadata_calls) == 16
        expected_labels = [f"uniform_s{s}" for s in (1, 6, 2, 3, 4, 5)] + [
            "no_compact_output",
            "mixed_reverse_owners",
            "uniform_s1_again",
            "priority_all_s6",
            "priority_reverse_owners",
            "fragmented_pages",
        ]
        assert [row["name"] for row in result["metadata_cases"]] == expected_labels
        assert metadata_calls[9][0] == [6] * requests
        assert metadata_calls[10][2] == list(reversed(range(requests)))
        assert [row["starts_from_previous_state"] for row in result["continuous_steps"]] == [False, True, True, True]
        assert all(
            row["checks"][side]["external_scratch_guard_count"] == 0
            for row in result["metadata_cases"]
            for side in entries
        )
    return {
        "fault": fault,
        "requests": requests,
        "rejected": fault is not None,
        "metadata_cases_executed": len(result["metadata_cases"]),
        "continuous_steps_executed": len(result["continuous_steps"]),
        "no_internal_protocol_claim": True,
    }


def main():
    original = (HARNESS / "source_validator/state_replay.py").read_text()
    adapted = (HARNESS / "state_replay.py").read_text()
    marker = "    # 同一捕获根从小S切到大S"
    assert original[original.index(marker) :] == adapted[adapted.index(marker) :]
    runner = (HARNESS / "run_pto.py").read_text()
    for name in ("state_replay.py", "run_pto.py"):
        ast.parse((HARNESS / name).read_text())
    assert not any(text in runner for text in ("begin_dfx", "end_dfx", "export_l1("))
    assert "enable_chip_swimlane=0" in runner
    assert ".allocations" not in adapted
    assert 'buffers[branch + "_offsets"]' not in adapted
    assert 'compressed_table_columns=root["cmp_block_table"].shape[1]' in adapted
    launcher = (HERE / "launch_internal_state.sh").read_text()
    assert "--shared-toolchain" in launcher and "--state-validation" in launcher
    assert "$" + "{TASK_DEVICE:?通过task-submit提交}" in launcher
    checks = [exercise(requests=16), exercise(requests=24)]
    checks += [
        exercise(fault)
        for fault in (
            "output",
            "readonly",
            "metadata",
            "cache_guard",
            "cache_bytes",
            "address",
            "missing_bit_equal",
        )
    ]
    report = {
        "status": "CPU_AST_and_fault_injection_passed",
        "device_execution": False,
        "scope": "Harness control flow only; CPU stubs do not validate CSA or hardware synchronization",
        "original_schedule_preserved_verbatim": True,
        "no_performance_calls": True,
        "tests": checks,
    }
    path = HERE / "internal_state_harness_v1_cpu_gate.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "tests": len(checks), "output": str(path)}))


if __name__ == "__main__":
    main()
