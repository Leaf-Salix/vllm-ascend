# SPDX-License-Identifier: Apache-2.0
"""Qualify completed C02 state/service evidence before production adoption."""

import json
from pathlib import Path

from analyze_priority_phase import executable
from measurement_provenance import require_same_sdk


def require_strict(numerical):
    assert numerical and all(row["status"] == "PASS" and row["bit_equal"] for row in numerical.values())


def main():
    here = Path(__file__).resolve().parent
    source = json.loads((here / "physical_capacity_tier_v1.json").read_text())
    pilot = json.loads((here / "runs/h262144_b16_s6_physical_capacity_2080_v1/result.json").read_text())
    cpu = here / "runs/compile_physical_capacity_tier_C02_v1/kernel"
    evidence = {}
    for mode, task in (
        ("state", "task_20261023_101533_4702752956"),
        ("service", "task_20261023_101534_47081128521"),
    ):
        run = here / f"runs/h262144_b16_s6_physical_capacity_tier_{mode}_v1"
        raw = json.loads((run / "result.json").read_text())
        assert raw["passed"] and raw["stage"] == "complete" and raw["device_execution"]
        assert raw["case"]["id"] == "C02" and not raw["performance_eligible"]
        require_same_sdk(raw["shared_toolchain"], pilot["shared_toolchain"])
        assert raw["source_hashes"]["candidate"] == source["files"]
        assert raw["policies"]["candidate"]["PLAN_TABLE_COLUMNS"] == 2080
        for check in raw["checks"].values():
            assert all(check["guards"].values())
            require_strict(check["numerical"])
        matching = []
        for build in (run / "build_output").glob("decode_csa_spmd_*"):
            paths = [p.relative_to(cpu) for p in cpu.rglob("*.cpp")]
            if all(
                (build / p).is_file() and executable((build / p).read_text()) == executable((cpu / p).read_text())
                for p in paths
            ):
                matching.append(str(build))
        assert matching, (mode, "executed candidate must match the compiled tier")
        row = dict(
            task_id=task,
            source_report=str(run / "result.json"),
            passed=True,
            performance_measured=False,
            matching_compiled_artifacts=matching,
        )
        if mode == "state":
            state = raw["state_validation"]
            assert state["passed"] and not state["performance_measured"]
            assert len(state["metadata_cases"]) == 20 and len(state["continuous_steps"]) == 4
            for case in state["metadata_cases"] + state["continuous_steps"]:
                assert case["passed"]
                for side in ("reference", "candidate"):
                    checked = case["checks"][side]
                    assert checked["passed"] and checked["same_addresses"]
                    assert all(checked["guards"].values()) and all(checked["generation"].values())
                    assert all(checked["geometry"].values())
                    require_strict(checked["numerical"])
                plan = case["checks"]["candidate"]["physical_plan"]
                assert plan["passed"] and plan["independent_physical_mapping"]
                assert plan["selected_capacity"] == 2080 and 2048 < plan["table_columns"] <= 2080
                if "compressed_page_holes" in case:
                    assert len(plan["all_compressed_blocks_invalid_but_swa_valid"]) == 16
            row.update(
                metadata_cases=[c["name"] for c in state["metadata_cases"]],
                continuous_steps=[c["name"] for c in state["continuous_steps"]],
                strict_check_groups=48,
                independent_plan_mapping_every_candidate_replay=True,
                selected_negative_and_oob_pages_with_recovery=True,
                compressed_table_columns=state["metadata_cases"][0]["checks"]["candidate"]["physical_plan"][
                    "table_columns"
                ],
            )
        else:
            service = raw["service_validation"]
            assert service["passed"] and service["workspace_reused_across_capture"]
            assert service["policy_constexprs"]["PLAN_TABLE_COLUMNS"] == 2080
            require_strict(service["numerical"])
            assert all(v["status"] == "PASS" for v in service["guards"].values())
            assert len(service["graph_replays"]) == 2
            for replay in service["graph_replays"]:
                assert replay["python_not_resubmitted"] and replay["same_addresses"] and replay["contract_allows"]
                require_strict(replay["numerical"])
                assert all(v["status"] == "PASS" for v in replay["guards"].values())
            empty = service["empty"]
            assert all(empty[k] for k in ("handler_called", "no_submission", "graph_rejected", "cache_unchanged"))
            assert empty["output"]["status"] == "PASS"
            fallback = service["fallback"]
            assert fallback["not_eligible"] and fallback["no_submission"]
            assert all(
                v["status"] == "PASS"
                for v in (fallback["output"], *fallback["cache"].values(), *fallback["guards"].values())
            )
            row.update(
                runtime_type=service["runtime_type"],
                policy_constexprs=service["policy_constexprs"],
                eager_and_two_graph_replays_exact=True,
                workspace_reused=True,
                empty_and_native_fallback_passed=True,
            )
        evidence[mode] = row
    report = dict(
        candidate="physical_capacity_tier_v1",
        passed=True,
        adopted=False,
        cpu_codegen="physical_capacity_tier_v1_codegen.json",
        checks=evidence,
        limitation="Adoption still requires production source integration and contract tests",
    )
    with (here / "physical_capacity_tier_v1_validation.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print("PASS: C02 20 metadata cases, four continuous steps and canonical service")


if __name__ == "__main__":
    main()
