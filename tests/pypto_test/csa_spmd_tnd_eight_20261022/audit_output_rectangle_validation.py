# SPDX-License-Identifier: Apache-2.0
"""Require code identity and complete state/service evidence before adoption."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_priority_candidate import match_build
from analyze_priority_phase import executable
from audit_capacity_validation import require_strict
from measurement_provenance import require_same_sdk


def codegen(here, manifest):
    rows = {}
    for case, entry in manifest["builds"].items():
        candidate, retained = [here / entry[k] for k in ("candidate", "retained")]
        assert json.loads((candidate.parent / "result.json").read_text())["full_codegen_passed"]
        abi = lambda root: json.loads((root / "compiled_meta.json").read_text())["kernel_abi"]
        assert abi(candidate) == abi(retained)
        paths = sorted(p.relative_to(candidate) for p in candidate.rglob("*.cpp"))
        assert len(paths) == 4
        assert paths == sorted(p.relative_to(retained) for p in retained.rglob("*.cpp"))
        hashes = {}
        for path in paths:
            body = executable((candidate / path).read_text())
            hashes[str(path)] = hashlib.sha256(body.encode()).hexdigest()
            if case != "C05" or path.parts[0] == "orchestration":
                assert body == executable((retained / path).read_text()), (case, path)
        rows[case] = dict(
            candidate=str(candidate),
            retained=str(retained),
            all_four_cpp_equal_to_retained=case != "C05",
            kernel_abi_equal=True,
            executable_cpp_sha256=hashes,
        )
    return dict(
        passed=True,
        cases=rows,
        existing_C05_resource_and_static_tail_audit="output_rectangle_v3_codegen.json",
        comparison="Exact text except whole // pto: source-location comment lines",
        performance_eligible=False,
        device_execution=False,
    )


def runtime(here, manifest):
    source = json.loads((here / manifest["source"]).read_text())
    pilot = json.loads((here / "runs/h131072_b24_s6_output_rectangle_v3/result.json").read_text())
    cpu = here / manifest["builds"]["C05"]["candidate"]
    rows = {}
    for mode, task in manifest["tasks"].items():
        run = here / task["output"]
        raw = json.loads((run / "result.json").read_text())
        assert raw["passed"] and raw["stage"] == "complete" and raw["device_execution"]
        assert raw["case"]["id"] == "C05" and not raw["performance_eligible"]
        require_same_sdk(raw["shared_toolchain"], pilot["shared_toolchain"])
        assert Path(raw["sources"]["candidate"]).resolve() == Path(source["source"]).resolve()
        assert raw["policies"]["candidate"]["OA_ROWS"] == 160
        assert raw["policies"]["candidate"]["OA_COLS"] == 192
        if mode == "state":
            assert raw["source_hashes"]["candidate"] == source["files"]
        else:
            # This established service harness records the frozen path and CPP;
            # it does not claim to have recorded launch-time source hashes.
            assert {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(source["source"]).glob("*.py")
            } == source["files"]
        for checked in raw["checks"].values():
            assert checked["guards"] and all(checked["guards"].values())
            require_strict(checked["numerical"])
        row = dict(
            task_id=task["task_id"],
            source_report=str(run / "result.json"),
            matching_compiled_artifacts=match_build(run, cpu),
            launch_time_source_hashes_recorded=mode == "state",
            passed=True,
        )
        if mode == "state":
            state = raw["state_validation"]
            assert state["passed"] and not state["performance_measured"]
            assert len(state["metadata_cases"]) == task["expected_metadata_cases"]
            assert len(state["continuous_steps"]) == task["expected_continuous_steps"]
            for case in state["metadata_cases"] + state["continuous_steps"]:
                assert case["passed"]
                for checked in case["checks"].values():
                    assert checked["passed"] and checked["same_addresses"]
                    assert all(checked["guards"].values()) and all(checked["generation"].values())
                    assert all(checked["geometry"].values())
                    require_strict(checked["numerical"])
            row.update(
                metadata_cases=[c["name"] for c in state["metadata_cases"]],
                continuous_steps=[c["name"] for c in state["continuous_steps"]],
                strict_check_groups=40,
                same_addresses_geometry_generation_guards=True,
            )
        else:
            service = raw["service_validation"]
            assert service["passed"] and service["workspace_reused_across_capture"]
            assert service["policy_constexprs"]["OA_ROWS"] == 160
            assert service["policy_constexprs"]["OA_COLS"] == 192
            require_strict(service["numerical"])
            assert all(v["status"] == "PASS" for v in service["guards"].values())
            assert len(service["graph_replays"]) == 2
            for replay in service["graph_replays"]:
                assert replay["python_not_resubmitted"] and replay["same_addresses"] and replay["contract_allows"]
                require_strict(replay["numerical"])
                assert all(v["status"] == "PASS" for v in replay["guards"].values())
            empty, fallback = service["empty"], service["fallback"]
            assert all(empty[k] for k in ("handler_called", "no_submission", "graph_rejected", "cache_unchanged"))
            assert empty["output"]["status"] == "PASS"
            assert fallback["not_eligible"] and fallback["no_submission"]
            assert all(
                v["status"] == "PASS"
                for v in (fallback["output"], *fallback["cache"].values(), *fallback["guards"].values())
            )
            row.update(
                runtime_type=service["runtime_type"],
                eager_and_two_graph_replays_exact=True,
                workspace_reused=True,
                empty_and_native_fallback_passed=True,
            )
        rows[mode] = row
    return dict(passed=True, adopted=False, checks=rows, performance_eligible=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("codegen", "runtime"))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "output_rectangle_v3_validation_tasks.json").read_text())
    report = (codegen if args.mode == "codegen" else runtime)(here, manifest)
    with (here / f"output_rectangle_v3_{args.mode}_validation.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(f"PASS: output rectangle {args.mode} validation")


if __name__ == "__main__":
    main()
