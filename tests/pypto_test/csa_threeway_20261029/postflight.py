# SPDX-License-Identifier: Apache-2.0
"""生成文件落盘后验收替换覆盖、真实回放、计时与逐请求结果，不初始化设备。"""

import importlib.util
import json


def check_reference_configuration(actual, native):
    keys = (
        "TP",
        "DP",
        "EP",
        "task_device",
        "common_framework",
        "bank",
        "batch_per_rank",
        "prefix",
        "output_tokens",
        "HCA",
        "gpu_blocks_override",
        "runtime_ring_heap_bytes",
        "event_mode",
        "eplb",
        "atomic_add",
        "capture_sizes",
        "precision_only",
        "request_wave_size",
    )
    if native["arm"] != "native" or native.get("startup_only", False):
        raise ValueError("Reference must be a complete Native generation, not startup-only")
    for key in keys:
        if actual[key] != native[key]:
            raise ValueError(f"Native reference configuration differs: {key}")


def acceptance_events(path):
    return {row["key"]: row["events"] for row in map(json.loads, path.read_text().splitlines())}


def timing_inputs(value):
    return [
        (row["actual_requests"], row["actual_tokens"], row["padded_tokens"], row["positions"])
        for row in value["observation"][0]["threeway_timing"]["samples"]
    ]


def check_rank(data, arm, precision_only=False):
    errors = {"coverage": [], "timing": [], "precision": []}
    implementations = data.get("attention_implementations", [])
    observations = data.get("observation", [])
    if len(implementations) != 1 or len(observations) != 1:
        errors["coverage"].append("TP1 requires one implementation/observation record")
        return errors
    layers, observed = implementations[0], observations[0]
    csa = [row for row in layers if row["kind"] == "csa"]
    hca = [row for row in layers if row["kind"] == "hca"]
    expected = "native" if arm == "native" else "pto"
    if len(csa) != 21 or len(hca) != 20 or len({row["layer"] for row in layers}) != 41:
        errors["coverage"].append("requires 21 distinct CSA and 20 HCA layers")
    if any(row["implementation"] != expected for row in csa) or any(row["implementation"] != "native" for row in hca):
        errors["coverage"].append("CSA/HCA implementation mismatch")
    bindings = observed.get("tnd_bindings", [])
    if arm == "native":
        if bindings or any(row.get("runtime_class") is not None for row in layers):
            errors["coverage"].append("Native must have no PTO runtime")
    else:
        if {row["layer"] for row in bindings} != {row["layer"] for row in csa} or len(bindings) != 21:
            errors["coverage"].append("not all 21 CSA runtimes were observed")
        if any("deepseek_v4_flash_csa_spmd_tnd" not in (row.get("runtime_class") or "") for row in csa):
            errors["coverage"].append("unexpected CSA runtime package")
        for row in bindings:
            checks = row.get("decode_graph_checks", {})
            # The first T40 target step may legally submit PTO outside FULL.
            # Replay/fallback is checked below only for the FULL240 routes.
            if not row.get("bindings") or checks.get("allowed", 0) <= 0:
                errors["coverage"].append(f"layer {row['layer']}: absent binding or graph replay")
        captured = observed.get("capture_time_selection", {})
        if any(not captured.get(row["name"], {}).get(f"pto_tokens{t}", 0) for row in csa for t in (6, 240)):
            errors["coverage"].append("not all CSA layers captured PTO T6 and T240")
        routes = observed.get("tnd_execution_routes", [])
        full = [
            row
            for row in routes
            if (0 < row["actual_requests"] <= 40 if precision_only else row["actual_requests"] == 40)
            and row["padded_tokens"] in ((6, 240) if precision_only else (240,))
            and row["requested_mode"] == "FULL"
        ]
        if not full or any(not row["all_csa_graph_replayed"] or row["python_resubmitted"] for row in full):
            errors["coverage"].append("FULL graph did not replay all CSA layers without resubmission")
        if precision_only:
            ordinary = [row for row in routes if row["actual_requests"] > 0 and row["requested_mode"] != "FULL"]
            if any(not row.get("all_csa_submitted", False) for row in ordinary):
                errors["coverage"].append("ordinary decode did not submit all CSA layers")
    if not observed.get("threeway_timing", {}).get("complete"):
        errors["timing"].append("missing complete consecutive FULL240 timing window")
    requests = data.get("requests", [])
    if (
        len(requests) != 40
        or len({row["key"] for row in requests}) != 40
        or any(len(row["output_token_ids"]) != 192 or not row["finished"] for row in requests)
    ):
        errors["precision"].append("requires 40 distinct finished requests of 192 tokens")
    return errors


def validate(source, root, bank, arm, reference=None):
    report = {"arm": arm, "coverage_errors": [], "timing_errors": [], "precision_errors": []}
    experiment = json.loads((root / "experiment.json").read_text())
    dp = experiment["DP"]
    precision_only = experiment.get("precision_only", False)
    assert dp in (8, 16) and experiment["EP"] == dp
    if reference is not None:
        check_reference_configuration(experiment, json.loads((reference / "experiment.json").read_text()))
    report["DP"] = report["EP"] = dp
    for rank in range(dp):
        try:
            data = json.loads((root / f"rank{rank}.mixed.json").read_text())
            for kind, errors in check_rank(data, arm, precision_only).items():
                report[f"{kind}_errors"].extend(f"rank{rank}: {error}" for error in errors)
            if reference is not None:
                native = json.loads((reference / f"rank{rank}.mixed.json").read_text())
                actual = [(row["key"], row["output_token_ids"]) for row in data["requests"]]
                expected = [(row["key"], row["output_token_ids"]) for row in native["requests"]]
                if actual != expected:
                    report["precision_errors"].append(f"rank{rank}: output tokens differ from Native")
                if acceptance_events(root / f"request_stats_rank{rank}.jsonl") != acceptance_events(
                    reference / f"request_stats_rank{rank}.jsonl"
                ):
                    report["precision_errors"].append(f"rank{rank}: acceptance events differ from Native")
                if timing_inputs(data) != timing_inputs(native):
                    report["timing_errors"].append(f"rank{rank}: timing input/admission trajectory differs")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            report["precision_errors"].append(f"rank{rank}: {error}")
    try:
        path = source / "tests/pypto_test/low_acceptance_20261001/mixed.py"
        spec = importlib.util.spec_from_file_location("threeway_mixed_analysis", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        acceptance = module.collect(root, bank, 40, 192)
        report["acceptance"] = acceptance
        if not acceptance["qualified"] or acceptance["total_preemptions"] != 0:
            report["precision_errors"].append("acceptance protocol failed or preemption was not zero")
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        report["precision_errors"].append(f"acceptance analysis: {error}")
    report["performance_qualified"] = not report["timing_errors"]
    report["precision_only"] = precision_only
    criteria = ("coverage", "precision") if precision_only else ("coverage", "timing", "precision")
    report["passed"] = not any(report[f"{kind}_errors"] for kind in criteria)
    (root / "postflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
