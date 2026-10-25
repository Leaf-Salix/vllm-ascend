# SPDX-License-Identifier: Apache-2.0
"""Freeze a STATE-only adapter of the established 12-case + 4-step validator."""

import ast
import difflib
import json
import shutil
from pathlib import Path

from prepare import replace_once


def remove_function(source, name):
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(True)
    del lines[node.lineno - 1 : node.end_lineno]
    return "".join(lines)


def adapt_runner(source):
    source = remove_function(source, "legacy_export_l1")
    source = replace_once(source, "from complete_l1_v2 import export_l1\n", "")
    source = replace_once(
        source,
        (
            '    if args.family != "tnd" or args.control_mpmd or args.baseline_only or args.s'
            "tate_validation or args.service_validation:\n"
            '        raise ValueError("This frozen harness measures an independent TND-SPMD c'
            'andidate only")\n'
        ),
        '    if (args.family != "tnd" or not args.state_validation or args.control_mpmd or args.baseline_only\n'
        "            or args.service_validation or args.boundary_query_length or args.dep_gen\n"
        "            or not args.uncompared_candidate):\n"
        '        raise ValueError("This private harness only validates same-address TND state; no performance mode")\n',
    )
    source = replace_once(
        source,
        'args.case not in ("h131072_b16", "h131072_b28")',
        'args.case not in ("h131072_b16_s6", "h131072_b24_s6")',
    )
    source = source.replace(
        "状态矩阵只从长B16/B28构造SPMD/MPMD两侧功能场景", "状态矩阵只从C04/C05构造冻结版/候选两侧功能场景"
    )
    source = replace_once(
        source,
        '        case["table_history"] += 128\n',
        '        case["table_history"] = (case["table_history"] or case["history_label"]) + 128\n',
    )
    source = replace_once(
        source,
        '        "sampling": {"warmup": 3, "valid_windows": 10, "replays_per_window": 1},\n',
        '        "sampling": None,\n        "performance_measured": False,\n        "performance_eligible": False,\n',
    )
    source = source.replace('"enable_chip_swimlane": 1', '"enable_chip_swimlane": 0')
    source = source.replace("enable_chip_swimlane=1,", "enable_chip_swimlane=0,")
    # The source already returns immediately after STATE validation. Remove the
    # subsequent profiling path entirely, so future flag mistakes cannot time it.
    begin = source.index("                for sample in range(3):\n")
    end = source.index("    except BaseException:\n", begin)
    source = (
        source[:begin]
        + '                raise AssertionError("STATE-only harness must return through state validation")\n'
        + source[end:]
    )
    source = source.replace(
        "固定真 TND 输入的 SPMD / MPMD 单卡对照；只使用完整 L1。", "同址metadata与连续状态功能验证；不采性能。"
    )
    source = source.replace(
        "分配设备；fixture/权重恢复、校验与 D2H 全部在测量区间之外。",
        "分配设备；内部signal/offset/count不直接观察，所有检查均针对外部可见合同。",
    )
    assert "begin_dfx" not in source and "end_dfx" not in source and "export_l1(" not in source
    return source


def adapt_validator(source):
    source = source.replace("SPMD/MPMD两侧同址根", "冻结版/候选两侧同址根")
    source = remove_function(source, "generation")
    source = replace_once(
        source,
        '    result = {"passed": False, "performance_measured": False, "metadata_cases": [], "continuous_steps": []}\n',
        (
            "    mutable_roots = {\n"
            '        "x_out", "idx_topk", "idx_topk_scores", "kv_cache", "cmp_kv",\n'
            '        "idx_native_kv_cache", "compress_state", "inner_compress_state",\n'
            "    }\n"
            "    readonly_roots = {\n"
            "        name: (value, value.cpu()) for name, value in root.items()\n"
            "        if isinstance(value, torch.Tensor) and name not in mutable_roots and nam"
            "e not in METADATA_NAMES\n"
            "    }\n"
            "    root_addresses = {name: value.data_ptr() for name, value in root.items() if "
            "isinstance(value, torch.Tensor)}\n"
            "    result = {\n"
            '        "passed": False, "performance_measured": False, "metadata_cases": [], "c'
            'ontinuous_steps": [],\n'
            '        "internal_observation": {\n'
            '            "signal_generation": "not_observed_internal_workspace",\n'
            '            "geometry_offsets_counts": "not_observed_internal_workspace",\n'
            '            "internal_scratch_redzones": "not_observed_internal_workspace",\n'
            '            "protocol_pass_claim": False,\n'
            "        },\n"
            '        "readonly_roots": sorted(readonly_roots),\n'
            '        "required_gate": "external output/topk/scores/full cache-state bytes equ'
            'al frozen reference; guards and addresses preserved",\n'
            "    }\n"
        ),
    )
    source = replace_once(
        source,
        "                max_seq_len=max(h + q for h, q in zip(histories, queries)),\n",
        "                max_seq_len=max(h + q for h, q in zip(histories, queries)),\n"
        '                compressed_table_columns=root["cmp_block_table"].shape[1],\n',
    )
    source = replace_once(
        source,
        "        prior = generation(entries[side])\n",
        '        readonly_before = {name: value.cpu() for name, (value, _) in fixture["readonly"].items()}\n',
    )
    begin = source.index("        after_generation = generation(entries[side])\n")
    end = source.index("        values = {\n", begin)
    source = source[:begin] + source[end:]
    source = replace_once(
        source,
        (
            '        protected.update({"metadata/" + name: torch.equal(root[name].cpu(), valu'
            "e) for name, value in metadata.items()})\n"
        ),
        (
            '        protected.update({"metadata/" + name: torch.equal(root[name].cpu().view('
            "torch.uint8), value.view(torch.uint8)) for name, value in metadata.items()})\n"
            '        protected.update({"readonly_fixture/" + name: torch.equal(value.cpu().vi'
            "ew(torch.uint8), readonly_before[name].view(torch.uint8)) for name, (value, _) i"
            'n fixture["readonly"].items()})\n'
            '        protected.update({"readonly_root/" + name: torch.equal(value.cpu().view('
            "torch.uint8), initial_value.view(torch.uint8)) for name, (value, initial_value) "
            "in readonly_roots.items()})\n"
        ),
    )
    begin = source.index("        geometry = {}\n")
    end = source.index("        return (\n", begin)
    source = (
        source[:begin]
        + (
            "        same_roots = root_addresses == {name: value.data_ptr() for name, value i"
            "n root.items() if isinstance(value, torch.Tensor)}\n"
        )
        + source[end:]
    )
    source = replace_once(
        source,
        '{"guards": protected, "same_addresses": same, "generation": protocol, "geometry": geometry}',
        '{"guards": protected, "same_addresses": same and same_roots,\n'
        '             "internal_observation": result["internal_observation"],\n'
        '             "external_scratch_guard_count": sum(owner == side for owner, _ in guards)}',
    )
    source = replace_once(
        source,
        '            checked["numerical"] = compare(values, expected)\n',
        '            checked["numerical"] = compare(values, expected)\n'
        '            checked["cache_state_bytes"] = {\n'
        '                name: torch.equal(raw.view(torch.uint8), next_caches["reference"][name].view(torch.uint8))\n'
        "                for name, raw in next_caches[side].items()\n"
        "            }\n",
    )
    source = replace_once(
        source,
        '                and all(checked["generation"].values())\n'
        '                and all(checked["geometry"].values())\n',
        '                and bool(checked["guards"])\n'
        '                and bool(checked["cache_state_bytes"])\n'
        '                and all(checked["cache_state_bytes"].values())\n'
        '                and {"output", "topk", "scores"} <= checked["numerical"].keys()\n',
    )
    source = replace_once(
        source,
        '                and all(v["status"] == "PASS" for v in checked["numerical"].values())\n',
        (
            '                and all(v["status"] == "PASS" and v.get("bit_equal") is True for'
            ' v in checked["numerical"].values())\n'
        ),
    )
    assert ".allocations" not in source
    assert "def generation" not in source and 'buffers[branch + "_offsets"]' not in source
    return source


def main():
    here = Path(__file__).resolve().parent
    source_runner = here / "frozen_v2/run_pto.py"
    source_validator = here.parent / "csa_spmd_tnd_eight_20261022/runs/priority_state_harness_v1/state_replay.py"
    target = here / "runs/internal_state_harness_v1"
    target.mkdir()
    saved = target / "source_validator"
    saved.mkdir()
    changes = {}
    for name, original, adapter in (
        ("run_pto.py", source_runner, adapt_runner),
        ("state_replay.py", source_validator, adapt_validator),
    ):
        before = original.read_text()
        after = adapter(before)
        ast.parse(after)
        (target / name).write_text(after)
        (saved / name).write_text(before)
        patch = "".join(
            difflib.unified_diff(
                before.splitlines(True), after.splitlines(True), fromfile=str(original), tofile=str(target / name)
            )
        )
        (target / (name + ".patch")).write_text(patch)
        changes[name] = {
            "source": str(original),
            "preserved_source": str(saved / name),
            "exact_patch": str(target / (name + ".patch")),
        }
    for name in ("scenarios.py", "measurement_provenance.py", "expected_sdk.json", "baseline.json"):
        shutil.copy2(here / "frozen_v2" / name, target / name)
    launcher = (here / "launch_fused_init.sh").read_text()
    launcher = replace_once(
        launcher, '"$here/frozen_v2/run_pto.py"', '"$here/runs/internal_state_harness_v1/run_pto.py"'
    )
    launcher = replace_once(
        launcher,
        '--uncompared-candidate --candidate-init-layout fused --device "$TASK_DEVICE"',
        '--uncompared-candidate --state-validation --device "$TASK_DEVICE"',
    )
    old_out = 'out="$here/runs/$' + "{case_name}_$" + '{attempt}"'
    new_out = 'out="$here/runs/$' + "{case_name}_state_$" + '{attempt}"'
    launcher = replace_once(
        launcher, old_out, 'case "$case_name" in h131072_b16_s6|h131072_b24_s6) ;; *) exit 2 ;; esac\n' + new_out
    )
    launcher_path = here / "launch_internal_state.sh"
    launcher_path.write_text(launcher)
    launcher_path.chmod(0o755)
    report = {
        "status": "CPU_AST_gate_pending",
        "harness": str(target),
        "launcher": str(launcher_path),
        "source_evidence": changes,
        "metadata_cases": 12,
        "continuous_steps": 4,
        "preserved_schedule": "original priority_state_harness_v1 schedule and four steps byte-for-byte",
        "changes": [
            "new internal binding requires no allocator or scratch access",
            "internal generations/geometry explicitly unobserved and excluded from pass logic",
            "real compressed table columns passed into select_policy",
            "full external cache/state byte equality added to existing typed slot checks",
            "all nonmutable Native root tensors and current fixture readonly tensors protected",
            "no performance collection code; chip swimlane disabled",
            "C04/C05 names and nullable table_history handled",
        ],
        "performance_measured": False,
        "performance_eligible": False,
        "device_execution": False,
    }
    (here / "internal_state_harness_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
