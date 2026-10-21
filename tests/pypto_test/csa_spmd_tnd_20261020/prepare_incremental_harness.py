# SPDX-License-Identifier: Apache-2.0
"""Freeze a paired probe against adopted TND, without changing the BSH gate."""

import argparse
import ast
import difflib
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from prepare_score_owner_harness import replace_once


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, default=here)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unused-score-arena", action="store_true")
    args = parser.parse_args()
    source, reference, target = (p.resolve() for p in (args.harness, args.reference, args.output))
    reference_adopted = True
    if reference == here / "runs/output_work_balance_native_candidate_v4":
        reference_kind = "adopted_tnd_output_balance_v4"
        reference_label = "TND output_balance v4 (incremental comparison only)"
    elif reference == here / "runs/borrowed_combined_candidate_v106":
        reference_kind = "adopted_tnd_borrowed_combined_v106"
        reference_label = "TND combined v106 (incremental comparison only)"
        accepted = json.loads((here / "borrowed_combined_cleanup_v106.json").read_text())
        for name, expected in accepted["source_manifest"].items():
            if hashlib.sha256((reference / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Accepted combined-v106 source changed: {name}")
    elif reference == here / "runs/qb_wide_candidate_v108b":
        reference_kind = "validated_qb_wide_v108b"
        reference_label = "QB-wide v108b (experimental reference, incremental comparison only)"
        reference_adopted = False
        accepted = json.loads((here / "qb_wide_v108b.json").read_text())
        for name, expected in accepted["source_manifest"].items():
            if hashlib.sha256((reference / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Validated QB-wide source changed: {name}")
        for case in ("qb_wide_b1_s1_s6_matrix_v110", "qb_wide_t41_r12_matrix_v110"):
            evidence = json.loads((here / "runs" / case / "result.json").read_text())
            if not (evidence["passed"] and evidence["stage"] == "complete"):
                raise ValueError(f"QB-wide reference requires completed correctness: {case}")
    else:
        raise ValueError(
            "This probe's reference label is reserved for adopted output_balance v4, combined v106, "
            "or the validated experimental QB-wide v108b"
        )
    target.mkdir(parents=True, exist_ok=False)
    names = (
        "run_probe.py",
        "service_probe.py",
        "service_metadata_replay.py",
        "tnd_performance.py",
        "bsh_probe.py",
        "metadata_matrix.py",
        "submit_probe.py",
        "run_probe.sh",
    )
    originals = {name: (source / name).read_text() for name in names}
    revised = dict(originals)
    frozen = target / "reference"
    shutil.copytree(reference, frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    reference_info = {
        "kind": reference_kind,
        "label": reference_label,
        "source": str(reference),
        "frozen_source": str(frozen),
        "formal_bsh_gate_eligible": False,
        "adopted": reference_adopted,
        "files": {
            str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(frozen.rglob("*"))
            if p.is_file()
        },
    }
    (target / "comparison_reference.json").write_text(json.dumps(reference_info, indent=2) + "\n")
    name = "run_probe.sh"
    text = originals[name]
    if source == here:
        text = replace_once(text, "mode = sys.argv[4]\n", f"mode = sys.argv[4]\nharness = Path({str(target)!r})\n")
        for filename in names:
            old = f'repo / "tests/pypto_test/csa_spmd_tnd_20261020/{filename}"'
            if old in text:
                text = replace_once(text, old, f'harness / "{filename}"')
        text = replace_once(
            text, 'repo / "tests/pypto_test/dsv4_csa_single_layer.py"', 'harness / "dsv4_csa_single_layer.py"'
        )
        shutil.copyfile(here.parent / "dsv4_csa_single_layer.py", target / "dsv4_csa_single_layer.py")
    else:
        text = replace_once(text, f"harness = Path({str(source)!r})", f"harness = Path({str(target)!r})")
        for path in source.glob("*.py"):
            if path.name not in names:
                shutil.copyfile(path, target / path.name)
    if "    baseline_source=${bsh_roots[2]}\n" in text:
        text = replace_once(
            text, "    baseline_source=${bsh_roots[2]}\n", f"    baseline_source={shlex.quote(str(frozen))}\n"
        )
    else:
        # Shared-SDK launchers read only the historical BSH operator path.
        text = replace_once(
            text, "SOURCE_PATH\n)\n", f"SOURCE_PATH\n)\n    baseline_source={shlex.quote(str(frozen))}\n"
        )
    text = replace_once(
        text,
        'source "$workspace/env-dsv4-0251rc1.sh"\n',
        "[[ $bsh_mode == true && $mode == probe && $native_toolchain == true ]]\n"
        'source "$workspace/env-dsv4-0251rc1.sh"\n',
    )
    text = replace_once(
        text,
        'if sys.argv[7]:\n    shutil.copyfile(sys.argv[7], snapshot / "bsh_baseline_manifest.json")\n',
        'shutil.copyfile(harness / "comparison_reference.json",\n'
        '                snapshot / "comparison_reference.json")\n',
    )
    revised[name] = text
    name = "bsh_probe.py"
    text = replace_once(
        originals[name],
        "    if module.WAIT_BACKOFF_TICKS != 150:\n"
        '        raise ValueError("The frozen accepted BSH baseline must use 150 ticks")\n'
        "    original = module.rewrite_wait_source\n",
        "    def original(source):\n"
        '        raise ValueError("Both incremental TND sides must use native 150-tick lowering")\n',
    )
    text = replace_once(
        text,
        "            result = json.loads(result_path.read_text())\n",
        "            result = json.loads(result_path.read_text())\n"
        "            reference = json.loads((args.wait_policy_source.parent.parent /\n"
        '                                         "comparison_reference.json").read_text())\n'
        '            for relative, expected in reference["files"].items():\n'
        "                actual = args.wait_policy_source.parent / relative\n"
        "                if hashlib.sha256(actual.read_bytes()).hexdigest() != expected:\n"
        '                    raise RuntimeError(f"Frozen TND reference changed: {relative}")\n'
        '            result["comparison_reference"] = reference\n',
    )
    text = replace_once(
        text,
        '"BSH fixed150; native TND fixed150 unless the separately recorded fast-poll experiment is enabled"',
        '"Incremental TND pair; fixed150 on both sides. Not a BSH performance-gate result."',
    )
    text = replace_once(text, "passed=(native or legacy)", "passed=native and not legacy")
    revised[name] = text
    if args.unused_score_arena:
        name = "run_probe.py"
        revised[name] = replace_once(
            originals[name],
            "    return checks\n",
            '    if getattr(entry.policy, "short_half_score_ub", False):\n'
            '        checks["score_arena_unused_untouched"] = bool((entry.buffers["score_arena"].cpu() == 37).all())\n'
            "    return checks\n",
        )
    name = "submit_probe.py"
    if source == here:
        revised[name] = replace_once(
            originals[name], '    path = here / "jobs.json"\n', f"    path = Path({str(here / 'jobs.json')!r})\n"
        )
    patch = []
    for name, text in revised.items():
        if name.endswith(".py"):
            ast.parse(text)
        (target / name).write_text(text)
        patch.extend(
            difflib.unified_diff(
                originals[name].splitlines(True), text.splitlines(True), fromfile="a/" + name, tofile="b/" + name
            )
        )
    subprocess.run(["bash", "-n", str(target / "run_probe.sh")], check=True)
    untouched = (
        "run_probe.py",
        "metadata_matrix.py",
        "tnd_performance.py",
        "service_probe.py",
        "service_metadata_replay.py",
    )
    if args.unused_score_arena:
        untouched = tuple(name for name in untouched if name != "run_probe.py")
    assert all((target / name).read_bytes() == (source / name).read_bytes() for name in untouched)
    report = {
        "harness": str(target),
        "original_harness": str(source),
        "reference": reference_info,
        "unchanged_checks_and_timing": list(untouched),
        "shell_syntax_passed": True,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file()},
        "device_execution": False,
        "unused_score_arena_check": args.unused_score_arena,
    }
    (here / (target.name + ".patch")).write_text("".join(patch))
    (here / (target.name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
