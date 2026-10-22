# SPDX-License-Identifier: Apache-2.0
"""Freeze a diagnostic BSH/current-TND/candidate comparison on one device."""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from prepare_score_owner_harness import replace_once


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=here / "runs/three_way_harness_v116")
    parser.add_argument("--control", type=Path, default=here / "runs/borrowed_combined_candidate_v106")
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(exist_ok=False)
    control = args.control.resolve()
    shutil.copytree(control, target / "control_spmd", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    names = (
        "run_probe.py",
        "bsh_probe.py",
        "service_probe.py",
        "service_metadata_replay.py",
        "metadata_matrix.py",
        "tnd_performance.py",
        "submit_probe.py",
        "run_probe.sh",
    )
    originals = {name: (here / name).read_text() for name in names}
    revised = dict(originals)
    text = replace_once(
        originals["run_probe.sh"], "mode = sys.argv[4]\n", f"mode = sys.argv[4]\nharness = Path({str(target)!r})\n"
    )
    for name in names:
        old = f'repo / "tests/pypto_test/csa_spmd_tnd_20261020/{name}"'
        if old in text:
            text = replace_once(text, old, f'harness / "{name}"')
    text = replace_once(
        text, 'repo / "tests/pypto_test/dsv4_csa_single_layer.py"', 'harness / "dsv4_csa_single_layer.py"'
    )
    text = replace_once(
        text,
        "baseline = Path(sys.argv[6]).resolve()\n",
        'shutil.copytree(harness / "control_spmd", snapshot / "control_spmd",\n'
        '                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))\n'
        "baseline = Path(sys.argv[6]).resolve()\n",
    )
    revised["run_probe.sh"] = text
    revised["submit_probe.py"] = replace_once(
        originals["submit_probe.py"],
        '    path = here / "jobs.json"\n',
        f"    path = Path({str(here / 'jobs.json')!r})\n",
    )
    preparation = """                control_path = Path(__file__).resolve().parent / "control_spmd"
                control_package = load_binding(control_path, "tnd_three_way_control")
                control = control_package.prepared(
                    root, max_query_tokens=args.query_tokens,
                    max_seq_len=args.history + args.query_tokens,
                    allocator=lambda name, shape, dtype, device: allocate("control/" + name, shape, dtype, device),
                )
                restore(fixture)
                for name, value in control.allocations.items():
                    if name not in ("signal", "ffts"):
                        value.fill_(37)
                known_tables = candidate_tables | baseline_tables
                with compilation_probe(args.output / "control_compile_diagnostics", wait_backoff=not args.raw_wait):
                    control(config=options)
                torch.npu.synchronize()
                control_tables = (
                    set(Path(os.environ["PYPTO_PROG_BUILD_DIR"]).glob("decode_csa_spmd_*/kernel_config.py"))
                    - known_tables
                )
                assert len(control_tables) == 1
                control_actual = collect_state(fixture, output, root["idx_topk"])
                control_checks = {k: compare_tensor(value, eager[k], 0, 0) for k, value in control_actual.items()}
                control_checks["scores"] = compare_tensor(root["idx_topk_scores"].cpu(), eager_scores, 0, 0)
                control_checks.update(guard_checks(fixture))
                if not all(value["status"] == "PASS" for value in control_checks.values()):
                    raise AssertionError("Three-way current TND control differs before timing")
                def guarded_padding(entry):
                    checks = workspace_padding_checks(entry)
                    checks.update({
                        "allocation/" + name: bool((storage[:guard].cpu() == 37).all()
                                                  and (storage[-guard:].cpu() == 37).all())
                        for name, (storage, guard) in allocations.items()
                    })
                    return checks
                report["control_initial_workspace_guards"] = guarded_padding(control)
                if not all(report["control_initial_workspace_guards"].values()):
                    raise AssertionError("Three-way workspace guard changed before timing")
                report["control_checks"] = control_checks
                report["comparison_reference"] = {
                    "kind": "three_way_diagnostic", "formal_bsh_gate_eligible": False,
                    "control_source": str(control_path), "control_label": "TNDcombinedv106",
                    "purpose": "Resolve inconsistent B1 incremental and BSH comparisons without selecting faster runs",
                }
"""
    revised["run_probe.py"] = replace_once(
        originals["run_probe.py"],
        "            if args.l1_windows:\n                from tnd_performance import profile\n",
        "            if args.l1_windows:\n                from tnd_performance import profile\n\n" + preparation,
    )
    revised["run_probe.py"] = replace_once(
        revised["run_probe.py"],
        "                    padding_checks=workspace_padding_checks,\n",
        "                    padding_checks=guarded_padding,\n",
    )
    revised["run_probe.py"] = replace_once(
        revised["run_probe.py"],
        "                    shared_workspace=args.l1_shared_workspace,\n",
        "                    shared_workspace=args.l1_shared_workspace,\n"
        '                    extra_entries={"tnd_control_v106": control},\n'
        '                    extra_configs={"tnd_control_v106": next(iter(control_tables))},\n',
    )
    text = replace_once(
        originals["tnd_performance.py"],
        "    padding_checks=None,\n",
        "    padding_checks=None,\n    extra_entries=None,\n    extra_configs=None,\n",
    )
    text = replace_once(
        text,
        '    entries = {"spmd_tnd": candidate, "spmd_original": baseline}\n',
        '    entries = {"spmd_tnd": candidate, "spmd_original": baseline}\n'
        "    if extra_entries:\n"
        "        if shared_workspace:\n"
        '            raise ValueError("Three-way diagnostic requires independent private workspaces")\n'
        "        entries.update(extra_entries)\n"
        "        configs = dict(configs, **extra_configs)\n"
        "        if len(entries) != 3 or windows != 6:\n"
        '            raise ValueError("Three-way diagnostic requires three entries and six windows")\n',
    )
    text = replace_once(
        text,
        "            order = list(entries) if sample % 2 == 0 else list(reversed(entries))\n",
        "            if extra_entries:\n"
        "                from itertools import permutations\n"
        "                orders = tuple(permutations(entries))\n"
        "                order = orders[sample % len(orders)]\n"
        "            else:\n"
        "                order = list(entries) if sample % 2 == 0 else list(reversed(entries))\n",
    )
    revised["tnd_performance.py"] = text
    for name, text in revised.items():
        if name.endswith(".py"):
            ast.parse(text)
        (target / name).write_text(text)
    shutil.copy2(here.parent / "dsv4_csa_single_layer.py", target / "dsv4_csa_single_layer.py")
    subprocess.run(["bash", "-n", str(target / "run_probe.sh")], check=True)
    # The existing strict verification and per-window timing body are retained.
    before, after = [ast.parse(s) for s in (originals["tnd_performance.py"], revised["tnd_performance.py"])]
    for name in ("verify",):
        nodes = [
            next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == name) for t in (before, after)
        ]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1])
    timed_loops = [
        next(
            n
            for n in ast.walk(t)
            if isinstance(n, ast.For) and ast.unparse(n.target) == "name" and ast.unparse(n.iter) == "order"
        )
        for t in (before, after)
    ]
    assert ast.dump(timed_loops[0]) == ast.dump(timed_loops[1])
    report = dict(
        harness=str(target),
        control=str(control),
        formal_bsh_gate_eligible=False,
        device_execution=False,
        windows=6,
        order="All six permutations of three entries, once each",
        strict_verifier_ast_unchanged=True,
        per_window_timing_body_ast_unchanged=True,
        source_manifest={
            str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in target.rglob("*")
            if p.is_file()
        },
    )
    report_path = here / (target.name + ".json")
    with report_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
