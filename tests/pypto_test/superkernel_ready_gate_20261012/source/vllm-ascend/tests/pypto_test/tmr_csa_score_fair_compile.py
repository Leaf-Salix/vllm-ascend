# SPDX-License-Identifier: Apache-2.0
"""Compare local handoff with identical Score arithmetic on both sides."""

import argparse
import difflib
import json
import re
import shutil
from pathlib import Path

import tmr_csa_full_128k_b16_compile as demo
from tmr_csa_scalar_prefix_layout_audit import inspect_kernels

PACKAGES = {
    "plain_score": "deepseek_v4_flash_csa_score_plain_control",
    "early_score": "deepseek_v4_flash_csa_score_scalar_plan",
}
BOUNDARY_LINE = re.compile(r"^[ \t]+pl\.system\.(wait_pre_task_end|set_next_task_start)\(\)\n", re.MULTILINE)


def prepare_plain(output):
    root = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/pypto"
    source, target = (root / PACKAGES[key] for key in ("early_score", "plain_score"))
    assert source.is_dir(), "retained frozen stage071 Score package is required"
    changes, patches = {}, []
    expected = {}
    for path in source.rglob("*.py"):
        text = path.read_text()
        updated, count = BOUNDARY_LINE.subn("", text)
        relative = path.relative_to(source)
        expected[relative] = updated
        assert "pl.system.wait_pre_task_end(" not in updated
        assert "pl.system.set_next_task_start(" not in updated
        if count:
            changes[str(relative)] = {
                name: text.count(f"pl.system.{name}()")
                for name in ("wait_pre_task_end", "set_next_task_start")
            }
            patches.extend(difflib.unified_diff(
                text.splitlines(keepends=True), updated.splitlines(keepends=True),
                fromfile=f"a/{relative}", tofile=f"b/{relative}",
            ))
    assert changes
    if target.exists():
        assert all((target / name).read_text() == text for name, text in expected.items())
    else:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name, text in expected.items():
            (target / name).write_text(text)
    (output / "plain_vs_early_operator.diff").write_text("".join(patches))
    (output / "removed_boundary_statements.json").write_text(json.dumps(changes, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare_plain(args.output)
    demo.PACKAGES.update(PACKAGES)
    for name in PACKAGES:
        demo.compile_variant(name, args.output / name)
    report = inspect_kernels(args.output / "early_score/build")
    assert all(not row["unsafe_pipe_ops"] for row in report)
    (args.output / "early_prefix_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Identical arithmetic; only local boundary statements removed from the ordinary control.")


if __name__ == "__main__":
    main()
