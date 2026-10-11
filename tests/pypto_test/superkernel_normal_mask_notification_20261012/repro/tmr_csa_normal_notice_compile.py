# SPDX-License-Identifier: Apache-2.0
"""Compile a fixed-mask AIV notification against frozen, same-SDK controls."""

import argparse
import json
from pathlib import Path

import tmr_csa_full_128k_b16_compile as demo
from tmr_csa_scalar_prefix_layout_audit import inspect_kernels
from tmr_csa_score_fair_compile import PACKAGES as SCORE_PACKAGES
from tmr_csa_untraced_notice_compile import canonical_inline_names

PACKAGES = {
    "plain_score": SCORE_PACKAGES["plain_score"],
    "counted_notice": SCORE_PACKAGES["early_score"],
    "normal_notice": SCORE_PACKAGES["early_score"],
}
COUNTED = (
    "    set_mask_count();\n"
    "    set_vector_mask(0, 16);\n"
    "    vector_dup(scratch, expected, 0, 1, 1, 8, 8);\n"
    "    set_mask_norm();\n"
)
NORMAL = "    set_mask_norm();\n    set_vector_mask(0, 0xffff);\n    vector_dup(scratch, expected, 1, 1, 1, 8, 8);\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    demo.PACKAGES.update(PACKAGES)
    for name, source_name in (("plain_score", "plain_score"), ("counted_notice", "untraced_notice")):
        source = (args.controls / source_name).resolve()
        assert (source / "build").is_dir(), source
        target = args.output / name
        if target.exists():
            assert target.resolve() == source
        else:
            target.symlink_to(source, target_is_directory=True)
    candidate = args.output / "normal_notice"
    demo.compile_variant("normal_notice", candidate)
    baseline = args.output / "counted_notice"
    assert (candidate / "program.py").read_text() == (baseline / "program.py").read_text()
    audit = inspect_kernels(candidate / "build")
    assert all(not row["unsafe_pipe_ops"] for row in audit)
    changes = []
    for path in sorted((candidate / "build/kernels").rglob("*.cpp")):
        relative = path.relative_to(candidate / "build")
        before = (baseline / "build" / relative).read_text()
        after = path.read_text()
        count = before.count(COUNTED)
        assert canonical_inline_names(before.replace(COUNTED, NORMAL)) == canonical_inline_names(after), relative
        changes.append({"file": str(relative), "normal_mask_callbacks": count})
    assert len(changes) == 73 and sum(row["normal_mask_callbacks"] for row in changes) > 0
    (args.output / "source_comparison.json").write_text(json.dumps(changes, indent=2) + "\n")
    (args.output / "prefix_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
