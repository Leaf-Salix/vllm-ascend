# SPDX-License-Identifier: Apache-2.0
"""Measure removing diagnostic accounting from an uninstrumented CSA callback."""

import argparse
import json
import re
from contextlib import contextmanager
from pathlib import Path

import tmr_csa_full_128k_b16_compile as demo
from pypto.backend import pto_backend
from tmr_csa_scalar_prefix_layout_audit import inspect_kernels
from tmr_csa_score_fair_compile import PACKAGES as SCORE_PACKAGES
from tmr_csa_score_fair_compile import prepare_plain

PACKAGES = {
    "plain_score": SCORE_PACKAGES["plain_score"],
    "dispatch_notice": SCORE_PACKAGES["early_score"],
    "untraced_notice": SCORE_PACKAGES["early_score"],
}
TRACE_ACCOUNTING = (
    "    const bool trace = ctx->scalar_overlap_trace != 0;\n"
    "    if (trace) ctx->late_handoff_scratch |= uint64_t{1} << 32;\n"
)


@contextmanager
def untraced_notice():
    """Experiment only: the frozen CSA supplies no diagnostic trace pointer."""
    original = pto_backend._compiled_late_poll

    def without_trace_accounting(func):
        helper, entry = original(func)
        if helper:
            if TRACE_ACCOUNTING in helper:
                assert helper.count(TRACE_ACCOUNTING) == 1, func.name
                helper = helper.replace(TRACE_ACCOUNTING, "")
            else:
                # The retained compiler now omits it by default. Keep this
                # experiment usable with both the pre-change and new base.
                assert "scalar_overlap_trace" not in helper, func.name
        return helper, entry

    pto_backend._compiled_late_poll = without_trace_accounting
    try:
        yield
    finally:
        pto_backend._compiled_late_poll = original


def canonical_inline_names(code):
    """Ignore bijective inline-instance renaming across compiler processes."""
    names = {}

    def rename(match):
        name = match.group()
        if name not in names:
            names[name] = re.sub(r"_inline[0-9]+", "_inline", name) + str(len(names))
        return names[name]

    return re.sub(r"\b[A-Za-z_]\w*_inline[0-9]+", rename, code)


def audit_compiled(output, controls):
    candidate = output / "untraced_notice"
    baseline = output / "dispatch_notice"
    assert (candidate / "program.py").read_text() == (baseline / "program.py").read_text()
    audit = inspect_kernels(candidate / "build")
    assert all(not row["unsafe_pipe_ops"] for row in audit)
    (output / "untraced_prefix_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    files = []
    for path in sorted((candidate / "build/kernels").rglob("*.cpp")):
        relative = path.relative_to(candidate / "build")
        before = (baseline / "build" / relative).read_text()
        after = path.read_text()
        removed = before.count(TRACE_ACCOUNTING)
        assert canonical_inline_names(before.replace(TRACE_ACCOUNTING, "")) == canonical_inline_names(after), relative
        assert "->scalar_overlap_trace =" not in after, relative
        files.append({"file": str(relative), "trace_clauses_removed": removed})
    assert len(files) == 73
    assert sum(row["trace_clauses_removed"] for row in files) == 71
    (output / "source_comparison.json").write_text(
        json.dumps(
            {
                "same_program_ir": True,
                "only_generated_change": "remove optional trace load and flag update from callback",
                "ignored": "bijective compiler-generated inline-instance renaming",
                "controls": str(controls.resolve()),
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--controls", type=Path, required=True, help="Stage077 artifacts built with the same Simpler SDK"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare_plain(args.output)
    demo.PACKAGES.update(PACKAGES)
    for name in ("plain_score", "dispatch_notice"):
        source = (args.controls / name).resolve()
        assert (source / "build").is_dir(), source
        target = args.output / name
        if target.exists():
            assert target.resolve() == source
        else:
            target.symlink_to(source, target_is_directory=True)
    with untraced_notice():
        demo.compile_variant("untraced_notice", args.output / "untraced_notice")
    audit_compiled(args.output, args.controls)


if __name__ == "__main__":
    main()
