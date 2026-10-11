# SPDX-License-Identifier: Apache-2.0
"""Compile a fair control and compiler-proved notification join candidate."""

import argparse
import json
import re
from pathlib import Path
from unittest.mock import patch

import pypto.backend.pto_backend as backend
import tmr_csa_full_128k_b16_compile as demo
from tmr_csa_scalar_prefix_layout_audit import inspect_kernels
from tmr_csa_score_fair_compile import PACKAGES as SCORE_PACKAGES
from tmr_csa_score_fair_compile import prepare_plain

PACKAGES = {
    "plain_score": SCORE_PACKAGES["plain_score"],
    "early_full": SCORE_PACKAGES["early_score"],
    "early_joined": SCORE_PACKAGES["early_score"],
}


def keep_original_joins(_name, helper, _code):
    # The same compiler still emits its proof metadata. Acknowledge that
    # contract while intentionally retaining the prior complete callback.
    return helper, "#define PYPTO_TASK_BOUNDARY_JOINS_V1 1\n", set()


def normalize_names(text):
    text = re.sub(r"^\s*// pto:.*\n", "", text, flags=re.MULTILINE)
    text = text.replace("#define PYPTO_TASK_BOUNDARY_JOINS_V1 1\n", "")
    identifiers = {}
    return re.sub(
        r"_inline[0-9]+",
        lambda match: identifiers.setdefault(match[0], f"_inline{len(identifiers)}"),
        text,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare_plain(args.output)
    demo.PACKAGES.update(PACKAGES)
    for name in () if args.audit_only else PACKAGES:
        if name == "early_full":
            with patch.object(backend, "_apply_task_boundary_joins", keep_original_joins):
                demo.compile_variant(name, args.output / name)
        else:
            demo.compile_variant(name, args.output / name)
    reports = {}
    for name in ("early_full", "early_joined"):
        reports[name] = inspect_kernels(args.output / name / "build")
        assert all(not row["unsafe_pipe_ops"] for row in reports[name])
    (args.output / "prefix_audit.json").write_text(json.dumps(reports, indent=2) + "\n")
    changes = []
    root = args.output / "early_joined/build/kernels"
    for path in root.rglob("*.cpp"):
        relative = path.relative_to(root)
        text = normalize_names(path.read_text())
        before = normalize_names((args.output / "early_full/build/kernels" / relative).read_text())
        # Apart from callback joins, the independently emitted kernel and
        # wrapper must remain identical. Do not infer this from performance.
        helper = r"static __aicore__ __attribute__\(\(noinline\)\) void pypto_poll_local_tail\([^\n]*\{.*?\n\}"
        a, b = re.search(helper, before, re.S), re.search(helper, text, re.S)
        if not a or not b:
            assert before == text
            continue
        assert before[:a.start()] + before[a.end():] == text[:b.start()] + text[b.end():], relative
        if a[0] != b[0]:
            removed = [line for line in a[0].splitlines() if line not in b[0].splitlines()]
            assert removed and all("_flag(PIPE_" in line and "PIPE_MTE3, EVENT_ID7" in line for line in removed)
            assert all(line in a[0].splitlines() for line in b[0].splitlines())
            changes.append({"path": str(relative), "removed": removed})
    assert changes, "candidate must reduce real CSA callbacks"
    (args.output / "changed_callbacks.json").write_text(json.dumps(changes, indent=2) + "\n")
    print(f"Reduced {len(changes)} callback bodies; all task-body code remains identical.")


if __name__ == "__main__":
    main()
