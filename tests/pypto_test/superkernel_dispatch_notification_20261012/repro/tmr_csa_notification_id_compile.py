# SPDX-License-Identifier: Apache-2.0
"""Compile identical CSA bodies with old-value or dispatch-ID notifications."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import tmr_csa_full_128k_b16_compile as demo
from pypto.backend import pto_backend
from tmr_csa_scalar_prefix_layout_audit import inspect_kernels
from tmr_csa_score_fair_compile import PACKAGES as SCORE_PACKAGES
from tmr_csa_score_fair_compile import prepare_plain
from tmr_tail_notification_control import _compiled_late_poll as original_notification

PACKAGES = {
    "plain_score": SCORE_PACKAGES["plain_score"],
    "original_notice": SCORE_PACKAGES["early_score"],
    "dispatch_notice": SCORE_PACKAGES["early_score"],
}


@contextmanager
def notification_variant(original: bool):
    current = pto_backend._compiled_late_poll
    if original:
        pto_backend._compiled_late_poll = original_notification
    try:
        yield
    finally:
        pto_backend._compiled_late_poll = current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare_plain(args.output)
    demo.PACKAGES.update(PACKAGES)
    for name in PACKAGES:
        with notification_variant(name == "original_notice"):
            demo.compile_variant(name, args.output / name)
        if name != "plain_score":
            audit = inspect_kernels(args.output / name / "build")
            assert all(not row["unsafe_pipe_ops"] for row in audit)
            (args.output / f"{name}_prefix_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
