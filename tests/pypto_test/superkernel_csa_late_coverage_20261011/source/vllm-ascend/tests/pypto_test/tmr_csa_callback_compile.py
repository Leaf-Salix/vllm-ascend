# SPDX-License-Identifier: Apache-2.0
"""Isolate tail callback entry cost and inlining on the current private SDK."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import tmr_csa_full_128k_b16_compile as compile_demo
from pypto.backend import pto_backend


@contextmanager
def callback_variant(mode):
    original = pto_backend._compiled_late_poll

    def callback(func):
        helper, entry = original(func)
        if not helper:
            return helper, entry
        if mode == "inline":
            return helper.replace(
                "__attribute__((noinline)) void pypto_poll_local_tail",
                "__attribute__((always_inline)) inline void pypto_poll_local_tail",
            ), entry
        if mode == "notice32":
            transfer = ", 0, 1, 2, 0, 0"
            assert helper.count(transfer) == 2
            # Keep each core's exclusive 64-byte cache line and scratch.
            # The completion protocol observes only the first 64-bit word.
            return helper.replace(transfer, ", 0, 1, 1, 0, 0"), entry
        if mode == "recheck":
            marker = "    auto* scratch = reinterpret_cast<"
            assert helper.count(marker) == 1
            check = """    uint32_t dispatched;
    asm volatile("MOV %0, DATA_MAIN_BASE" : "=l"(dispatched) : : "memory");
    if (dispatched != ctx->reg_task_id && dispatched != 0x7ffffffdu) return;
"""
            return helper.replace(marker, check + marker), entry
        assert mode == "empty"
        # Diagnostic only: retain the runtime call/entry, omit notification and
        # polling. The executor still drains the full tail before publishing FIN.
        return (
            """
static __aicore__ __attribute__((noinline)) void pypto_poll_local_tail(__gm__ int64_t* args) {
    asm volatile("" : : "l"(args) : "memory");
}
""",
            entry,
        )

    try:
        pto_backend._compiled_late_poll = callback
        yield
    finally:
        pto_backend._compiled_late_poll = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--variant", choices=("empty", "inline", "recheck", "notice32"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Reuse only this worktree's unchanged, exact-SDK artifacts.
    reused = {}
    references = ("baseline", "mixed") if args.variant == "notice32" else ("baseline", "once", "mixed")
    for name in references:
        target = (args.reuse / name).resolve(strict=True)
        destination = args.output / name
        if not destination.exists():
            destination.symlink_to(target, target_is_directory=True)
        assert destination.resolve() == target
        reused[name] = str(target)
    (args.output / "reused_artifacts.json").write_text(json.dumps(reused, indent=2) + "\n")
    for mode in (args.variant,) if args.variant else ("empty", "inline"):
        compile_demo.PACKAGES[mode] = compile_demo.PACKAGES["mixed"]
        with callback_variant(mode):
            compile_demo.compile_variant(mode, args.output / mode)


if __name__ == "__main__":
    main()
