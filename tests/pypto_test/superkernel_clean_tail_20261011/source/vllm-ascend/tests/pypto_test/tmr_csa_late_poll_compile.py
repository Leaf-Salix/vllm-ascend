# SPDX-License-Identifier: Apache-2.0
"""Compile ordinary, one-shot, mutable-Set and clean-tail CSA on one SDK."""

import argparse
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import tmr_csa_full_128k_b16_compile as compile_demo
from pypto.backend import pto_backend


@contextmanager
def old_tail(mode):
    original_poll = pto_backend._compiled_late_poll
    original_orch = pto_backend._codegen_core.generate_orchestration
    original_wrapper = pto_backend._generate_kernel_wrapper

    def orch_without_poll(*args, **kwargs):
        result = original_orch(*args, **kwargs)
        fields = {
            key: getattr(result, key)
            for key in (
                "code",
                "func_name_to_id",
                "func_name_to_core_type",
                "func_name_to_signature",
                "orchestration_signature",
            )
        }
        fields["code"] = re.sub(r"^.*\.set_runtime_compiled_poll(?:_mask)?\([^\n]*\);\n", "", result.code, flags=re.M)
        return SimpleNamespace(**fields)

    def mutable_wrapper(func, *args, **kwargs):
        code = original_wrapper(func, *args, **kwargs)
        scratch = dict(func.attrs).get("task_boundary_scratch_base")
        if scratch is None:
            return code
        marker = (
            "    // The compiler transfers the ordinary tail PIPE_ALL to the executor.\n"
            "    // No signal or mutable GM completion state is needed at this boundary.\n"
            "    (void)args;"
        )
        assert code.count(marker) == 1
        return code.replace(
            marker,
            (
                "    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);\n"
                f"    ctx->late_handoff_scratch = {scratch};\n"
                "    ctx->async_tail_kind = static_cast<uint32_t>(AsyncTailKind::RuntimeCompiledPoll);"
            ),
        )

    try:
        pto_backend._codegen_core.generate_orchestration = orch_without_poll
        if mode == "once":
            pto_backend._compiled_late_poll = lambda func: ("", "")
        else:
            assert mode == "written"
            pto_backend._generate_kernel_wrapper = mutable_wrapper
        yield
    finally:
        pto_backend._compiled_late_poll = original_poll
        pto_backend._codegen_core.generate_orchestration = original_orch
        pto_backend._generate_kernel_wrapper = original_wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compile_demo.compile_variant("baseline", args.output / "baseline")
    # Same frozen MIX program and scratch/events. Change only tail protocol.
    for mode in ("once", "written"):
        compile_demo.PACKAGES[mode] = compile_demo.PACKAGES["mixed"]
        with old_tail(mode):
            compile_demo.compile_variant(mode, args.output / mode)
    compile_demo.compile_variant("mixed", args.output / "mixed")


if __name__ == "__main__":
    main()
