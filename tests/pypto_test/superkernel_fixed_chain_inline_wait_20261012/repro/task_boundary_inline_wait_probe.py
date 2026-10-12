# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Measure an inline clean-tail Wait while retaining the original callback fallback."""

import argparse
import sys
from pathlib import Path

import task_boundary_acceptance as acceptance
from pypto.backend import pto_backend
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.runtime_builder import RuntimeBuilder

HELPER = r"""
#include "common/platform_config.h"
static __aicore__ __attribute__((always_inline)) inline
uint32_t pypto_boundary_dispatch_token() {
    uint32_t value;
    asm volatile("MOV %0, DATA_MAIN_BASE\n" : "=l"(value));
    return value;
}
static __aicore__ __attribute__((always_inline)) inline
uint32_t pypto_boundary_gate_token() {
    uint64_t value;
    asm volatile("MOV %0, DATA_MAIN_BASE\n" : "=l"(value));
    return static_cast<uint32_t>(value >> 32);
}
static __aicore__ __attribute__((always_inline)) inline
int pypto_inline_clean_boundary(__gm__ int64_t* args, __gm__ GlobalContext* ctx) {
    auto* previous = reinterpret_cast<__gm__ GlobalContext*>(ctx->predecessor_context);
    bool clean = (ctx->scalar_prefix & SCALAR_PREFIX_TAGGED_ARGS) != 0 &&
                 ctx->scalar_overlap_trace == 0;
    if (previous != nullptr) {
        const auto kind = static_cast<AsyncTailKind>(previous->async_tail_kind);
        clean = clean && previous->scalar_overlap_trace == 0 &&
                (kind == AsyncTailKind::RuntimeBarrierAll ||
                 kind == AsyncTailKind::RuntimeCompiledPollClean ||
                 kind == AsyncTailKind::RuntimeDirectPollClean);
    }
    if (!clean) {
        using WaitFn = int (*)(__gm__ int64_t*);
        return reinterpret_cast<WaitFn>(ctx->wait_predecessor_fn)(args);
    }
    // Identical ordering to the runtime callback. Only the indirect call and
    // legacy/diagnostic branches are absent from this clean tagged path.
    if (previous != nullptr) pipe_barrier(PIPE_ALL);
    const uint32_t predecessor_id = ctx->predecessor_reg_task_id;
    const uint32_t own_id = ctx->reg_task_id;
    dcci(reinterpret_cast<__gm__ char*>(ctx) + offsetof(GlobalContext, async_tail_kind),
         SINGLE_CACHE_LINE, CACHELINE_OUT);
    auto* own = reinterpret_cast<__gm__ DispatchPayload*>(
        reinterpret_cast<__gm__ char*>(ctx) - offsetof(DispatchPayload, global_context));
    const bool gated = own->src_payload != 0;
    if (previous != nullptr) set_cond(static_cast<uint32_t>(MAKE_FIN_VALUE(predecessor_id)));
    if (gated) {
        while (pypto_boundary_gate_token() != own_id) {
            if (pypto_boundary_dispatch_token() == AICORE_EXIT_SIGNAL) return 0;
        }
    }
    if (pypto_boundary_dispatch_token() == AICORE_EXIT_SIGNAL) return 0;
    set_cond(static_cast<uint32_t>(MAKE_ACK_VALUE(own_id)));
    return 1;
}
"""


def install_adapter():
    original_includes = KernelCompiler.get_incore_include_dirs

    def incore_includes(compiler):
        return [*original_includes(compiler), str(compiler.platform_dir / "include")]

    KernelCompiler.get_incore_include_dirs = incore_includes
    original = pto_backend._SCALAR_PREFIX_WAIT_ADAPTER
    anchor = """        using WaitFn = int (*)(__gm__ int64_t*);
        auto wait = reinterpret_cast<WaitFn>(ctx->wait_predecessor_fn);
        ready = wait(args);"""
    assert original.count(anchor) == 1
    pto_backend._SCALAR_PREFIX_WAIT_ADAPTER = HELPER + original.replace(
        anchor, "        ready = pypto_inline_clean_boundary(args, ctx);"
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("action", choices=("compile", "run", "diagnostics"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime_root.resolve()
    RuntimeBuilder._CACHE_DIR = runtime / "runtime/cache"
    RuntimeBuilder._LIB_DIR = runtime / "runtime/lib"
    if args.action in ("compile", "diagnostics"):
        install_adapter()
        (root / "wait_adapter.cpp").write_text(pto_backend._SCALAR_PREFIX_WAIT_ADAPTER)
    if args.action == "compile":
        acceptance.artifact(root / "compiled", "early", compile_now=True)
    elif args.action == "diagnostics":
        import pytest  # noqa: PLC0415

        raise SystemExit(pytest.main(remaining))
    else:
        sys.argv = [sys.argv[0], *remaining]
        acceptance.main()


if __name__ == "__main__":
    main()
