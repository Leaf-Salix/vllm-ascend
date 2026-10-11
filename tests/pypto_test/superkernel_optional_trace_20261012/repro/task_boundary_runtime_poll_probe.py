# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Exercise runtime-owned late polling after the original handoff probe misses.

Reuse the two-task inputs, guard checks and graph replay driver. Instrumentation
records dispatch generations and the existing eight-word tail trace only; it
does not supply notification memory, insert DMA or alter dispatch decisions.
"""

import re
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from pypto import ir
from pypto.backend import pto_backend
from pypto.ir import TensorType

if __package__:
    from . import task_boundary_late_arrival_probe as probe
else:
    import task_boundary_late_arrival_probe as probe


@contextmanager
def instrument(mode, cube):
    original = pto_backend._generate_kernel_wrapper
    original_orch = pto_backend._codegen_core.generate_orchestration

    def orch_without_poll(*args, **kwargs):
        r = original_orch(*args, **kwargs)
        fields = {
            key: getattr(r, key)
            for key in (
                "code",
                "func_name_to_id",
                "func_name_to_core_type",
                "func_name_to_signature",
                "orchestration_signature",
            )
        }
        fields["code"] = re.sub(
            r"^.*\.set_runtime_compiled_poll(?:_mask)?\([^\n]*\);\n", "", r.code, flags=re.M
        )
        return SimpleNamespace(**fields)

    def wrap(func, *args, **kwargs):
        if mode == "once":
            attrs = dict(func.attrs)
            attrs.pop("task_boundary_scratch_base", None)
            func = ir.Function(
                func.name,
                list(zip(func.params, func.param_directions)),
                func.return_types,
                func.body,
                func.span,
                type=func.func_type,
                level=func.level,
                role=func.role,
                attrs=attrs,
                requires_runtime_binding=func.requires_runtime_binding,
                ir_stage=func.ir_stage,
            )
        code = original(func, *args, **kwargs)
        tensors = [p for p in func.params if isinstance(p.type, TensorType)]
        marker = "    // Forward to ptoas-generated function\n"
        assert code.count(marker) == 1
        if func.name == "late_producer":
            code = pto_backend._with_compiled_poll_trace(code)
            trace = next(p for p in tensors if p.name_hint.startswith("a_trace")).name_hint
            setup = f"""    auto* row = reinterpret_cast<__gm__ uint64_t*>({trace})
        + get_block_idx(args) * 32;
    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    ctx->scalar_overlap_trace = reinterpret_cast<uint64_t>(row + 24);
    ctx->scalar_overlap_start = get_sys_cnt();
    row[3] = ctx->reg_task_id;
    dcci(row, SINGLE_CACHE_LINE, CACHELINE_OUT);
"""
            return code.replace(marker, setup + marker)
        assert func.name == "late_consumer"
        trace = next(p for p in tensors if p.name_hint.startswith("b_trace")).name_hint
        setup = """    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    const uint64_t own = ctx->reg_task_id;
    const uint64_t prior = ctx->predecessor_reg_task_id;
    const uint64_t flags = ctx->scalar_prefix;
    const uint64_t local_handoff = ctx->predecessor_context != 0;
"""
        code = code.replace(marker, setup + marker)
        insert = f"""    auto* row = reinterpret_cast<__gm__ uint64_t*>({trace}) + get_block_idx(args) * 16;
    row[8] = own;
    row[9] = prior;
    row[10] = flags;
    row[11] = local_handoff;
    dcci(row + 8, SINGLE_CACHE_LINE, CACHELINE_OUT);
"""
        end = code.rfind("}\n")
        return code[:end] + insert + code[end:]

    pto_backend._generate_kernel_wrapper = wrap
    if mode == "once":
        pto_backend._codegen_core.generate_orchestration = orch_without_poll
    try:
        yield
    finally:
        pto_backend._generate_kernel_wrapper = original
        pto_backend._codegen_core.generate_orchestration = original_orch


def validate(a, b, expected_a, expected_b, at, bt, tile_rows, seed, steps):
    blocks = len(at) - 1
    assert torch.equal(a[: blocks * tile_rows], expected_a), "producer output"
    assert torch.equal(b[: blocks * tile_rows], expected_b), "consumer output"
    assert torch.isnan(a[blocks * tile_rows :]).all() and torch.isnan(b[blocks * tile_rows :]).all()
    assert at[-1] == [-1] * 32 and bt[-1] == [-1] * 16
    producers = {(v[24], v[3]): (i, v) for i, v in enumerate(at[:-1])}
    assert len(producers) == blocks
    handoffs, late, polled = [], [], []
    for i, v in enumerate(at[:-1]):
        assert 0 < v[25] <= v[26] <= v[29] <= v[31] <= v[27]
        assert v[28] & 255 == 7
        if v[28] & 256:
            polled.append(i)
    for i, v in enumerate(bt[:-1]):
        state = seed + i
        for _ in range(steps):
            state = (state * 17 + 3) % 65521
        assert v[4] == state and v[5] == int(expected_a[i * tile_rows, 0])
        assert 0 < v[1] <= v[2] <= v[3]
        if v[11]:
            ai, prev = producers[v[0], v[9]]
            assert v[1] >= prev[26] and v[3] >= prev[27]
            entry = {"producer": ai, "consumer": i, "remaining_full_tail_ticks": prev[30]}
            handoffs.append(entry)
            if prev[28] & 512:
                late.append(entry)
    # Every reported late handoff must be tied to the same core/generation's
    # successor. Tail duration includes notification DMA, not only old compute.
    assert len(late) == sum(bool(v[28] & 512) for v in at[:-1])
    return {"handoffs": handoffs, "late_handoffs": late, "polled_after_first_miss": polled}


if __name__ == "__main__":
    probe.instrument = instrument
    probe.validate = validate
    probe.main()
