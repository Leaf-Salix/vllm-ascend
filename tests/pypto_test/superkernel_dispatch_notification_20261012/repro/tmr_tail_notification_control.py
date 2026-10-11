# SPDX-License-Identifier: Apache-2.0
"""Frozen notification helper used for the stage077 comparison."""

from pypto.pypto_core import ir as _ir_core


def _compiled_late_poll(func: _ir_core.Function) -> tuple[str, str]:
    """Keep DMA code in the task image, reachable through a private entry tag."""
    scratch = dict(func.attrs).get("task_boundary_scratch_base")
    if scratch is None:
        return "", ""
    cube = func.func_type == _ir_core.FunctionType.AIC
    # Match PTO-ISA TExpandS: generate the constant in reserved local memory
    # without issuing a GM read. The original pipe joins below also order the
    # fill (MTE2 on AIC, V on AIV) before the notification write.
    local_fill = (
        f"auto* scratch = reinterpret_cast<__cbuf__ uint16_t*>({scratch});\n"
        "    create_cbuf_matrix(scratch, int64_t{0x10002}, static_cast<uint16_t>(expected));"
        if cube
        else f"auto* scratch = reinterpret_cast<__ubuf__ uint16_t*>({scratch});\n"
        "    set_mask_count();\n"
        "    set_vector_mask(0, 32);\n"
        "    vector_dup(scratch, static_cast<uint16_t>(expected), 0, 1, 1, 8, 8);\n"
        "    set_mask_norm();\n"
        "    set_vector_mask(~uint64_t{0}, ~uint64_t{0});"
    )
    engines = ("MTE2", "MTE1", "M", "FIX") if cube else ("MTE2", "V")
    joins = "\n".join(
        f"    {op}_flag(PIPE_{engine}, PIPE_MTE3, EVENT_ID7);" for op in ("set", "wait") for engine in engines
    )
    copy_out = "copy_cbuf_to_gm" if cube else "copy_ubuf_to_gm"
    helper = f"""
static __aicore__ __attribute__((noinline)) void pypto_poll_local_tail(__gm__ int64_t* args) {{
    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    auto* notice = reinterpret_cast<__gm__ TaskBoundaryNotification*>(ctx->task_boundary_notification);
    auto* value = reinterpret_cast<volatile __gm__ uint64_t*>(notice->value);
    dcci(notice->value, SINGLE_CACHE_LINE);
    const uint64_t expected = *value == 0 ? ~uint64_t{{0}} : 0;
    {local_fill}
{joins}
    pipe_barrier(PIPE_MTE3);
    {copy_out}(reinterpret_cast<__gm__ uint16_t*>(notice->value), scratch, 0, 1, 2, 0, 0);
    const uint32_t own = ctx->reg_task_id;
    const bool trace = ctx->scalar_overlap_trace != 0;
    if (trace) ctx->late_handoff_scratch |= uint64_t{{1}} << 32;
    for (;;) {{
        uint32_t next;
        asm volatile("MOV %0, DATA_MAIN_BASE" : "=l"(next) : : "memory");
        if (next != own && next != 0x7ffffffdu) {{
            return;
        }}
        dcci(notice->value, SINGLE_CACHE_LINE);
        if (*value == expected) return;
    }}
}}
"""
    entry = """    // Internal tail callback; this tag never executes the task body twice.
    if ((reinterpret_cast<uint64_t>(args) & uint64_t{2}) != 0) {
        auto* tail_args = reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<uint64_t>(args) & ~uint64_t{3});
        pypto_poll_local_tail(tail_args);
        return;
    }
"""
    return helper, entry
