# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Publish a scheduler's notification lines with one completion barrier."""

import task_boundary_contract_cache_probe as cached
import task_boundary_normal_state_probe as normal


def changes(root):
    result = cached.changes(root)
    path = root / "src/common/platform/include/aicpu/cache_maintenance.h"
    old = path.read_text()
    anchor = "void flush_range_impl(const void *addr, size_t size);"
    assert old.count(anchor) == 1
    new = old.replace(anchor, anchor + "\nvoid flush_ranges_impl(const void *const *addresses, size_t count, size_t bytes);")
    anchor = "inline void cache_flush_range(const void *addr, size_t size) { aicpu_cache_maintenance::flush_range_impl(addr, size); }"
    assert new.count(anchor) == 1
    new = new.replace(anchor, anchor + """

// All ranges are owned and initialized by the caller before publication.
inline void cache_flush_ranges(const void *const *addresses, size_t count, size_t bytes) {
    aicpu_cache_maintenance::flush_ranges_impl(addresses, count, bytes);
}
""")
    result[path] = (old, new)

    path = root / "src/common/platform/onboard/aicpu/cache_ops.cpp"
    old = path.read_text()
    anchor = "\n#else\n"
    assert old.count(anchor) == 1
    new = old.replace(anchor, """
void flush_ranges_impl(const void *const *addresses, size_t count, size_t bytes) {
    if (count == 0 || bytes == 0) return;
    constexpr uintptr_t line = 64;
    for (size_t i = 0; i < count; ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        const uintptr_t start = address & ~(line - 1);
        const uintptr_t end = (address + bytes + line - 1) & ~(line - 1);
        for (uintptr_t p = start; p < end; p += line) {
            __asm__ __volatile__("dc cvac, %0" ::"r"(p) : "memory");
        }
    }
    __asm__ __volatile__("dsb sy" ::: "memory");
    __asm__ __volatile__("isb" ::: "memory");
}
""" + anchor)
    anchor = "void flush_range_impl(const void * /* addr */, size_t /* size */) {}"
    assert new.count(anchor) == 1
    new = new.replace(anchor, anchor + "\nvoid flush_ranges_impl(const void *const *, size_t, size_t) {}")
    result[path] = (old, new)

    path = root / "src/common/platform/sim/aicpu/cache_ops.cpp"
    old = path.read_text()
    anchor = "void flush_range_impl(const void * /* addr */, size_t /* size */) {}"
    assert old.count(anchor) == 1
    result[path] = (old, old.replace(anchor, anchor + "\nvoid flush_ranges_impl(const void *const *, size_t, size_t) {}"))

    path = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp"
    old = path.read_text()
    new = old
    ranges = (
        ("void SchedulerContext::assign_own_clusters(", "// Abort the run on a handshake failure"),
        ("int32_t SchedulerContext::post_handshake_init(Runtime *runtime,", "void SchedulerContext::deinit()"),
    )
    for begin, end in ranges:
        start = new.index(begin)
        stop = new.index(end, start)
        scope = new[start:stop]
        anchor = "    // Per-dispatch AsyncCtx constant prefill" if "assign_own" in begin else "    // Prefill the per-dispatch AsyncCtx constant fields once."
        assert scope.count(anchor) == 1
        scope = scope.replace(anchor, "    const void *notification_ranges[RUNTIME_MAX_WORKER];\n    size_t notification_count = 0;\n\n" + anchor)
        anchor = "cache_flush_range(&notification, sizeof(notification));"
        assert scope.count(anchor) == 1
        scope = scope.replace(anchor, "notification_ranges[notification_count++] = &notification;")
        anchor = "\n}\n\n" if "assign_own" in begin else "    functions_ = functions;"
        assert scope.count(anchor) == 1
        scope = scope.replace(anchor, "\n    cache_flush_ranges(notification_ranges, notification_count, sizeof(TaskBoundaryNotification));\n" + anchor)
        new = new[:start] + scope + new[stop:]
    result[path] = (old, new)
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
