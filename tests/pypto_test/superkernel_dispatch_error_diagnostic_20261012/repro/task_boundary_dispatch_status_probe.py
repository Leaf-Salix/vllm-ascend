# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Expose the logical dispatch error before CANN folds it into a generic code."""

import task_boundary_contract_cache_probe as cached
import task_boundary_normal_state_probe as normal


def changes(root):
    result = cached.changes(root)
    path = root / "src/common/platform/shared/aicpu/kernel_dispatch.cpp"
    old = path.read_text()
    anchor = '#include "kernel_dispatch_args.h"'
    assert old.count(anchor) == 1
    new = old.replace(anchor, anchor + '\n#include "common/unified_log.h"')
    anchor = "    return consume_kernel_task(arg) == 0 ? 0 : 2;"
    assert new.count(anchor) == 1
    new = new.replace(anchor, """    const int status = consume_kernel_task(arg);
    if (status != 0) LOG_ERROR("[BOUNDARY_DISPATCH_FAILURE] logical_status=%d", status);
    return status == 0 ? 0 : 2;""")
    result[path] = (old, new)
    return result


if __name__ == "__main__":
    normal.changes = changes
    normal.main()
