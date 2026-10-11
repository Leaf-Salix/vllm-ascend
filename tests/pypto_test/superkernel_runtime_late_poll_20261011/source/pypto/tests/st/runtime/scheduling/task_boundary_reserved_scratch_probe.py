# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Isolate allocator-selected scratch from the runtime notification call site."""

from contextlib import contextmanager

import task_boundary_completion_probe as probe
from pypto import ir
from pypto.backend import pto_backend

original_adapter = probe.completion_adapter


@contextmanager
def reserved_scratch_adapter(mode, cube):
    with original_adapter(mode, cube):
        wrapped = pto_backend._generate_kernel_wrapper

        def generate(func, *args, **kwargs):
            attrs = dict(func.attrs)
            scratch = attrs.pop("task_boundary_scratch_base")
            plain = ir.Function(
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
            code = wrapped(plain, *args, **kwargs)
            return code.replace("524224", str(scratch)).replace("196544", str(scratch))

        pto_backend._generate_kernel_wrapper = generate
        yield


if __name__ == "__main__":
    probe.completion_adapter = reserved_scratch_adapter
    probe.main()
