# SPDX-License-Identifier: Apache-2.0
"""编译失败定向诊断：仅保存进入代码生成前的一份IR，不展开所有pass。"""

import contextlib
from pathlib import Path


@contextlib.contextmanager
def final_ir_capture(destination):
    from pypto.backend import pto_backend
    from pypto.ir.printer import python_print

    destination = Path(destination)
    original = pto_backend._generate_single_chip

    def hooked(program, *args, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(python_print(program, explicit_layout=True))
        return original(program, *args, **kwargs)

    pto_backend._generate_single_chip = hooked
    try:
        yield
    finally:
        pto_backend._generate_single_chip = original
