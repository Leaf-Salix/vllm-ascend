# SPDX-License-Identifier: Apache-2.0
"""实验共用公开ring容量选项；不修改SDK或算子。"""

from functools import wraps

RING_HEAP = (256 << 20, 512 << 20, 256 << 20, 256 << 20)


def configure_runtime():
    import pypto.torch

    original = pypto.torch.init

    @wraps(original)
    def init(**kwargs):
        kwargs.setdefault("ring_heap", RING_HEAP)
        if tuple(kwargs["ring_heap"]) != RING_HEAP:
            raise ValueError("Three-way tests must share the recorded ring heap configuration")
        return original(**kwargs)

    pypto.torch.init = init
