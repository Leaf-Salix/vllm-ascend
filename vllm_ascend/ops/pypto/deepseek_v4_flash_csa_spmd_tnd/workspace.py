# SPDX-License-Identifier: Apache-2.0
"""同一vLLM模型内顺序CSA层的临时GM复用，不复用cache或同步代际。

模型V1逐层在同一执行流上提交，后一层依赖前一层的完整输出。因此不同层
不必各自保留相同形状的临时tensor。每个模型实例拥有自己的池；微批并发场景不接入。
捕获前分配且不扩容/替换已有tensor，使不同图保留的地址持续有效。
各容量仍有自己的分配，避免较小档的布局覆盖较大档。单层独立调用仍默认独占。
"""

import torch


class TNDWorkspacePool:
    """仅复用完整调用之间无需持久保存的中间量。"""

    def __init__(self):
        self._buffers = {}

    def allocate(self, name, shape, dtype, device):
        # signal承载跨调用代际，必须由每个PreparedDynamicCSA独占。
        # ffts保留独立运行时控制存储；其容量很小，不为省这部分引入协议耦合。
        if name == "ffts" or name == "signal" or name.endswith("_signal"):
            return torch.empty(shape, dtype=dtype, device=device)
        key = (name, tuple(shape), dtype, torch.device(device))
        if key not in self._buffers:
            self._buffers[key] = torch.empty(shape, dtype=dtype, device=device)
        return self._buffers[key]

    @property
    def allocated_bytes(self):
        return sum(value.numel() * value.element_size() for value in self._buffers.values())
