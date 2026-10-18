# SPDX-License-Identifier: Apache-2.0
"""只在失败后尝试读取诊断控制区，不重放任务、不改信号、不重置设备。

torch的流已报错时，Tensor.cpu可能被相同错误拒绝。这里尝试ACL同步D2H，
读取独占signal/group metadata以定位停留阶段；若驱动也拒绝则记录返回码。
这不是已完成计算的快照，不能用于精度验收或证明缓存/异步流水正确。
"""

import ctypes
import json
from pathlib import Path


def capture(prepared, output):
    import acl

    rows = {}
    for name in ("signal", "group_plan", "group_offsets", "attention_mask"):
        tensor = prepared.buffers[name]
        size = tensor.numel() * tensor.element_size()
        if tensor.element_size() != 4 or size > 16 * 1024 * 1024:
            raise ValueError("诊断只接受至多16MiB的INT32控制区")
        storage = ctypes.create_string_buffer(size)
        result = acl.rt.memcpy(ctypes.addressof(storage), size, tensor.data_ptr(), size, 2)
        record = dict(bytes=size, shape=list(tensor.shape), acl_memcpy_return=result)
        if result == 0:
            values = list((ctypes.c_int32 * tensor.numel()).from_buffer(storage))
            if name == "signal":
                slots = tensor.numel() // (72 * 128)
                record["word0_by_phase_core"] = [[values[(phase * 72 + core) * 128]
                                                  for core in range(72)] for phase in range(slots)]
                record["nonzero_padding"] = sum(v != 0 for i, v in enumerate(values) if i % 128)
            else:
                record["values"] = values
        rows[name] = record
    report = dict(scope="失败后ACL控制区诊断，计算未完成，不是精度/性能验收", tensors=rows)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {key: dict(bytes=value["bytes"], acl_memcpy_return=value["acl_memcpy_return"])
            for key, value in rows.items()}
