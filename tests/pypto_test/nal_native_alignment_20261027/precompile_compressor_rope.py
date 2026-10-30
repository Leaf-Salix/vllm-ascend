# SPDX-License-Identifier: Apache-2.0
"""227 CPU预编译Norm/RoPE单因素微测试，不创建NPU Worker。"""

import json
import sys
import time
from pathlib import Path

import torch_npu
from compressor_rope_kernels import compressor_rope_arm
from pypto.runtime.kernel.abi import KernelConfig
from pypto.runtime.kernel.context import get_process_kernel_state

output = Path(sys.argv[1])
assert not output.exists()
state = get_process_kernel_state()
assert not torch_npu.npu.is_initialized() and state.config is None
arms = []
for direct in (False, True):
    start = time.monotonic()
    artifact, _, _ = compressor_rope_arm._resolve_compiled(
        (),
        {"DIRECT": direct, "EPS": 1e-6},
        allow_signature_mode=True,
        _kernel=True,
        _kernel_config=KernelConfig("a2a3", "tensormap_and_ringbuffer", 0),
    )
    artifact.load()
    assert len(artifact.kernel_abi.parameters) == 5
    assert not torch_npu.npu.is_initialized() and state.config is None
    arms.append({"direct": direct, "seconds": time.monotonic() - start, "artifact": artifact.identity().hex()})
output.write_text(
    json.dumps({"pass": True, "npu_initialized": False, "worker_created": False, "arms": arms}, indent=2) + "\n"
)
