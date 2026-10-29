# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the same package namespace as the device runner.
"""在227登录端预热eager持久编译缓存；不创建Worker、不初始化或提交NPU。

限定本次共享PyPTO版本的现有内部解析接口；不修改SDK或生成代码。
普通program compile使用另一份缓存，不能用它证明NPU调用会命中eager缓存。
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch_npu
from bind_package import bind

bind()
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.entry import decode_csa_spmd, decode_csa_spmd_debug
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy
from pypto.runtime.kernel.abi import KernelConfig
from pypto.runtime.kernel.context import get_process_kernel_state
from spmd_probe_entry import spmd_stage_probe

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument(
    "kind",
    choices=(
        "spmd",
        "spmd-debug",
        "spmd-probe",
        "qa240-0",
        "qa240-1",
        "qa240-2",
        "topktie-0",
        "topktie-1",
        "topktie-2",
        "woa240-0",
        "woa240-1",
        "woa240-2",
    ),
)
parser.add_argument("--lengths", default=",".join(["6"] * 40))
args = parser.parse_args()
assert not args.output.exists()
assert os.environ["PYPTO_CACHE"] == "1"
state = get_process_kernel_state()
assert not torch_npu.npu.is_initialized() and state.config is None
lengths = [int(value) for value in args.lengths.split(",")]
assert 1 <= len(lengths) <= 40 and all(1 <= value <= 6 for value in lengths)
if args.kind.startswith("woa240-"):
    from woa240_kernels import woa240_arm

    root, bindings, expected = woa240_arm, {"MODE": int(args.kind[-1])}, 3
elif args.kind.startswith("topktie-"):
    from topktie_kernels import topktie_arm

    root, bindings, expected = topktie_arm, {"VISIBLE": 2049, "MODE": int(args.kind[-1])}, 4
elif args.kind.startswith("qa240-"):
    from qa240_kernels import qa240_arm

    root, bindings, expected = qa240_arm, {"MODE": int(args.kind[-1])}, 3
else:
    policy = select_policy(tokens=sum(lengths), requests=len(lengths), max_query_tokens=max(lengths), max_seq_len=8198)
    root, expected = {
        "spmd": (decode_csa_spmd, 55),
        "spmd-debug": (decode_csa_spmd_debug, 57),
        "spmd-probe": (spmd_stage_probe, 85),
    }[args.kind]
    bindings = policy.constexprs()
start = time.monotonic()
artifact, _, _ = root._resolve_compiled(
    (),
    bindings,
    allow_signature_mode=True,
    _kernel=True,
    _kernel_config=KernelConfig("a2a3", "tensormap_and_ringbuffer", 0),
)
artifact.load()
assert len(artifact.kernel_abi.parameters) == expected
assert not torch_npu.npu.is_initialized() and state.config is None
record = {
    "pass": True,
    "kind": args.kind,
    "elapsed_s": time.monotonic() - start,
    "npu_initialized": False,
    "worker_created": False,
    "root_tensors": expected,
    "artifact_identity": artifact.identity().hex(),
    "cache": os.environ["PYPTO_CACHE_DIR"],
    "sdk_release": os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"],
    "bindings": bindings,
    "scope": "eager artifact cache build only; device reuse and precision require separate runs",
}
args.output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
