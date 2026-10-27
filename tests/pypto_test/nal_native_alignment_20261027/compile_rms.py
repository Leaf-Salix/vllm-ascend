# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the private package before importing its DSL modules.
"""占卡前编译RMS根入口并加载ABI，不初始化NPU。"""

import json
import sys
from pathlib import Path

from bind_package import bind

bind()
import torch_npu
from hc_kernels import hc_pair_arm
from layout_kernels import pack_pair
from pypto.ir._kernel_compile import kernel_abi_for_program
from pypto.ir.compile import _compile_impl
from pypto.runtime._kernel_artifact import KernelArtifact
from rms_kernels import rms_pair

out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
assert not torch_npu.npu.is_initialized()
kind = sys.argv[2] if len(sys.argv) > 2 else "rms"
if kind == "qr":
    from qr_kernels import qr_arm

    program = qr_arm.specialize()
elif kind == "pack":
    program = pack_pair.specialize()
elif kind == "rms":
    program = rms_pair.specialize()
elif kind in ("spmd", "spmd-debug", "spmd-probe"):
    from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.entry import decode_csa_spmd, decode_csa_spmd_debug
    from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.policy import select_policy

    policy = select_policy(tokens=24, requests=4, max_query_tokens=6, max_seq_len=8198)
    if kind == "spmd-probe":
        from spmd_probe_entry import spmd_stage_probe

        root = spmd_stage_probe
    else:
        root = decode_csa_spmd if kind == "spmd" else decode_csa_spmd_debug
    program = root.specialize(**policy.constexprs())
elif kind == "hc-reference":
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_full_precision.hc_pre import hc_pre_test

    program = hc_pre_test.specialize()
else:
    program = hc_pair_arm.specialize(
        NATIVE_RMS=kind != "hc-base",
        NATIVE_LINEAR=kind in ("hc-linear", "hc-fold", "hc-norm"),
        NATIVE_FOLD=kind in ("hc-fold", "hc-norm"),
        NATIVE_NORM=kind == "hc-norm",
        DIRECT_PACK=kind == "hc-norm",
    )
abi = kernel_abi_for_program(program, platform="a2a3", runtime="tensormap_and_ringbuffer")
_compile_impl(program, output_dir=str(out / "kernel"), platform="a2a3", skip_ptoas=False, _kernel_abi=abi)
artifact = KernelArtifact(out / "kernel", abi)
artifact.load()
assert not torch_npu.npu.is_initialized()
(out / "receipt.json").write_text(
    json.dumps(
        {
            "pass": True,
            "npu_initialized": False,
            "root_tensors": len(abi.parameters),
            "artifact_identity": artifact.identity().hex(),
        },
        indent=2,
    )
    + "\n"
)
