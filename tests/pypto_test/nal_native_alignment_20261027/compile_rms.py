# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- Bind the private package before importing its DSL modules.
"""占卡前编译RMS根入口并加载ABI，不初始化NPU。"""

import json
import sys
from pathlib import Path

import regex as re
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
if kind.startswith("woa-"):
    from woa_kernels import woa_arm

    woa_mode = int(kind.removeprefix("woa-"))
    program = woa_arm.specialize(MODE=2 if woa_mode == 3 else woa_mode, ROW_TILE=128 if woa_mode == 3 else 16)
elif kind == "attention-native-spmd":
    from attention_native_micro import native_attention_arm

    program = native_attention_arm.specialize()
elif kind == "attention-reference":
    from attention_reference import attention_reference_arm

    program = attention_reference_arm.specialize()
elif kind.startswith("weightproj-"):
    from weightproj_kernels import weightproj_arm

    program = weightproj_arm.specialize(MODE=int(kind.removeprefix("weightproj-")))
elif kind.startswith("idxquant-"):
    from idxquant_kernels import idxquant_arm

    program = idxquant_arm.specialize(MODE=int(kind.removeprefix("idxquant-")))
elif kind.startswith("idxquery-rope-"):
    from idxquery_kernels import idxquery_rope_arm

    program = idxquery_rope_arm.specialize(MODE=int(kind.removeprefix("idxquery-rope-")))
elif kind.startswith("idxquery-dequant-"):
    from idxquery_kernels import idxquery_dequant_arm

    program = idxquery_dequant_arm.specialize(COMBINED=bool(int(kind.removeprefix("idxquery-dequant-"))))
elif kind.startswith("indexproj-"):
    from indexproj_kernels import indexproj_arm

    program = indexproj_arm.specialize(MODE=int(kind.removeprefix("indexproj-")))
elif kind.startswith("mainproj-"):
    from mainproj_kernels import mainproj_arm

    program = mainproj_arm.specialize(MODE=int(kind.removeprefix("mainproj-")))
elif kind.startswith("qoutput-"):
    from qoutput_kernels import qoutput_arm

    program = qoutput_arm.specialize(MODE=int(kind.removeprefix("qoutput-")))
elif kind in ("qdequant-base", "qdequant-combined"):
    from qoutput_kernels import qdequant_arm

    program = qdequant_arm.specialize(COMBINED=kind == "qdequant-combined")
elif kind in ("kvnorm-base", "kvnorm-native"):
    from kvnorm_kernels import kvnorm_arm

    program = kvnorm_arm.specialize(NATIVE_BOUNDARIES=kind == "kvnorm-native")
elif kind in ("wkv-base", "wkv-natural", "wkv-paired", "wkv-dispatch"):
    from wkv_kernels import wkv_arm

    program = wkv_arm.specialize(NARROW=kind != "wkv-base", ORDER=kind == "wkv-paired", DISPATCH=kind == "wkv-dispatch")
elif kind in ("qr", "qr-native"):
    from qr_kernels import qr_arm

    program = qr_arm.specialize(NATIVE=kind == "qr-native")
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
dual_aiv_dispatch = None
if kind in ("attention-native-spmd", "spmd", "spmd-debug", "spmd-probe"):
    orchestration = list((out / "kernel/orchestration").glob("*.cpp"))
    assert len(orchestration) == 1
    mixed = re.findall(
        r"MixedKernels\s+mixed_\d+\s*=\s*\{\s*(\d+),\s*(\d+),\s*(\d+)\s*\};",
        orchestration[0].read_text(),
    )
    assert len(mixed) == 1 and mixed[0][1] == mixed[0][2], "both AIV lanes must be dispatched"
    dual_aiv_dispatch = True
(out / "receipt.json").write_text(
    json.dumps(
        {
            "pass": True,
            "npu_initialized": False,
            "root_tensors": len(abi.parameters),
            "artifact_identity": artifact.identity().hex(),
            "dual_aiv_dispatch": dual_aiv_dispatch,
        },
        indent=2,
    )
    + "\n"
)
