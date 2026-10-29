# SPDX-License-Identifier: Apache-2.0
"""在227 CPU按整模型实际public模块名预热缓存，禁止创建Worker或初始化NPU。

private诊断namespace的缓存不能代替服务实际模块的缓存资格。
必须由公共bootstrap选定source及共享SDK；不改冻结快照或SDK。
"""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import torch_npu
from pypto.runtime.kernel.abi import KernelConfig
from pypto.runtime.kernel.context import get_process_kernel_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, choices=(6, 40, 240), required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    manifest = json.loads((source.parent / "manifest.json").read_text())
    assert manifest["arm"] in ("ours", "nalinaly")
    assert not args.output.exists() and os.environ["PYPTO_CACHE"] == "1"
    assert os.environ["PYTHONNOUSERSITE"] == "1"
    state = get_process_kernel_state()
    assert not torch_npu.npu.is_initialized() and state.config is None
    package = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd"
    entry = importlib.import_module(package + ".entry")
    policy_module = importlib.import_module(package + ".policy")
    assert source in Path(entry.__file__).resolve().parents
    assert source in Path(policy_module.__file__).resolve().parents
    policy = policy_module.select_policy(
        tokens=args.tokens,
        requests=1 if args.tokens == 6 else 40,
        max_query_tokens=6,
        max_seq_len=9000,
    )
    start = time.monotonic()
    artifact, _, _ = entry.decode_csa_spmd._resolve_compiled(
        (),
        policy.constexprs(),
        allow_signature_mode=True,
        _kernel=True,
        _kernel_config=KernelConfig("a2a3", "tensormap_and_ringbuffer", 0),
    )
    artifact.load()
    assert len(artifact.kernel_abi.parameters) == 55
    assert not torch_npu.npu.is_initialized() and state.config is None
    receipt = {
        "pass": True,
        "arm": manifest["arm"],
        "csa_source": manifest["csa_source"],
        "entry": entry.__file__,
        "module": entry.__name__,
        "tokens": args.tokens,
        "elapsed_s": time.monotonic() - start,
        "npu_initialized": False,
        "worker_created": False,
        "root_tensors": 55,
        "artifact_identity": artifact.identity().hex(),
        "cache": os.environ["PYPTO_CACHE_DIR"],
        "sdk_release": os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"],
        "bindings": policy.constexprs(),
        "scope": "CPU compile only; no model/NPU or token qualification",
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
