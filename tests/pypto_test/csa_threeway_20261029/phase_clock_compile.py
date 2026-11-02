# SPDX-License-Identifier: Apache-2.0
"""仅不占卡任务构建用户授权的诊断时钟副本。"""

import argparse
import importlib
import json
from pathlib import Path

import torch_npu
from csa_threeway_20261029.paired_single_layer import mount_package
from csa_threeway_20261029.phase_clock import artifact_receipt, instrument, prepare, prepare_hc_detail
from pypto.runtime.kernel.abi import KernelConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("arm", choices=("ours", "nalinaly"))
parser.add_argument("output", type=Path)
parser.add_argument("--seal-existing", action="store_true")
parser.add_argument("--batch", type=int, choices=range(1, 41), default=8)
parser.add_argument("--reuse-source", action="store_true")
parser.add_argument("--hc-detail", action="store_true")
args = parser.parse_args()
if not args.seal_existing:
    args.output.mkdir(exist_ok=False)
source = (
    args.root
    / {
        "ours": "cohorts/eightcard-source-v5/ours/source",
        "nalinaly": "cohorts/eightcard-nal-pre-aclnn-1574-v1/nalinaly/source",
    }[args.arm]
    / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
)
target = (
    args.root
    / ("candidates/hc-detail-clock-20261102-v1" if args.hc_detail else "candidates/phase-clock-20261102-v2")
    / args.arm
)
if not args.seal_existing and not args.reuse_source:
    prepare(source, target)
    if args.hc_detail:
        prepare_hc_detail(target, args.arm)
else:
    assert target.is_dir()
name = mount_package(target, "vllm_ascend.ops.pypto._phase_" + args.arm)
entry = importlib.import_module(name + ".entry")
policy = importlib.import_module(name + ".policy").select_policy(
    tokens=args.batch * 6, requests=args.batch, max_query_tokens=6, max_seq_len=8198
)
if args.hc_detail:
    assert policy.constexprs()["OUTPUT_GROUP_PIPELINE"], "HC detail requires unused legacy output slots"
assert not torch_npu.npu.is_initialized()


def resolve():
    artifact, _, _ = entry.decode_csa_spmd._resolve_compiled(
        (),
        policy.constexprs(),
        allow_signature_mode=True,
        _kernel=True,
        _kernel_config=KernelConfig("a2a3", "tensormap_and_ringbuffer", 0, ring_task_window=2048),
    )
    artifact.load()
    return artifact


if args.seal_existing:
    artifact = resolve()
else:
    with instrument(args.root, args.output / "codegen", target):
        artifact = resolve()
assert len(artifact.kernel_abi.parameters) == 56 and not torch_npu.npu.is_initialized()
assert (args.output / "codegen/audit.json").exists(), "clock instrumentation hook did not run"
receipt = artifact_receipt(artifact, args.output / "codegen")
(args.output / "sealed.json").write_text(json.dumps({"cpu_pass": True, "arm": args.arm, **receipt}) + "\n")
