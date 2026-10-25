# SPDX-License-Identifier: Apache-2.0
"""用已记录的真实描述符和meta Tensor编译，既不申请实际缓冲，也不运行NPU。"""

import argparse
import importlib
import inspect
import json
import sys
import traceback
from pathlib import Path

import torch
from prepare import hashes
from pypto import passes
from pypto.runtime import CompileOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("record", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(here / "frozen_optimization"))
    from run_pto import mount_package

    report = {"passed": False, "CPU_only": True, "record": str(args.record), "source_hashes": hashes(args.candidate)}
    try:
        recorded = json.loads(args.record.read_text())
        assert recorded["passed"] and recorded["stage"] == "complete"
        prefix = "vllm_ascend.ops.pypto."
        mount_package(here / "frozen_optimization/pypto/deepseek_v4_flash_csa", prefix + "deepseek_v4_flash_csa")
        name = mount_package(args.candidate, prefix + "_static_capacity_probe")
        module = importlib.import_module(name + ".entry")
        entry = module.decode_csa_spmd_debug
        params = inspect.signature(entry._func).parameters
        descriptors = recorded["descriptors"]["current"]
        samples = {
            name: torch.empty(info["shape"], dtype=getattr(torch, info["dtype"].removeprefix("torch.")), device="meta")
            for name, info in descriptors.items()
            if name in params
        }
        policy = recorded["policies"]["current"]
        assert set(params) == set(samples) | set(policy)
        report.update(sample_descriptors={k: descriptors[k] for k in samples}, policy=policy)
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        entry.compile(**samples, **policy, config=options)
        report.update(passed=True, stage="cpu_backend_compiled_from_meta_samples")
    except Exception:
        report.update(stage="failed", error=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["stage"])


if __name__ == "__main__":
    main()
