# SPDX-License-Identifier: Apache-2.0
"""无设备核对共同源码、共享SDK和只读bank；不加载模型权重。"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("bank", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not torch_npu.npu.is_initialized()
    plan = json.loads((args.bank / "plan.json").read_text())
    assert json.loads((args.bank / "audit.json").read_text())["status"] == "PASS"
    assert len(plan["cases"]) == 40 and len({case["key"] for case in plan["cases"]}) == 40
    assert all(case["source_history"] == 8192 for case in plan["cases"])
    model = Path(plan["model"])
    index = json.loads((model / "quant_model_weights.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) == 75 and all((model / name).is_file() for name in shards)
    modules = {}
    for name in ("pypto", "simpler", "vllm", "vllm_ascend"):
        module = importlib.import_module(name)
        modules[name] = {"file": module.__file__, "version": getattr(module, "__version__", None)}
    for name, variable in (("pypto", "CSA_SHARED_PYPTO_ROOT"), ("simpler", "CSA_SHARED_SIMPLER_ROOT")):
        assert Path(os.environ[variable]).resolve() in Path(modules[name]["file"]).resolve().parents
    assert args.source.resolve() in Path(modules["vllm_ascend"]["file"]).resolve().parents
    assert modules["vllm"]["version"] == "0.25.1"
    assert os.environ["PTOAS_ROOT"] == os.environ["CSA_SHARED_PTOAS_ROOT"]
    assert os.environ["PYTHONNOUSERSITE"] == "1" and os.environ["HCCL_DETERMINISTIC"] == "true"
    assert not torch_npu.npu.is_initialized()
    receipt = {
        "pass": True,
        "npu_initialized": False,
        "python": sys.version,
        "modules": modules,
        "bank": str(args.bank),
        "cases": len(plan["cases"]),
        "shards": len(shards),
        "sdk_release": os.environ["CSA_SHARED_TOOLCHAIN_RELEASE"],
        "PTOAS_ROOT": os.environ["PTOAS_ROOT"],
        "ASCEND_HOME_PATH": os.environ["ASCEND_HOME_PATH"],
        "source": json.loads((args.source.parent / "manifest.json").read_text()),
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "source"}), flush=True)


if __name__ == "__main__":
    main()
