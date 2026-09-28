"""Parse and compile the complete private CSA package before queue submission."""

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
SOURCE = WORKSPACE / ".cache/csa-b24-cann92-3b27c7fd"
VARIANT = "pkg:dsv4_csa_b24_cann92_3b27c7fd"


def main():
    sys.path.insert(0, str(SOURCE / "tests/pypto_test"))
    from dsv4_csa_env import activate

    os.environ.update(VLLM_ASCEND_ENABLE_NZ="2", PTO_CSA_VARIANT=VARIANT,
                      VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0")
    if not os.environ["ASCEND_HOME_PATH"].endswith("/cann-9.2.0-beta.2"):
        raise RuntimeError("CANN 9.2 required")
    activate()
    from pypto.runtime import RunConfig

    from vllm_ascend.ops.pypto.variant import variant_package

    module = importlib.import_module(f"{variant_package()}.decode_csa")
    graphs = {}
    for name in ("decode_csa_tp1_layer", "decode_csa_tp1_layer_test"):
        graph = getattr(module, name)._get_dep_graph()
        graphs[name] = type(graph).__name__
    compiled = module.decode_csa_tp1_layer_test.compile(
        config=RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(ROOT / "compiled"))
    )
    compiled.load()
    result = {"source": str(SOURCE), "operator": "3b27c7fd", "variant": VARIANT,
              "cann": os.environ["ASCEND_HOME_PATH"], "dependency_graphs": graphs,
              "status": "COMPILE_PASS", "device_execution": False}
    (ROOT / "prepare.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
