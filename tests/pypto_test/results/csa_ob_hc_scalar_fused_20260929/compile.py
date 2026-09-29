"""Resolve frozen private-package dependencies and compile both CSA roots on CPU."""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
VARIANT = "pkg:dsv4_csa_ob_hc_scalar_fused_9a868d26"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=("baseline", "candidate"))
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()
    source = WORKSPACE / f".cache/csa-ob-hc-scalar-fused-9a868d26-{args.side}"
    sys.path.insert(0, str(source / "tests/pypto_test"))
    from dsv4_csa_env import activate

    os.environ.update(VLLM_ASCEND_ENABLE_NZ="2", PTO_CSA_VARIANT=VARIANT,
                      VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0")
    activate()
    from pypto.runtime import RunConfig

    from vllm_ascend.ops.pypto.variant import variant_package

    module = importlib.import_module(f"{variant_package()}.decode_csa")
    graphs = {n: type(getattr(module, n)._get_dep_graph()).__name__ for n in (
        "decode_csa_tp1_layer", "decode_csa_tp1_layer_test")}
    if args.parse_only:
        print(json.dumps({"side": args.side, "graphs": graphs, "status": "PARSE_PASS"}))
        return
    compiled = module.decode_csa_tp1_layer_test.compile(config=RunConfig(
        platform="a2a3", dump_passes=True, save_kernels=True, save_kernels_dir=str(ROOT / "compiled" / args.side)))
    compiled.load()
    report = {"side": args.side, "source": str(source), "variant": VARIANT,
              "cann": os.environ["ASCEND_HOME_PATH"], "dependency_graphs": graphs,
              "status": "COMPILE_PASS", "device_execution": False}
    (ROOT / f"compile_{args.side}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("COMPILE_PASS", args.side, flush=True)


if __name__ == "__main__":
    main()
