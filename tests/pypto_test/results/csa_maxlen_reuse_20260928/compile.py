"""Compile and load one frozen CSA under CANN 9.2 without device execution."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "tests/pypto_test"))
    from dsv4_csa_env import activate

    os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"
    os.environ["PTO_CSA_VARIANT"] = "performance"
    os.environ["VLLM_ASCEND_PTO_CSA_ATOMIC_ADD"] = "0"
    assert os.environ["ASCEND_HOME_PATH"].endswith("/cann-9.2.0-beta.2")
    activate()
    from pypto.runtime import RunConfig

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_csa import decode_csa_tp1_layer_test

    compiled = decode_csa_tp1_layer_test.compile(
        config=RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output))
    )
    compiled.load()
    print(f"COMPILE_PASS {args.source}; CANN 9.2 full performance CSA; no device execution")


if __name__ == "__main__":
    main()
