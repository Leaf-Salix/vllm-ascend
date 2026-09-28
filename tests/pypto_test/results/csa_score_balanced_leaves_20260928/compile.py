"""CPU核查长档工作分片均衡候选，不执行NPU。"""

import os
import sys
from pathlib import Path

SOURCE = Path("/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-score-balanced-8e176285")


def main():
    sys.path.insert(0, str(SOURCE / "tests/pypto_test"))
    from dsv4_csa_env import activate

    os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"
    os.environ["PTO_CSA_VARIANT"] = "performance"
    os.environ["VLLM_ASCEND_PTO_CSA_ATOMIC_ADD"] = "0"
    activate()
    from pypto.runtime import RunConfig

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_csa import decode_csa_tp1_layer_test

    compiled = decode_csa_tp1_layer_test.compile(
        config=RunConfig(
            platform="a2a3", save_kernels=True, save_kernels_dir=str(Path(__file__).resolve().parent / "compiled")
        )
    )
    compiled.load()
    print(
        "COMPILE_PASS full performance CSA; balanced long leaves for 16 query groups; "
        "no arena growth or device execution"
    )


if __name__ == "__main__":
    main()
