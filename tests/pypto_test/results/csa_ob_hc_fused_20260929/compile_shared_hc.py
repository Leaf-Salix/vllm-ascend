"""CPU编译共享HC的原单行入口，验证抽取helper没有破坏精度版调用形式。"""

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[4]
SOURCE = WORKSPACE / ".cache/csa-ob-hc-fused-9a868d26-candidate"


def main():
    sys.path.insert(0, str(SOURCE / "tests/pypto_test"))
    from dsv4_csa_env import activate

    os.environ.update(VLLM_ASCEND_ENABLE_NZ="2", PTO_CSA_VARIANT="pkg:dsv4_csa_ob_hc_fused_9a868d26",
                      VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0")
    activate()
    from pypto.runtime import RunConfig

    module = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.hc_post")
    compiled = module.hc_post_test.compile(config=RunConfig(
        platform="a2a3", dump_passes=True, save_kernels=True,
        save_kernels_dir=str(ROOT / "compiled_shared_hc")))
    compiled.load()
    report = {"status": "COMPILE_PASS", "source": str(SOURCE), "entry": "shared hc_post_test",
              "scope": "CPU compile/load only; not a numerical or precision-CSA acceptance"}
    (ROOT / "compile_shared_hc.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("COMPILE_PASS shared_hc", flush=True)


if __name__ == "__main__":
    main()
