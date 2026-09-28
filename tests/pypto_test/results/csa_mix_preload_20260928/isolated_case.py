"""使用隔离运行时执行既有CSA用例，不修改默认环境的editable映射。"""

import os
import runpy
import sys
from pathlib import Path

WORKSPACE = Path("/data/pyptouser/qinchuanyu/pto-eager")
PYPTO = WORKSPACE / ".cache/pypto-pr2389-3e87a843"
REPO = WORKSPACE / "vllm-ascend-dsv4-pto-0251rc1"
SOURCE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SOURCE / "tests/pypto_test"))
import dsv4_csa_env


def activate():
    os.environ["DYNAMIC_EPLB"] = "false"
    os.environ["EXPERT_MAP_RECORD"] = "false"
    vllm = WORKSPACE / ".cache/migration-v0.25.1rc1/vllm"
    sources = [str(vllm), str(SOURCE)]
    sys.path[:0] = sources
    os.environ["PYTHONPATH"] = os.pathsep.join(sources + os.environ.get("PYTHONPATH", "").split(os.pathsep))
    import pypto
    import simpler_setup
    from pypto._kernel_abi import SIMPLER_KERNEL_REVISION
    from pypto.torch.launch import _load_native
    import _task_interface

    assert PYPTO in Path(pypto.__file__).resolve().parents, pypto.__file__
    assert PYPTO in Path(simpler_setup.__file__).resolve().parents, simpler_setup.__file__
    assert SIMPLER_KERNEL_REVISION == _task_interface.__build_commit__ == _load_native().simpler_revision
    assert SIMPLER_KERNEL_REVISION == "52c4e019e60d84d267b67617d0439d97bbd9116b"
    return SOURCE


def main():
    dsv4_csa_env.activate = activate
    original_write = dsv4_csa_env.write_json

    def write_json(path, data):
        if Path(path).name == "report.json":
            data["runtime_experiment"] = {
                "simpler": "52c4e019e60d84d267b67617d0439d97bbd9116b",
                "pypto": "58aab925",
                "mix_preload_max_remaining_us": os.environ.get("SIMPLER_MIX_PRELOAD_MAX_REMAINING_US", "50"),
                "scope": "PR #2389 local backport; isolated SDK and native bindings",
            }
        original_write(path, data)

    dsv4_csa_env.write_json = write_json
    runpy.run_path(str(REPO / "tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py"), run_name="__main__")


if __name__ == "__main__":
    main()
