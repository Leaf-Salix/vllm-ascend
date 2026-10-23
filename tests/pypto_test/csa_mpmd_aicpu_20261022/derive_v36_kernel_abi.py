# SPDX-License-Identifier: Apache-2.0
"""从合并版真实pre-pass IR推导kernel ABI；不重新编译、不创建Worker或访问NPU。"""

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    if os.environ.get("VLLM_ASCEND_ENABLE_NZ") != "2":
        raise ValueError("ABI推导必须与实际测试一致，先设置VLLM_ASCEND_ENABLE_NZ=2")
    experiment = HERE / "experiments/aicpu_v36_capacity_rectangle"
    sys.path.insert(0, str(experiment / "helpers"))
    import dsv4_csa_env

    dsv4_csa_env.activate()
    from pypto.ir._kernel_compile import kernel_abi_for_program

    spec = importlib.util.spec_from_file_location("csa_abi_harness", experiment / "run_case.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    frozen = HERE.parent / "csa_mpmd_tnd_20261021/sources/84751e34/vllm_ascend/ops/pypto"
    harness.mount_package(frozen / "deepseek_v4_flash_csa", "vllm_ascend.ops.pypto.deepseek_v4_flash_csa")
    name = harness.mount_package(experiment / "operator", "vllm_ascend.ops.pypto._v36_abi")
    entry = importlib.import_module(name + ".entry")
    policy = importlib.import_module(name + ".policy")
    cases = json.loads((HERE.parent / "csa_performance_20261022_aicpu/cases.json").read_text())["cases"]
    records = []
    for case_name, reference, columns in (
        ("h262144_b16_s6", "v34", 2050),
        ("h131072_b24_s6", "v32", 1026),
    ):
        case = next(c for c in cases if c["name"] == case_name)
        selected = policy.select_policy(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
            compressed_table_columns=columns,
        )
        program = entry.decode_csa_tnd.specialize(**selected.constexprs())
        abi = kernel_abi_for_program(program, platform="a2a3", runtime="tensormap_and_ringbuffer").record()
        evidence = json.loads((HERE / f"scheduling_{reference}_validation.json").read_text())
        validated = Path(evidence["cpp_by_side"]["mpmd"]).parents[1]
        previous = json.loads((validated / "compiled_meta.json").read_text())
        if abi != previous["kernel_abi"]:
            differences = {
                key: [previous["kernel_abi"].get(key), abi.get(key)]
                for key in set(abi) | set(previous["kernel_abi"])
                if previous["kernel_abi"].get(key) != abi.get(key)
            }
            raise ValueError(json.dumps(differences, ensure_ascii=False))
        cpu = next((HERE / f"runs/cpu_v36_{case_name}/build_output").glob("decode_csa_tnd_*"))
        actual = json.loads((cpu / "compiled_meta.json").read_text())
        for key in ("params", "num_return_types", "platform", "backend_type"):
            assert actual[key] == previous[key], (case_name, key)
        records.append(
            dict(
                case=case_name,
                reference=reference,
                reference_build=str(validated),
                cpu_build=str(cpu),
                derived_from_actual_prepass_ir=True,
                abi_equal=True,
                lowered_params_equal=True,
                kernel_abi=abi,
            )
        )
        print(case_name, "kernel ABI and lowered parameter order match")
    (HERE / "scheduling_v36_kernel_abi.json").write_text(
        json.dumps({"passed": True, "device_execution": False, "cases": records}, ensure_ascii=False, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
