# SPDX-License-Identifier: Apache-2.0
"""记录三方比较的SPMD工作区地址，保持既有分配与采样行为。

默认只在Host读取Tensor的地址、shape、stride等描述符，不读取device值。
显式--share-scratch时，仅在同形状/类型/设备下复用候选工作区，signal和ffts
仍各自独立，以保持代际和事件合同。这个补充对照不覆盖旧独立工作区结果。
"""

import argparse
import importlib
import json
from pathlib import Path
import runpy
import sys


def describe(value):
    if value is None:
        return None
    if not hasattr(value, "data_ptr"):
        return {"type": type(value).__name__}
    return {
        "address": value.data_ptr(),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "storage_offset": value.storage_offset(),
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--share-scratch", action="store_true")
    args, forwarded = parser.parse_known_args()
    driver = here.parent / (
        "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/"
        "parallel_20261018/mgather_physical_plan_retest_20261019"
    )
    sys.path.insert(0, str(driver))
    matrix = importlib.import_module("output_policy_matrix")
    original = matrix.run_matrix

    def audited_matrix(**kwargs):
        existing = kwargs["existing"]
        report = {
            "candidate_buffers": {key: describe(tensor) for key, tensor in existing.buffers.items()},
            "shared_native": {key: describe(tensor) for key, tensor in existing.native.items()},
            "reference_allocations": {},
            "device_values_read": False,
            "allocation_behavior_changed": args.share_scratch,
            "shared_scratch_keys": [],
            "independent_protocol_keys": ["signal", "ffts"],
        }
        output = Path(kwargs["output"])
        allocator = kwargs["allocate"]

        def recorded_allocate(key, shape, dtype, device):
            base_key = key.removeprefix("spmd_reference/")
            if args.share_scratch and base_key not in ("signal", "ffts"):
                if base_key not in existing.allocations:
                    raise ValueError(f"两侧工作区名称不一致，拒绝共享：{base_key}")
                tensor = existing.allocations[base_key]
                if (tuple(tensor.shape) != tuple(shape)
                        or tensor.dtype != dtype or str(tensor.device) != str(device)):
                    raise ValueError(f"两侧工作区描述符不一致，拒绝共享：{base_key}")
                report["shared_scratch_keys"].append(base_key)
            else:
                tensor = allocator(key, shape, dtype, device)
            report["reference_allocations"][key] = describe(tensor)
            return tensor

        kwargs["allocate"] = recorded_allocate
        try:
            return original(**kwargs)
        finally:
            (output / "WORKSPACE_ADDRESSES.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )

    matrix.run_matrix = audited_matrix
    sys.argv = [str(here / "run_current_wait_full.py"), *forwarded]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        matrix.run_matrix = original


if __name__ == "__main__":
    main()
