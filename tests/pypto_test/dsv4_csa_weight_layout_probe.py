# SPDX-License-Identifier: Apache-2.0
"""单卡核对 Native 格式转换与 pypto-lib NZ 字节合同；不加载模型或编译 CSA。"""

import argparse
import ast
import ctypes
import math
import os
from pathlib import Path

from dsv4_csa_env import activate, write_json
from dsv4_csa_replay import capture_tensors, materialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypto-lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assigned = os.environ["TASK_DEVICE"]
    if "," in assigned:
        raise ValueError("本探针只需要单卡")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = assigned
    activate()
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    source = args.pypto_lib / "models/deepseek_v4_flash_dspark/utils.py"
    # 直接执行上游纯打包函数，避免导入整个模型 config 的命令行副作用。
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name == "pack_nz"
    ) or (
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in ("NZ_C0_BYTES", "NZ_FRACTAL_ROWS")
            for target in node.targets
        )
    )]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    pack_nz = namespace["pack_nz"]

    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    acl.aclrtMemcpy.restype = ctypes.c_int

    def describe(value):
        return {"shape": list(value.shape), "stride": list(value.stride()),
                "format": int(torch_npu.get_npu_format(value)),
                "storage_bytes": value.untyped_storage().nbytes(), "contiguous": value.is_contiguous()}

    report = {"scope": "A3/CANN9：2D/3D、BF16/INT8 的小张量布局验证", "source": str(source), "cases": []}
    for dtype in (torch.bfloat16, torch.int8):
        for shape in ((32, 64), (2, 32, 64)):
            logical = (torch.arange(math.prod(shape)).reshape(shape) % 127 - 63).to(dtype)
            reference = pack_nz(logical)
            device = torch_npu.npu_format_cast(logical.to("npu:0"), 2)
            native = torch_npu.npu_format_cast(device, 29)
            device_packed = pack_nz(device)
            base_packed = torch_npu.npu_format_cast(device_packed, 2)
            torch.npu.synchronize()
            # 只读取物理字节；Tensor.cpu() 会按 Native 格式解码，不能用来比较 NZ 存储。
            raw = torch.empty_like(reference)
            nbytes = raw.numel() * raw.element_size()
            if native.untyped_storage().nbytes() != nbytes or native.storage_offset() != 0:
                raise ValueError("本小用例要求无 padding、完整独占 NZ 存储")
            code = acl.aclrtMemcpy(raw.data_ptr(), nbytes, native.data_ptr(), nbytes, 2)
            if code:
                raise RuntimeError(f"aclrtMemcpy D2H 失败：{code}")
            case = {"dtype": str(dtype), "logical": describe(device), "native": describe(native),
                    "device_packed": describe(device_packed), "base_packed": describe(base_packed),
                    "native_raw_equals_pypto_lib": torch.equal(raw, reference),
                    "device_pack_equals_pypto_lib": torch.equal(base_packed.cpu(), reference),
                    "native_logical_roundtrip": torch.equal(native.cpu(), logical)}
            meta, payload = capture_tensors({"weight": native}, {"weight": "in"}, {"weight": "NZ"},
                                            {"state_timing": "before_call"})
            replay, _ = materialize(meta, payload)
            case["snapshot_raw_equals_pypto_lib"] = torch.equal(replay["weight"], reference)
            report["cases"].append(case)
    report["status"] = "PASS" if all(
        case["native_raw_equals_pypto_lib"] and case["device_pack_equals_pypto_lib"]
        and case["native_logical_roundtrip"] and case["base_packed"]["format"] in (0, 2)
        and case["snapshot_raw_equals_pypto_lib"]
        for case in report["cases"]
    ) else "FAIL"
    write_json(args.output, report)
    print(report["status"])
    if report["status"] != "PASS":
        raise ValueError("Native/pypto-lib NZ 布局合同不一致，见报告")


if __name__ == "__main__":
    main()
