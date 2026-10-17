# SPDX-License-Identifier: Apache-2.0
"""验证 inline Group 的注册、单 MIX 提交、当前流和双 AIV；不是 CSA 精度测试。"""

import argparse
import json
import os
import re
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import torch
import pypto.torch
from pypto.runtime import CompileOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--shared-inout", action="store_true")
    parser.add_argument("--check-sdk-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"scope": "5 参数 JIT inline Group 接口验证", "passed": False,
              "device_execution": False, "full_csa_validated": False}
    if args.shared_inout:
        from repro_jit_group_shared_inout import entry
    else:
        from repro_jit_group_inline import entry
    result["shared_inout"] = args.shared_inout
    try:
        config = CompileOptions(platform="a2a3", dump_passes=True,
                                output_dir=str((args.output / "kernel").resolve()))
        op = pypto.torch.register(entry, "csa_static_probe::inline_group", config=config)
        meta = [torch.empty(shape, device="meta", dtype=dtype) for shape, dtype in (
            ((384, 16), torch.bfloat16), ((16, 16), torch.bfloat16),
            ((384, 16), torch.float32), ((384, 16), torch.float32),
            ((384, 16), torch.float32))]
        meta_result = op(*meta)
        assert meta_result[0] is meta[2] and meta_result[1] is meta[4]
        result.update(registered=True, metadata_aliases_passed=True, schema=str(op._schema))
        if args.metadata_only:
            result["passed"] = True
            return

        # 必须来自独立扩容 worktree，禁止混用共享环境的旧 native binding。
        import _task_interface
        import simpler
        import torch_npu
        from pypto import _torch_npu as native_adapter
        from pypto import pypto_core
        from pypto._kernel_abi import SIMPLER_KERNEL_REVISION

        expected = Path(os.environ["HCA_SIMPLER_ROOT"]).resolve()
        assert expected in Path(simpler.__file__).resolve().parents
        assert expected in Path(_task_interface.__file__).resolve().parents
        expected_pypto = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]).resolve()
        assert expected_pypto in Path(native_adapter.__file__).resolve().parents
        assert expected_pypto in Path(pypto_core.__file__).resolve().parents
        assert native_adapter.simpler_revision == _task_interface.__build_commit__ == SIMPLER_KERNEL_REVISION
        result.update(simpler_source=simpler.__file__, simpler_binding=_task_interface.__file__,
                      torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                      pypto_binding=pypto_core.__file__, torch_adapter=native_adapter.__file__,
                      simpler_revision=SIMPLER_KERNEL_REVISION)
        if args.check_sdk_only:
            result.update(passed=True, sdk_only=True)
            return
        torch.npu.set_device(args.device)
        pypto.torch.init(device=args.device, platform="a2a3", runtime="host_build_graph")
        device = torch.device(f"npu:{args.device}")
        stream = torch.npu.Stream(device=device)
        guard = 32
        count = 384 * 16
        sentinel = -12345.0
        # 小整数使 BF16 输入和 FP32 参考累加均精确可表示，避免混入数值阈值问题。
        a_cpu = ((torch.arange(count).reshape(384, 16) % 7) - 3).to(torch.bfloat16)
        b_cpu = ((torch.arange(256).reshape(16, 16) % 5) - 2).to(torch.bfloat16)
        x_cpu = (torch.arange(count).reshape(384, 16) % 17).float()
        result["launches"] = []
        with torch.npu.stream(stream):
            a, b, x = a_cpu.to(device), b_cpu.to(device), x_cpu.to(device)
            c_all = torch.full((count + 2 * guard,), sentinel, device=device)
            y_all = torch.full_like(c_all, sentinel)
            c = c_all[guard:-guard].view(384, 16)
            y = y_all[guard:-guard].view(384, 16)
            for launch in range(3):
                # 更新和算子同流入队，检验 current-stream 传递，保持输出地址不变。
                x.copy_((x_cpu + launch).to(device))
                c.fill_(sentinel)
                y.fill_(sentinel)
                actual = op(a, b, c, x, y)
                assert actual[0] is c and actual[1] is y
                stream.synchronize()
                expected_c = torch.full((count + 2 * guard,), sentinel)
                expected_y = expected_c.clone()
                expected_c[guard:-guard] = (a_cpu.float() @ b_cpu.float()).flatten()
                expected_y[guard:-guard] = ((x_cpu + launch) * 2).flatten()
                torch.testing.assert_close(c_all.cpu(), expected_c, rtol=0, atol=0)
                torch.testing.assert_close(y_all.cpu(), expected_y, rtol=0, atol=0)
                result["launches"].append({"index": launch, "output_and_guards_passed": True})
            torch.testing.assert_close(a.cpu(), a_cpu, rtol=0, atol=0)
            torch.testing.assert_close(b.cpu(), b_cpu, rtol=0, atol=0)
            torch.testing.assert_close(x.cpu(), x_cpu + 2, rtol=0, atol=0)

        paths = list((args.output / "kernel").rglob("*.cpp"))
        sources = [p for p in paths if "orchestration" in p.parts]
        code = "\n".join(p.read_text() for p in sources)
        calls = re.findall(r"\brt_submit_\w+\s*\(", code)
        assert len(calls) == 1 and "MixedKernels" in code, calls
        assert "set_block_num(24)" in code and "set_require_sync_start(true)" in code
        result.update(passed=True, device_execution=True, runtime="host_build_graph",
                      submissions=calls, single_mix_submission=True, block_num=24,
                      two_aiv_lanes_passed=True, current_stream_passed=True,
                      inputs_unchanged=True, stream_id=int(stream.stream_id),
                      orchestration_sources=[str(p.resolve()) for p in sources])
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
