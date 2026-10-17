# SPDX-License-Identifier: Apache-2.0
"""动态 HC→Q_A→QR 的有界功能验证，当前不是完整 CSA，也不测性能。

输入为正式第 4 层 safetensors 的六份权重和可复现合成 BF16 hidden；两边
都使用同一 ND wq_a、BF16 norm，不调用 Native、不转 NZ。checkpoint 中两份
norm 是 FP32，按现有 make_layer 的 BF16 参数 copy_ 语义转成 BF16；明确记录
文件 dtype 与运行 dtype，不把 checkpoint 误称为 BF16，也不升级算子 norm 精度。
候选加载相邻 dynamic_front_pipeline_20261017/probe.py，参考由当前原动态数学
源码生成，仍为多个 TMR task。实际 T=168/169 分别覆盖完整八行块与一行尾块。

每个 T 分配一套固定地址，候选顺序执行 A→B→A；输入和输出内容每次重写，
signal 只在首次清零。逐 bit 检查有效输出、检查约定可写 padding 之外的哨兵，
最后核对 72 核实际发布的阶段槽和其余空槽。512 字节槽不相互重叠，所有调用
串行完成后才允许覆写业务数据。脚本不把 padding 值视为有效模型输出。

默认仅 CPU 准备；只有显式 --execute 且队列提供 TASK_DEVICE 才运行 NPU。
--compile-reference 仅编译高层动态参考，候选已完成的 CPU 编译不重复执行。
"""

import argparse
import ast
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path

from prepare_reference import HERE, MATH, PROBE, REPO, build

CHECKPOINT = Path("/data/model/DeepSeek-V4-Flash-0731-w8a8")
WEIGHT_KEYS = {
    "hc_fn": ("hc_attn_fn", (24, 16384), "float32"),
    "hc_scale": ("hc_attn_scale", (3,), "float32"),
    "hc_base": ("hc_attn_base", (24,), "float32"),
    "norm_w": ("attn_norm.weight", (4096,), "bfloat16"),
    "wq_a": ("attn.wq_a.weight", (1024, 4096), "bfloat16"),
    "gamma_cq": ("attn.q_norm.weight", (1024,), "bfloat16"),
}
ORDER = ("x", "hc_fn", "hc_scale", "hc_base", "hc_base_2d", "norm_w", "wq_a", "gamma_cq",
         "x_fp32", "inv", "widen_tail", "partials", "raw", "pre", "post", "post_tail",
         "comb", "comb_tail", "mixed", "normed", "norm_tail", "qa", "qr", "qr_scale",
         "qr_matmul", "qr_scale_pad", "signal")
INPUT_NAMES = ("x", *WEIGHT_KEYS)
WORDS = 9 * 72 * 128
SENTINEL = -73


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_report(output, report):
    temporary = output / "result.tmp.json"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "result.json")
    print(json.dumps({key: report.get(key) for key in ("stage", "passed", "device_execution")},
                     ensure_ascii=False), flush=True)


def load_formal_weights(torch, checkpoint, layer):
    """直接读正式权重；两份 norm 匹配 make_layer 的 BF16 参数加载语义。"""
    from safetensors import safe_open

    description = json.loads((checkpoint / "quant_model_description.json").read_text())
    assert description["model_quant_type"] == "W8A8_DYNAMIC"
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    weights, records = {}, []
    for name, (suffix, shape, dtype) in WEIGHT_KEYS.items():
        key = f"layers.{layer}.{suffix}"
        shard = checkpoint / index[key]
        with safe_open(shard, framework="pt", device="cpu") as reader:
            value = reader.get_tensor(key)
        assert tuple(value.shape) == shape, (key, value.shape, shape)
        checkpoint_dtype = str(value.dtype)
        if name in ("norm_w", "gamma_cq"):
            assert value.dtype == torch.float32, (key, value.dtype)
            # make_layer 先在 BF16 默认 dtype 下构造 RMSNorm，再 copy_ 正式 FP32
            # checkpoint；这里显式复现该加载边界，候选与参考共用转换后的同一张量。
            original = value
            value = value.to(torch.bfloat16)
            roundtrip_equal = torch.equal(original, value.float())
        else:
            assert value.dtype == getattr(torch, dtype), (key, value.dtype, dtype)
            roundtrip_equal = True
        assert value.is_contiguous() and bool(torch.isfinite(value).all())
        weights[name] = value
        records.append({"input": name, "key": key, "shard": str(shard), "shape": list(shape),
                        "checkpoint_dtype": checkpoint_dtype, "runtime_dtype": dtype, "layout": "ND",
                        "dtype_conversion": name in ("norm_w", "gamma_cq"),
                        "roundtrip_equal": roundtrip_equal,
                        "loading_contract": ("dsv4_csa_single_layer.make_layer BF16 parameter copy_"
                                             if name in ("norm_w", "gamma_cq") else "保持 checkpoint dtype")})
    return weights, records


def schema(tokens, reference=False):
    """逻辑有效 T 与内部容量分别给出；参考的 QR 暂存沿用原 512 行合同。"""
    p8, p16 = (tokens + 7) // 8 * 8, (tokens + 15) // 16 * 16
    qr_capacity = 512 if reference else p8
    return {
        "x": ((tokens, 16384), "bfloat16"),
        **{name: (shape, dtype) for name, (_, shape, dtype) in WEIGHT_KEYS.items()},
        "x_fp32": ((tokens, 16384), "float32"), "inv": ((p16, 1), "float32"),
        "widen_tail": ((8, 16384), "float32"), "partials": ((4 * p16, 32), "float32"),
        "raw": ((p16, 32), "float32"), "pre": ((p16, 8), "float32"),
        "post": ((tokens, 4), "float32"), "post_tail": ((8, 8), "float32"),
        "comb": ((tokens, 16), "float32"), "comb_tail": ((8, 32), "float32"),
        "mixed": ((p8, 4096), "bfloat16"), "normed": ((tokens, 4096), "bfloat16"),
        "norm_tail": ((8, 4096), "bfloat16"), "qa": ((p16, 1024), "float32"),
        "qr": ((tokens, 1024), "int8"), "qr_scale": ((tokens, 1), "float32"),
        "qr_matmul": ((qr_capacity, 1024), "int8"), "qr_scale_pad": ((qr_capacity, 1), "float32"),
        "signal": ((WORDS,), "int32"),
    }


class Fixture:
    """每份独立 allocation 两侧各留 512 字节；hc_base 的 2D view 共享同一存储。"""

    def __init__(self, torch, tokens, device, weights, reference=False):
        self.torch, self.tokens, self.reference = torch, tokens, reference
        self.buffers, self.allocations, self.guards = {}, {}, {}
        for name, (shape, dtype_name) in schema(tokens, reference).items():
            dtype = getattr(torch, dtype_name)
            guard = 512 // torch.empty((), dtype=dtype).element_size()
            allocation = torch.full((math.prod(shape) + 2 * guard,), SENTINEL, dtype=dtype, device=device)
            self.allocations[name], self.guards[name] = allocation, guard
            self.buffers[name] = allocation[guard:-guard].view(shape)
        self.buffers["hc_base_2d"] = self.buffers["hc_base"].view(1, 24)
        assert self.buffers["hc_base_2d"].data_ptr() == self.buffers["hc_base"].data_ptr()
        for name, weight in weights.items():
            self.buffers[name].copy_(weight)
        self.buffers["signal"].zero_()
        self.addresses = {name: value.data_ptr() for name, value in self.buffers.items()}

    def prepare_input(self, hidden):
        # 只重置 payload，绝不重置保护区；signal 跨三次调用保留，才能暴露代际错误。
        for name, value in self.buffers.items():
            if name not in INPUT_NAMES and name not in ("hc_base_2d", "signal"):
                value.fill_(SENTINEL)
        self.buffers["x"].copy_(hidden)
        assert self.addresses == {name: value.data_ptr() for name, value in self.buffers.items()}

    def arguments(self):
        return tuple(self.buffers[name] for name in ORDER)

    def snapshot(self):
        return {name: value.cpu().clone() for name, value in self.buffers.items()}

    def guards_and_unwritten(self):
        checks = {}
        for name, allocation in self.allocations.items():
            guard = self.guards[name]
            checks[name + ":prefix"] = bool((allocation[:guard].cpu() == SENTINEL).all())
            checks[name + ":suffix"] = bool((allocation[-guard:].cpu() == SENTINEL).all())
        p8 = (self.tokens + 7) // 8 * 8
        # inv/pre 仅写 ceil(T/8) 行，QA/raw/partials 可写 ceil(T/16) 行。
        # staging 的最后八行属于合法可写 padding，不误报为保护区破坏。
        for name in ("inv", "pre", "qr_matmul", "qr_scale_pad"):
            checks[name + ":rows_after_p8"] = bool((self.buffers[name][p8:].cpu() == SENTINEL).all())
        if self.tokens % 8 == 0:
            for name in ("widen_tail", "post_tail", "comb_tail", "norm_tail"):
                checks[name + ":unused_tail"] = bool((self.buffers[name].cpu() == SENTINEL).all())
        return {"passed": all(checks.values()), "checks": checks}


def exact(torch, actual, expected):
    """逐 bit 比较同时给出 FP32 数值差，+0/-0 差异不会被 torch.equal 隐藏。"""
    actual, expected = actual.contiguous(), expected.contiguous()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    bytes_a, bytes_b = actual.view(torch.uint8), expected.view(torch.uint8)
    bit_equal = torch.equal(bytes_a, bytes_b)
    unequal = actual != expected
    delta = (actual.float() - expected.float()).abs()
    finite = bool(torch.isfinite(actual.float()).all() and torch.isfinite(expected.float()).all())
    return {"passed": bit_equal and finite, "bitwise_equal": bit_equal, "finite": finite,
            "elements": actual.numel(), "numeric_different": int(unequal.sum()),
            "bytes_different": int((bytes_a != bytes_b).sum()),
            "max_abs": float(delta.max()) if delta.numel() else 0.0,
            "rmse": float(delta.square().mean().sqrt()) if delta.numel() else 0.0}


def effective_outputs(values, tokens):
    """限定业务有效区；raw/partial 的无效 8 列、所有尾块 pad 不作为精度判据。"""
    p16 = (tokens + 15) // 16 * 16
    selected = {name: values[name][:tokens] for name in
                ("x_fp32", "inv", "pre", "post", "comb", "mixed", "normed", "qa", "qr", "qr_scale",
                 "qr_matmul", "qr_scale_pad")}
    selected["pre"] = selected["pre"][:, :4]
    selected["raw"] = values["raw"][:tokens, :24]
    for split in range(4):
        selected[f"partials_split{split}"] = values["partials"][split * p16:split * p16 + tokens, :24]
    return selected


def compare_outputs(torch, actual, expected, tokens):
    left, right = effective_outputs(actual, tokens), effective_outputs(expected, tokens)
    checks = {name: exact(torch, left[name], right[name]) for name in left}
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def signal_check(torch, signal, generation):
    expected = torch.zeros((9, 72, 128), dtype=torch.int32)
    expected[0, :, 0] = generation
    for phase in (2, 7):
        expected[phase, :24, 0] = generation
    for phase in (1, 3, 4, 5, 6, 8):
        expected[phase, 24:, 0] = generation
    # 数学空核也会执行该角色的 publish；另一角色的空 phase 槽必须一直为零。
    result = exact(torch, signal.reshape(9, 72, 128), expected)
    result["generation"] = generation
    result["observed_phase_slots"] = signal.reshape(9, 72, 128)[:, :, 0].tolist()
    return result


def verify_isolation(torch):
    import _task_interface
    import simpler
    import torch_npu
    from pypto import _torch_npu, pypto_core
    from pypto._kernel_abi import SIMPLER_KERNEL_REVISION

    pypto_root = Path(os.environ["CSA_STATIC_PYPTO_ROOT"]).resolve()
    simpler_root = Path(os.environ["HCA_SIMPLER_ROOT"]).resolve()
    for module in (pypto_core, _torch_npu):
        assert pypto_root in Path(module.__file__).resolve().parents
    for module in (simpler, _task_interface):
        assert simpler_root in Path(module.__file__).resolve().parents
    assert _torch_npu.simpler_revision == _task_interface.__build_commit__ == SIMPLER_KERNEL_REVISION
    return {"pypto_binding": pypto_core.__file__, "torch_adapter": _torch_npu.__file__,
            "simpler_binding": _task_interface.__file__, "simpler_revision": SIMPLER_KERNEL_REVISION,
            "torch": torch.__version__, "torch_npu": torch_npu.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--compile-reference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "stage": "prepare", "device_execution": False,
              "performance_measured": False, "full_csa_validated": False,
              "scope": "有界 HC→QA→QR，T168/169，正式权重、合成 hidden、ND 布局",
              "precision_gate": "有效区逐 bit；未修改门槛", "cases": [],
              "candidate_source": str(PROBE), "reference_math_source": str(MATH),
              "native_used": False, "frozen_ir_used": False}
    try:
        if os.environ.get("VLLM_ASCEND_ENABLE_NZ") != "1":
            raise RuntimeError("本 ND probe 必须在导入数学前设置 VLLM_ASCEND_ENABLE_NZ=1")
        if os.environ.get("VLLM_ASCEND_PTO_CSA_ATOMIC_ADD") != "0":
            raise RuntimeError("此同源前段要求固定 K 顺序：VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0")
        import dsv4_csa_env
        dsv4_csa_env.activate()
        import torch
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import CompileOptions, RunConfig

        weights, report["weights"] = load_formal_weights(torch, args.checkpoint, args.layer)
        report["weight_loading_source"] = {
            "bf16_parameter_context": "tests/pypto_test/dsv4_csa_single_layer.py:64（默认BF16构造层）",
            "input_norm_copy": "tests/pypto_test/dsv4_csa_single_layer.py:77（parameter.data.copy_）",
            "attention_parameter_loader": "tests/pypto_test/dsv4_csa_formal_weights.py:58",
            "rmsnorm_default_dtype": str(REPO.parent / ".cache/migration-v0.25.1rc1/vllm/vllm/model_executor/layers/layernorm.py") + ":61",
            "default_weight_loader_copy": str(REPO.parent / ".cache/migration-v0.25.1rc1/vllm/vllm/model_executor/model_loader/weight_utils.py") + ":1212",
            "q_norm_construct": "vllm_ascend/models/deepseek_v4.py:759",
            "explanation": "两份FP32 checkpoint norm按已有BF16参数copy_语义加载；不会把Native BF16参数升级为FP32",
        }
        # 留下本次实际 import 的高层源快照，便于将来审查；执行仍引用当前正式包。
        snapshot = args.output / "source_snapshot"
        snapshot.mkdir()
        for path in (Path(__file__), HERE / "prepare_reference.py", PROBE):
            (snapshot / path.name).write_text(path.read_text())
        package = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
        for name in ("hc_stages.py", "qkv_stages.py", "phase_sync.py", "signals.py"):
            (snapshot / name).write_text((package / name).read_text())
        reference_path, report["reference_audit"] = build(args.output)
        reference = load_module("front_original_dynamic_reference", reference_path)
        candidate = load_module("front_candidate_probe", PROBE)
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode import BF16_WEIGHT_NZ
        from vllm_ascend.ops.pypto.deepseek_v4_flash_csa.reduction import ATOMIC_ADD
        assert not BF16_WEIGHT_NZ and ATOMIC_ADD == 0
        probe_entry = next(node for node in ast.parse(PROBE.read_text()).body
                           if isinstance(node, ast.FunctionDef) and node.name == "entry")
        assert tuple(argument.arg for argument in probe_entry.args.args) == ORDER
        logical = python_print(reference.entry.specialize())
        (args.output / "reference_symbolic.py").write_text(logical)
        assert "REFERENCE_T" in logical and "pl.spmd" in logical
        report["reference_symbolic_passed"] = True
        if args.compile_reference:
            artifact = reference.entry.compile(config=RunConfig(
                platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output / "reference_kernel"),
                memory_planner=passes.MemoryPlanner.PYPTO))
            report["reference_full_codegen_passed"] = True
            report["reference_compile_output"] = str(artifact.output_dir)
        for tokens in (168, 169):
            cpu = Fixture(torch, tokens, torch.device("cpu"), weights)
            inputs = [torch.randn((tokens, 16384), generator=torch.Generator().manual_seed(20261017 + tokens + variant),
                                  dtype=torch.float32).mul_(0.1).to(torch.bfloat16) for variant in (0, 1000)]
            cpu.prepare_input(inputs[0])
            assert cpu.guards_and_unwritten()["passed"]
            assert not torch.equal(inputs[0], inputs[1])
            report["cases"].append({"tokens": tokens, "pad8": (tokens + 7) // 8 * 8,
                                    "pad16": (tokens + 15) // 16 * 16, "shape_gate_passed": True,
                                    "candidate_shapes": {name: list(value.shape) for name, value in cpu.buffers.items()},
                                    "reference_qr_capacity": 512, "sequence": ["A", "B", "A"], "launches": []})
        report["cpu_preparation_passed"] = True
        write_report(args.output, report)
        if not args.execute:
            report.update(stage="prepared_not_executed", passed=True)
            return
        if "TASK_DEVICE" not in os.environ:
            raise RuntimeError("设备验证必须经 task-submit --device auto；未提供 TASK_DEVICE")
        device_id = int(os.environ["TASK_DEVICE"])
        report["environment"] = verify_isolation(torch)
        import pypto.torch
        torch.npu.set_device(device_id)
        torch.npu.config.allow_internal_format = True
        pypto.torch.init(device=device_id, platform="a2a3", runtime="tensormap_and_ringbuffer",
                         ring_heap=(256 * 1024 * 1024,) * 4, ring_task_window=(4096,) * 4)
        options = CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO)
        report.update(stage="device_running", device_execution=True, device=device_id)
        with torch.inference_mode():
            for case in report["cases"]:
                tokens = case["tokens"]
                fixture = Fixture(torch, tokens, torch.device(f"npu:{device_id}"), weights)
                baseline = Fixture(torch, tokens, torch.device(f"npu:{device_id}"), weights, reference=True)
                assert fixture.buffers["signal"].data_ptr() % 512 == 0
                case["signal_address_alignment"] = 512
                hidden = {label: torch.randn((tokens, 16384),
                          generator=torch.Generator().manual_seed(20261017 + tokens + variant),
                          dtype=torch.float32).mul_(0.1).to(torch.bfloat16)
                          for label, variant in (("A", 0), ("B", 1000))}
                references = {}
                for label in ("A", "B"):
                    baseline.prepare_input(hidden[label])
                    reference.entry(*baseline.arguments(), config=options)
                    torch.npu.synchronize()
                    references[label] = baseline.snapshot()
                    guards = baseline.guards_and_unwritten()
                    unchanged = {name: exact(torch, references[label][name], hidden[label] if name == "x" else weights[name])
                                 for name in INPUT_NAMES}
                    case.setdefault("reference_checks", {})[label] = {
                        "guards": guards, "inputs_unchanged": unchanged}
                    assert guards["passed"] and all(check["passed"] for check in unchanged.values())
                    write_report(args.output, report)
                first_a = None
                for generation, label in enumerate(("A", "B", "A"), 1):
                    fixture.prepare_input(hidden[label])
                    candidate.entry(*fixture.arguments(), config=options)
                    torch.npu.synchronize()
                    actual = fixture.snapshot()
                    launch = {"label": label, "generation": generation,
                              "outputs": compare_outputs(torch, actual, references[label], tokens),
                              "guards": fixture.guards_and_unwritten(),
                              "signals": signal_check(torch, actual["signal"], generation),
                              "inputs_unchanged": {name: exact(torch, actual[name], hidden[label] if name == "x" else weights[name])
                                                   for name in INPUT_NAMES}}
                    if generation == 1:
                        first_a = actual
                    if generation == 3:
                        launch["repeat_a"] = compare_outputs(torch, actual, first_a, tokens)
                    launch["passed"] = (launch["outputs"]["passed"] and launch["guards"]["passed"]
                                        and launch["signals"]["passed"]
                                        and all(check["passed"] for check in launch["inputs_unchanged"].values())
                                        and launch.get("repeat_a", {"passed": True})["passed"])
                    case["launches"].append(launch)
                    write_report(args.output, report)
                    # 数值差异保留后续轮证据；内存/代际失败必须停，避免复用不可信协议状态。
                    if not launch["guards"]["passed"] or not launch["signals"]["passed"]:
                        raise RuntimeError(f"T{tokens}/{label} 保护区或信号失败；停止复用本次 signal")
                case["passed"] = all(item["passed"] for item in case["launches"])
        report.update(stage="device_complete", passed=all(case["passed"] for case in report["cases"]))
        if not report["passed"]:
            raise RuntimeError("有效区逐 bit 对照失败；见各阶段详细差异，未放宽门槛")
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc(), passed=False)
        raise
    finally:
        write_report(args.output, report)


if __name__ == "__main__":
    main()
