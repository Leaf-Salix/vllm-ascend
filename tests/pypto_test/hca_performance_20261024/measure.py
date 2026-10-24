# SPDX-License-Identifier: Apache-2.0
"""按CSA规范采HCA两种边界；外部与L1均执行同一真实服务入口，不捕获PTO图。"""

import importlib
import importlib.metadata
import json
import os
import statistics
from pathlib import Path


def measure(*, package, layer, call, config, fixture, output, baseline, restore, collect,
            output_dir, kernel_config, samples):
    import pypto.torch
    import torch
    import torch_npu
    from dsv4_csa_single_layer import guard_checks
    from dsv4_csa_validation import compare_tensor
    from csa_mpmd_tnd_20261021.run_case import export_l1
    from hca_tnd_mpmd_20261022.profile_eager import profile_call
    from hca_tnd_mpmd_20261022.service_probe import validate
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    service = importlib.import_module(package + ".service")
    proof = validate(package=package, layer=layer, call=call, config=config, fixture=fixture,
                     output=output, baseline=baseline, restore=restore, collect=collect)
    if not proof["passed"]:
        raise AssertionError("真实服务未接管或按步compact metadata/KV钩子验证失败")
    service.bind_layer(layer, call.operators, max_num_seqs=64, max_num_batched_tokens=384)

    def run():
        # 每窗新建forward context，真实compact生产与描述符绑定不能跨步缓存掉。
        with set_ascend_forward_context(fixture["metadata"], config, num_tokens=fixture["tokens"],
                                       num_actual_tokens=fixture["actual_tokens"]):
            torch.ops.vllm.dsv4_hca_forward(fixture["hidden"], fixture["positions"], output,
                                          layer.self_attn.dsa_attn.prefix)

    def reset():
        restore(fixture)
        output.fill_(float("nan"))

    def check():
        actual = collect()
        checks = {"output": compare_tensor(actual["output"], baseline["output"], 0, 0)}
        for name in fixture["groups"]:
            checks[name] = [compare_tensor(a, b, 0, 0) for a, b in zip(actual[name], baseline[name])]
        checks["guards"] = guard_checks(fixture)

        def passed(value):
            if isinstance(value, list):
                return all(passed(v) for v in value)
            if "status" in value:
                return value["status"] == "PASS"
            return all(passed(v) for v in value.values())

        if not passed(checks):
            (output_dir / "invalid_window_checks.json").write_text(json.dumps(checks, indent=2))
            raise AssertionError("规范计时窗口结果/状态/保护区不再保持基线")
        return {"output_state_exact": True, "guards_pass": True}

    outer = profile_call(run, reset, output_dir=output_dir / "profile_pto_service",
                         side="pto_service", samples=samples, validate=check)
    l1 = []
    for _ in range(3):
        reset()
        run()
        torch.npu.synchronize()
    for index in range(samples):
        reset()
        torch.npu.synchronize()
        pypto.torch.begin_dfx()
        try:
            run()
        finally:
            pypto.torch.end_dfx()
        raw = output_dir / "dfx"
        if index:
            raw /= f"window_{index}"
        window = export_l1(raw, output_dir / f"l1_{index:02d}", "mpmd_hca", kernel_config)
        window["checks"] = check()
        l1.append(window)
    values = [w["incore_span_us"] for w in l1]
    return {
        "service_validation": proof,
        "execution_mode": "eager_direct", "aclgraph_replay": False,
        "warmup_per_metric": 3, "calls_per_window": 1,
        "separate_capture_reason": "外部Profiler与Simpler L1分开采集，避免采集器嵌套；同进程同fixture同服务调用",
        "aicpu": outer,
        "l1": {"samples_us": values, "median_us": statistics.median(values), "windows": l1},
        "l1_scope": "完整PTO整层所有incore首尾；其外的Native compact producer属于外部AICPU列，不是PTO incore",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "vllm")},
        "device_name": torch.npu.get_device_name(),
        "cann": os.environ["ASCEND_HOME_PATH"], "custom_opp": os.environ["ASCEND_CUSTOM_OPP_PATH"],
        "shared_manifest": json.loads(Path(os.environ["CSA_SHARED_TOOLCHAIN_MANIFEST"]).read_text()),
        "native_weight_formats": {name: int(torch_npu.get_npu_format(getattr(layer.self_attn, name).weight))
                                  for name in ("wq_a", "wq_b", "wo_a", "wo_b")},
    }
