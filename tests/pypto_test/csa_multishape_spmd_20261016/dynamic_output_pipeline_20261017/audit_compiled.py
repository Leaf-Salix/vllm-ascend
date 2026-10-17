# SPDX-License-Identifier: Apache-2.0
"""只读输出组合的既有 CPU 产物，核对真实写回、阶段顺序和动态 ABI。

本脚本不重新编译，不导入 PyPTO，不执行设备。CPP 顺序只能证明生成代码保留
了相应操作，不能代替 A3 核间缓存可见性、设备精度、图重放或性能验证。
"""

import argparse
import ast
import importlib.util
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(__file__).parent / "cpu_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "不覆盖历史证据"
    run = args.run.resolve()
    # 复用相邻审计的纯文本参数追踪器，不调用其 main、不导入任何设备库。
    parser_path = run.parent.parent / "dynamic_compressor_pipeline_20261017/audit_compiled.py"
    spec = importlib.util.spec_from_file_location("_output_gm_text_audit", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {"cube": run / "kernel/kernels/aic/cube.cpp", "vector": run / "kernel/kernels/aiv/vector.cpp"}
    texts = {name: path.read_text() for name, path in paths.items()}
    roles = {name: module.role_body(texts[name], name) for name in paths}
    cube, vector = roles["cube"], roles["vector"]
    cw, cp = cube["waits"], cube["publishes"]
    vw, vp = vector["waits"], vector["publishes"]
    assert len(cw) == 1 and len(cp) == 3 and len(vw) == 2 and len(vp) == 3
    bound = module.bounded
    bound(cube, "packed", "TLOAD", cp[0], cp[1])
    bound(cube, "wo_a", "TLOAD", cp[0], cp[1])
    bound(cube, "o_r", "TSTORE", cp[0], cp[1])
    bound(cube, "quant", "TLOAD", cw[0], cp[2])
    bound(cube, "wo_b", "TLOAD", cw[0], cp[2])
    bound(cube, "partials", "TSTORE", cw[0], cp[2])
    bound(vector, "o_r", "TLOAD", vw[0], vp[1])
    bound(vector, "quant", "TSTORE", vw[0], vp[1])
    bound(vector, "scales", "TSTORE", vw[0], vp[1])
    for name in ("partials", "residual", "weight_scale"):
        bound(vector, name, "TLOAD", vw[1], vp[2])
    for name in ("scales", "post", "comb"):
        bound(vector, name, "scalar_load", vw[1], vp[2])
    bound(vector, "output", "TSTORE", vw[1], vp[2])
    assert len(vector["accesses"]["output"]["TSTORE"]) == 4
    # 最后一个 INT8 store 是量化尾零；其真实写回必须早于 QUANT 发布。
    pad_store = vector["accesses"]["quant"]["TSTORE"][-1]
    assert pad_store["line"] == 2663
    pad_context = "\n".join(texts["vector"].splitlines()[2590:2663])
    assert "TEXPANDS(v408, v61);" in pad_context and "TCVT(v410, v408," in pad_context
    assert "const half v61 = 0.0f;" in vector["body"]
    # 每个角色仅一个代际加载/首次通知；发布前及等待后的业务缓存维护保留。
    cache_checks = {}
    for name, role in roles.items():
        signals = module.require_access(role, "signal", "scalar_load")
        assert len(signals) == 1 and signals[0]["line"] < role["publishes"][0]
        lines = texts[name].splitlines()
        for publication in role["publishes"][1:]:
            prefix = "\n".join(lines[publication - 42 : publication - 1])
            assert "ENTIRE_DATA_CACHE" in prefix and "dsb(DSB_DDR);" in prefix
            assert "pipe_barrier(PIPE_ALL);" in prefix
        for wait in role["waits"]:
            suffix = "\n".join(lines[wait : wait + 15])
            assert "ENTIRE_DATA_CACHE" in suffix and "dsb(DSB_DDR);" in suffix
        cache_checks[name] = {
            "generation_scalar_load_line": signals[0]["line"],
            "begin_notify_line": role["publishes"][0],
            "phase_notify_lines": role["publishes"][1:],
            "wait_lines": role["waits"],
            "publish_pipeline_drain_and_dcci_dsb_present": True,
            "post_wait_dcci_dsb_present": True,
        }
    ca, va = list(cube["arguments"].values()), list(vector["arguments"].values())
    assert ca[:18] == va[:18] and len(ca) == 20 and len(va) == 21
    assert sum("__ssa_" in arg for arg in ca) == 13
    dimensions = [arg for arg in ca if arg.startswith("OUT_PIPE_")]
    assert len(dimensions) == 5
    result = json.loads((run / "result.json").read_text())
    metadata = json.loads((run / "kernel/compiled_meta.json").read_text())
    assert result["passed"] and len(metadata["params"]) == 13
    orchestration = "\n".join(p.read_text() for p in (run / "kernel/orchestration").glob("*.cpp"))
    assert len(re.findall(r"\brt_submit_\w+\s*\(", orchestration)) == 1
    assert "MixedKernels" in orchestration and "set_require_sync_start(true)" in orchestration
    # 审计运行所对应快照；当前组合脚本与两个正式数学 helper 的 AST 未被替换。
    package = run.parents[4] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    unchanged = []
    for name in (
        "probe.py",
        "output_projection_a_stages.py",
        "output_projection_b_stages.py",
        "phase_sync.py",
        "signals.py",
    ):
        current = run.parent / name if name == "probe.py" else package / name
        assert ast.dump(ast.parse(current.read_text())) == ast.dump(ast.parse((run / name).read_text())), name
        unchanged.append(name)
    report = {
        "passed": True,
        "scope": "只读既有 CPU 生成代码的 ABI/写回/等待顺序审计；非设备或完整 CSA 验证",
        "artifact": str(run / "kernel"),
        "shared_tensor_arguments": 13,
        "dynamic_descriptor_dimensions": dimensions,
        "incore_cpp_arguments": {"cube": 20, "vector": 21},
        "single_mix_submission": True,
        "sync_start": True,
        "source_ast_matches_compiled_snapshot": unchanged,
        "roles": {
            name: {key: value for key, value in role.items() if key not in ("body", "base_line")}
            for name, role in roles.items()
        },
        "cache_order_checks": cache_checks,
        "quant_padding_store": pad_store,
        "all_four_hc_residual_output_stores_preserved": True,
        "verified_order": [
            "AIC O_A 实际 FP32 写回→phase 1；AIV 等待→FP32 读取",
            "AIV INT8/scale/尾零实际写回→phase 2；AIC 等待→INT8 读取",
            "AIC O_B partial 实际 INT32 写回→phase 3；AIV 等待→partial/scale 读取",
            "AIV 反量化及 HCpost 的四路 BF16 实际写回→phase 4",
        ],
        "device_execution": False,
        "hardware_cache_visibility_proven": False,
        "device_precision_validated": False,
        "full_csa_validated": False,
        "performance_measured": False,
        "limits": [
            "CPU CPP 顺序不证明 A3 运行时跨核可见性或设备精度",
            "真实 owner 与等待图由 check_phase_model.py 单独核对，不以 TWAIT 数量替代",
            "20/21 为 13 Tensor+5 描述符动态轴+2/3 worker 的 CPP 形参，不是 dispatch 槽数",
            "只审计 NZ、RB96/N256 的输出后段组合；输入 packed/post/comb/residual 须先就绪",
        ],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "tensors": 13, "dynamic_axes": 5, "real_gm_stores_preserved": True}))


if __name__ == "__main__":
    main()
