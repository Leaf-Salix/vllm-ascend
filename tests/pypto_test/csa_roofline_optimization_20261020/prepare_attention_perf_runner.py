# SPDX-License-Identifier: Apache-2.0
"""生成可审查的独立性能诊断入口，不修改原严格测试器。

用户在已知 Attention 输出有逐 bit 差异后明确要求性能对比。本入口保留
原数值 FAIL；只允许有限、同 shape/dtype 的 x_out 差异进入诊断采样。
所有其它输出、状态、保护区、代际仍严格相同，每版图重放必须与本版 eager
逐 bit 相同。诊断结束不将完整 CSA 精度或最终采用标为通过。
"""
import ast
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("原测试器文本改变，拒绝不明确替换：" + old[:100])
    return text.replace(old, new, 1)


def main():
    here = Path(__file__).resolve().parent
    original = (here.parent / "csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
                "/parallel_20261018/mgather_physical_plan_retest_20261019")
    source = (original / "run_device.py").read_text()
    doc_start = source.index('"""')
    doc_end = source.index('"""', doc_start + 3) + 3
    source = source[:doc_start] + '''"""用户明确授权的Attention性能诊断。

保留跨版数值FAIL，不放宽正式精度门槛；仅允许有限的x_out差异继续采样。
Top-K、状态、保护区与代际保持严格检查，每版图必须与自身eager逐bit一致。
最终passed仍由原数值门禁决定，diagnostic_completed只表示本次测量完成。
"""''' + source[doc_end:]
    source = replace_once(source, "HERE = Path(__file__).resolve().parent", f"HERE = Path({str(original)!r})")
    source = replace_once(source, "            baseline = {}\n",
                          "            baseline = {}\n            candidate_eager = {}\n")
    source = replace_once(
        source,
        '                report["launches"].append(checks)',
        '''                # 严格数值结果保持原样；诊断只允许x_out有限数值差异。
                if label not in candidate_eager:
                    candidate_eager[label] = actual["x_out"].clone()
                repeat = compare_tensor(actual["x_out"], candidate_eager[label], 0, 0)
                checks["candidate_repeat_exact"] = repeat
                checks["diagnostic_ready"] = (
                    all(c["status"] == "PASS" for name, c in comparisons.items() if name != "x_out")
                    and comparisons["x_out"].get("nonfinite") == 0
                    and comparisons["x_out"]["shape"] == comparisons["x_out"]["expected_shape"]
                    and comparisons["x_out"]["dtype"] == comparisons["x_out"]["expected_dtype"]
                    and repeat["status"] == "PASS"
                    and all(c["status"] == "PASS" for c in guards.values())
                    and all(scratch_guards.values()) and protocol
                )
                report["launches"].append(checks)'''
    )
    source = replace_once(source, '                if not checks["passed"]:\n',
                          '                if not checks["diagnostic_ready"]:\n')
    source = replace_once(source, "                def verify():\n",
                          "                def verify(expected_output=None):\n")
    source = replace_once(
        source,
        '                    compared = {name: compare_tensor(value, baseline["A"][name], 0, 0) for name, value in state.items()}',
        '''                    compared = {
                        name: compare_tensor(value,
                            expected_output if name == "x_out" and expected_output is not None
                            else baseline["A"][name], 0, 0)
                        for name, value in state.items()
                    }'''
    )
    source = replace_once(source, "                    from output_policy_matrix import run_matrix",
                          "                    from attention_perf_matrix import run_matrix")
    source = replace_once(source, "                        verify=verify,\n"
                          "                        cache_guards=",
                          "                        verify=verify,\n"
                          "                        collect_output=lambda: output.detach().cpu().clone(),\n"
                          "                        cache_guards=")
    source = replace_once(
        source,
        '                stage="complete",\n                passed=True,\n                full_csa_validated=True,',
        '''                stage="diagnostic_complete",
                passed=all(item["passed"] for item in report["launches"]),
                full_csa_validated=False,
                diagnostic_completed=True,
                numerical_gate_passed=all(item["passed"] for item in report["launches"]),
                adopted=False,'''
    )
    source = replace_once(
        source, "    args = parser.parse_args()\n",
        '''    args = parser.parse_args()
    if args.metadata_replay or args.padding_replay or args.static_reference:
        raise ValueError("本入口只做已授权Attention性能诊断，不扩展精度验收范围")
    if not args.source_matrix_reference or not args.l1_windows:
        raise ValueError("必须提供原单SPMD参考和配对L1窗口数")
'''
    )
    source = replace_once(
        source, '        comparison="当前原动态分task PTO；同输入/正式第4层权重/合成历史",',
        '        comparison="独立Attention性能诊断；数值FAIL保留，图重放与各自eager精确一致",'
    )

    matrix = (original / "output_policy_matrix.py").read_text()
    matrix = replace_once(matrix, "    verify,\n", "    verify,\n    collect_output,\n")
    matrix = replace_once(matrix, "    def checked():\n        numerical = verify()\n",
                          '''    eager_outputs = {}
    def checked(name, establish=False):
        if establish:
            eager_outputs[name] = collect_output()
        if name not in eager_outputs:
            raise AssertionError("缺少本版eager输出，禁止将图结果当自身参考")
        numerical = verify(eager_outputs[name])
''')
    matrix = replace_once(
        matrix, "    candidates = {name_for(existing.policy): existing}\n",
        '''    # existing已完成A→B→A；在原输入A上取得它自己的eager参考。
    reset()
    existing(config=compile_options)
    torch.npu.synchronize()
    report["checks"]["spmd_candidate_eager"] = checked("spmd_candidate", establish=True)
    reset()
    baseline_entry()
    torch.npu.synchronize()
    verify()  # 原多task必须仍与原基准完全相同。
    report["checks"]["dynamic_pto_eager"] = checked("dynamic_pto", establish=True)
    candidates = {name_for(existing.policy): existing}
'''
    )
    matrix = replace_once(matrix, 'report["checks"][name + "_eager"] = checked()',
                          'report["checks"][name + "_eager"] = checked(name, establish=True)')
    matrix = replace_once(matrix, 'report["checks"][name + "_graph"] = checked()',
                          'report["checks"][name + "_graph"] = checked(name)')
    matrix = replace_once(matrix, 'report["checks"][name + "_post_l1"] = checked()',
                          'report["checks"][name + "_post_l1"] = checked(name)')
    matrix = replace_once(
        matrix, '        report["passed"] = True\n',
        '''        # 此处通过仅指诊断运行/图/保护/协议，不改变外层严格数值FAIL。
        report["passed"] = True
        report["validation_scope"] = "每版对自身eager的图一致性、状态及协议；不代表跨版数值通过"
        report["numerical_adoption_proven"] = False
'''
    )
    # 原文件来自同一旧测试目录；完整保存本次副本便于直接查看与复现。
    for name, content in (("attention_perf_runner.py", source), ("attention_perf_matrix.py", matrix)):
        target = here / name
        if target.exists():
            raise RuntimeError("禁止覆盖正在使用或已记录的诊断版本：" + name)
        ast.parse(content)
        target.write_text(content)


if __name__ == "__main__":
    main()
