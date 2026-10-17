# SPDX-License-Identifier: Apache-2.0
"""只读现有CPU产物：动态ABI、真实T边界及尾mask/零填，绝不启动设备或再次编译。

这里检查本轮已保存产物的具体SSA符号，重编译若改变命名须人工复核后更新，
不能把字符串缺失直接解释为数学失败。完整设备语义及组合依赖仍未验证。
"""

import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/output_projection_a_stages.py"


def body(text, name):
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def main():
    selected = {"nd": "full_v1", "nz": "full_v4", "quant": "full_v1"}
    checks = []
    data = {}
    for stage, version in selected.items():
        result = json.loads((HERE / version / "result.json").read_text())
        row = next(r for r in result["stages"] if r["stage"] == stage)
        assert row["full_codegen_passed"]
        root = Path(row["artifact"])
        cpp = next((root / "kernels").rglob("*.cpp"))
        text = cpp.read_text()
        data[stage] = text
        helper = "o_projection_group_quant_stage" if stage == "quant" else "o_projection_a_stage_" + stage
        assert body(SOURCE.read_text(), helper) == body((HERE / version / SOURCE.name).read_text(), helper)
        assert "CSA_OA_PROBE_T_conv.u64 = args[3];" in text
        assert "CSA_OA_PROBE_T_conv.val" in text
        # shape维度与真实T均进入最终incore ABI；T通过运行时Scalar包传递，非编译常量。
        assert re.search(r"int64_t CSA_OA_MM_ROWS = .*shapes\[0\]", text)
        dims = re.findall(r"int64_t (CSA_OA_\w+) = .*?shapes\[\d+\]", text)
        checks.append({"stage": stage, "version": version, "cpp": str(cpp),
                       "dynamic_tensor_dims": dims, "actual_T": "CSA_OA_PROBE_T运行时Scalar→v4",
                       "math_source_matches_current": True, "TSTORE_sites": text.count("TSTORE("),
                       "TLOAD_sites": text.count("TLOAD(")})

    nd, nz, quant = data["nd"], data["nz"], data["quant"]
    assert "int64_t v54 = v53 < v20 ? v53 : v20;" in nd
    assert "v235, v54, v20);" in nd and "TSTORE(v240, v226);" in nd
    assert "int64_t v64 = v63 < v27 ? v63 : v27;" in nz
    assert "v249, v64, v22);" in nz and "TSTORE(v254, v240);" in nz
    assert "pto::Layout::NZ> v56" in nz and "pto::Layout::NZ>" not in nd
    # 两份O_A仅实际T的valid M写回，不能把P_MM全部当成已生产FP32。
    oa = {"nd": "v54=min(T-v52,128)同时限制packed TLOAD与output TSTORE形状。",
          "nz": "v64=min(T-v62,128)限制Mat输入与output GM视图；weight根仍Layout::NZ。",
          "packed_stride": "两份最终CPP均以独立PACK_TOTAL_ROWS/8计算组stride，不复用P_MM。"}
    assert "v5 / v17" in nd and "v5 / v20" in nz
    # 量化pipeline展开为一对8行加一个尾8行；检查三条算术链的有效rows。
    for rows, constants, outputs in [
        ("v73", ["v116", "v120"], ["v118", "v122", "v124", "v128", "v130", "v134", "v138"]),
        ("v75", ["v172", "v176"], ["v174", "v178", "v180", "v184", "v186", "v190", "v194"]),
        ("v220", ["v243", "v247"], ["v245", "v249", "v251", "v269", "v271", "v275", "v279"]),
    ]:
        for variable in constants:
            assert f"{variable}.SetValidShape(v16, {rows});" in quant, variable
        for variable in outputs:
            assert (f"{variable}.SetValidShape(v16, {rows});" in quant
                    or f"{variable}.SetValidShape({rows}, v23);" in quant), variable
    assert "TROWMAX(v239, v237, v235);" in quant
    assert "TDIV(v249, v247, v245);" in quant
    assert "TMULS(v251, v245, v30);" in quant
    assert "v251.SetValidShape(v16, v220);" in quant
    assert "TSTORE(v264, v251);" in quant and "TSTORE(v292, v279);" in quant
    zero = quant[quant.index("if (v4 <= (int64_t) ((uint64_t) v65 + (uint64_t) v19)) {"):]
    zero = zero[:zero.index('extern "C"')]
    assert "for (int64_t v293 = v4; v293 < v61; v293 += v17)" in zero
    assert "TLOAD(" not in zero and "TSTORE(v313, v296);" in zero
    assert "v308, v301, v23);" in zero
    result = {
        "status": "PASS", "selected_artifacts": checks, "oa_effective_rows": oa,
        "quant_effective_rows": {
            "paired_pipeline_rows": ["v73=min(T-qt,8)", "v75=min(T-qt-8,8)"],
            "tail_pipeline_rows": "v220=min(T-qt_tail,8)",
            "checked": ("三条链常数、amax/div、dequant_scale、乘法与三次cast保留有效shape；"
                        "尾行scale/INT8 TSTORE没有DCE。"),
            "zero_owner": ("v4(T)<=quant_start+32的唯一组内任务；[T,ceil(T/96)*96)每8行纯写INT8，"
                           "末段限valid_rows，不读FP32。"),
            "scale_capacity": "独立CSA_OA_SCALE_ROWS从scale.shape[1]取得，仅写T真实行。",
        },
        "device_execution": False,
        "limits": ["检查固定已保存后端产物与源级合同；不是设备数据、完整融合同步或性能证明。",
                   "CPU实际编译ND/N128、NZ/N256、quant/RB96；RB32/128等容量边界由源码几何模型检查，未额外重复编译。"],
    }
    (HERE / "compiled_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "selected": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
