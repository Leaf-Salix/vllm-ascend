# SPDX-License-Identifier: Apache-2.0
"""只读现有full_v1后端产物，确认动态计数分支/循环和真实sort/merge/store未消失。"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "full_v1"


def main():
    result = json.loads((RUN / "result.json").read_text())
    assert len(result["stages"]) == 4 and all(row["full_codegen_passed"] for row in result["stages"])
    expected = {"merge2": (0, 1, 1), "roots": (0, 3, 1), "half_leaf": (6, 20, 6), "segment": (3, 8, 1)}
    rows = []
    for stage in expected:
        folder = RUN / stage / "kernel"
        kernel = next((folder / "kernels/aiv").glob("*.cpp"))
        orch = next((folder / "orchestration").glob("*.cpp"))
        text = kernel.read_text()
        actual = (text.count("TSORT32("), len(re.findall(r"\bTMRGSORT[<(]", text)), text.count("TSTORE("))
        # merge计数来自原六种/三种长度分支，完整保留全部路径。
        assert actual == expected[stage], (stage, actual, expected[stage])
        assert orch.read_text().count("rt_submit_aiv_task(") == 1
        assert "control__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);" in text
        assert re.search(r"int32_t v\d+ = \(v1\)\[", text)
        if stage == "roots":
            assert "int64_t v48 = (int64_t) v47;" in text
            assert "for (int64_t v58 = v16; v58 < v48; v58 += v19)" in text
        if stage == "half_leaf":
            assert "int64_t v58 = (int64_t) v57;" in text
            assert text.count("if (v58 <=") == 6
        if stage == "segment":
            assert text.count("if (v54 <=") == 2
        rows.append(
            {
                "stage": stage,
                "kernel": str(kernel.relative_to(HERE)),
                "dynamic_dimensions": re.findall(r"// Extract dynamic dim: (\w+)", text),
                "sort32_sites": actual[0],
                "mrgsort_sites": actual[1],
                "actual_GM_store_sites": actual[2],
                "control_values_read_on_device": True,
            }
        )
    report = {
        "passed": True,
        "scope": "四个独立CPU完整产物审计，不重新编译",
        "stages": rows,
        "warnings": "测试壳单worker循环变量未使用，编译器有UnusedVariableCheck提示；不影响实际sort/store。",
        "limits": "操作存在与动态ABI不能证明设备同分排序、cache同步或实际TopK数值。",
        "audit_correction": (
            "初次人工预期漏算half_leaf两处mrgsort，按六分支原AST重数应为20；"
            "仅修审计常数，无源码修改/重编译。"
        ),
        "device_execution": False,
        "performance_claim": False,
    }
    (RUN / "compiled_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "stages": len(rows)}))


if __name__ == "__main__":
    main()
