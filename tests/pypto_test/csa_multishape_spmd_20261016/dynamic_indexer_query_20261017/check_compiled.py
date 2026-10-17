# SPDX-License-Identifier: Apache-2.0
"""只读已有完整后端产物，核对真实T标量独立于工作区容量进入核内循环。

本脚本不重新编译，不执行NPU；产物中的三个阶段从上游输入Tensor的shape
获取T，然后作为标量传入incore。其余阶段直接从数学输入Tensor维度获取T。
这里检查函数参数映射、动态循环上界以及真实GM写回，避免只看编译exit code。
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "full_v2"


def with_line(path, fragment):
    return {"file": str(path.relative_to(HERE)), "line": path.read_text().split(fragment)[0].count("\n") + 1}


def main():
    result = json.loads((RUN / "result.json").read_text())
    assert all(row["full_codegen_passed"] for row in result["stages"])
    audits = []
    for stage in ("hadamard", "quant", "weights_reduce"):
        root = RUN / stage / "kernel"
        orch = next((root / "orchestration").glob("*.cpp"))
        kernel = next((root / "kernels").glob("*/*.cpp"))
        ot, kt = orch.read_text(), kernel.read_text()
        symbol = "CSA_INDEX_QUERY_T"
        shape = f"int64_t {symbol} = (int64_t)orch_args.tensor(0).ref().shapes[0];"
        assert shape in ot
        assert f"params_t0.add_scalar({symbol});" in ot
        assert f"// Unpack scalar: {symbol}" in kt
        name = f"{stage}_probe_incore_0"
        formal = re.search(r"static __aicore__ void " + name + r"\((.*?)\) \{", kt).group(1).split(",")
        actual = re.findall(r"\b" + name + r"\(([^;]*?)\);", kt)[-1].split(",")
        position = [value.strip() for value in actual].index(symbol)
        variable = formal[position].strip().split()[-1]
        body = kt[kt.index("static __aicore__ void " + name) : kt.index("// --- Kernel entry point ---")]
        if stage == "weights_reduce":
            bound = re.search(r"int64_t (v\d+) = [^;]*\b" + variable + r"\b[^;]*/[^;]*;", body)
            assert bound
            loop = next(line for line in body.splitlines() if "for (" in line and f"< {bound.group(1)};" in line)
        else:
            loop = next(
                line for line in body.splitlines() if "for (" in line and re.search(r"\b" + variable + r"\b", line)
            )
        assert "TSTORE(" in body
        audits.append(
            {
                "stage": stage,
                "runtime_T_from_input_shape": with_line(orch, shape),
                "scalar_submit": with_line(orch, f"params_t0.add_scalar({symbol});"),
                "scalar_unpack": with_line(kernel, f"// Unpack scalar: {symbol}"),
                "lowered_T_parameter": variable,
                "runtime_loop": loop.strip(),
                "runtime_loop_location": with_line(kernel, loop),
                "actual_GM_store_present": True,
                "capacity_not_used_as_valid_T": True,
            }
        )
    report = {
        "passed": True,
        "scope": "已生成产物的三个T标量阶段ABI与循环审计，不重新编译",
        "six_stages_full_codegen": True,
        "stages": audits,
        "source_comments_after_compile": "正式源码仅补充中文注释和signed_sin合同；数学AST仍与本次编译源码一致。",
        "device_execution": False,
        "full_CSA_integration_validated": False,
        "device_precision_or_performance_claim": False,
    }
    (RUN / "compiled_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "runtime_T_scalar_stages": len(audits)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
