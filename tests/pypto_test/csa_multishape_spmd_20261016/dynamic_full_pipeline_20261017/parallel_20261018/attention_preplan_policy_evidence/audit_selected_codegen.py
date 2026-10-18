# SPDX-License-Identifier: Apache-2.0
"""核对分档只选择已有数学体：逐行比较真实生成C++，不做hash、不计性能。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREPLAN = HERE.parent / "attention_preplan_evidence"


def code(path):
    # 只删除整行//注释；不归一化变量名、数值、空白、顺序或可执行语句。
    return "\n".join(line for line in path.read_text().splitlines()
                     if not line.lstrip().startswith("//"))


def kernel(directory, side, role):
    result = json.loads((directory / "result.json").read_text())
    root = Path(result["source_matrix"]["kernel_configs"][side]).parent
    return next((root / "kernels" / role).glob("*.cpp"))


def main():
    rows = []
    pairs = [
        ("128K/B8后置", HERE/"h131072_b8_v1", "spmd_candidate", HERE/"h131072_b8_v1", "spmd_reference"),
        ("G6小M16区间，T30对已测T24", HERE/"h131072_b5_v1", "spmd_candidate", PREPLAN/"h131072_b4_v1", "spmd_candidate"),
        ("G6较大T区间，T66对已测T96", HERE/"h131072_b11_v1", "spmd_candidate", PREPLAN/"h131072_b16_v1", "spmd_candidate"),
        ("128K/B16前置", HERE/"h131072_b16_v1", "spmd_candidate", PREPLAN/"h131072_b16_v1", "spmd_candidate"),
        ("8K/B32前置", HERE/"h8192_b32_v1", "spmd_candidate", PREPLAN/"h8192_b32_v1", "spmd_candidate"),
        ("原B24/B28属于同一符号策略", PREPLAN/"h131072_b24_v1", "spmd_candidate", PREPLAN/"h131072_b28_v1", "spmd_candidate"),
    ]
    for name, left, left_side, right, right_side in pairs:
        for role in ("aic", "aiv"):
            a, b = kernel(left,left_side,role), kernel(right,right_side,role)
            same = code(a) == code(b)
            rows.append(dict(strategy=name, role=role, candidate=str(a), reference=str(b), executable_cpp_equal=same))
            assert same, (name,role)
    # CPU符号编译未注入等待退避，故与实测产物的同一hook前源码比较，
    # 不能错误地把WAIT helper的插入方式差异当成候选算法差异。
    for cpu, case in (("cpu_long_large_v1","h131072_b24_v1"),
                      ("cpu_short_g2_v1","h8192_b16_v1")):
        result = json.loads((HERE/cpu/"result.json").read_text())
        assert result["passed"] and result["single_mix"]
        a = Path(result["artifact"]) / "ptoas/dynamic_csa_full.cpp"
        b = PREPLAN/case/"codegen_probe/original.cpp"
        same = code(a) == code(b)
        rows.append(dict(strategy=cpu, role="combined_before_same_wait_hook", candidate=str(a),
                         reference=str(b), executable_cpp_equal=same))
        assert same, cpu
    report = dict(passed=True, normalization="仅忽略整行//注释，逐行直接比较可执行C++",
                  measured_performance=False, rows=rows,
                  limitation="代码相等不等于每次采样耗时相等；不把CPU或跨轮对比当性能证据")
    (HERE/"SELECTED_CODEGEN_EQUIVALENCE.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(passed=True, comparisons=len(rows))))


if __name__ == "__main__":
    main()
