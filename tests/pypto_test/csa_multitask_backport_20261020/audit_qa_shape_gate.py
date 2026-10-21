# SPDX-License-Identifier: Apache-2.0
"""只比较开启/关闭分档的已有编译产物；不编译、不运行NPU、不计算hash。"""

import argparse
import json
import re

from prepare import HERE


def normalize(text):
    # 只去注释/空白，并对编译器临时标识符做一一重命名。保留所有字面量、
    # 指令、类型、调用实参、任务编号和地址；不会合并两个不同变量。
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/|[A-Za-z_]\w*|\s+|.', text)
    names, result = {}, []
    for token in tokens:
        if token.isspace() or token.startswith(("//", "/*")):
            continue
        if re.fullmatch(r"[A-Za-z_]\w*", token) and (
            re.search(r"_inline\d+|__(?:phi|ssa|rv|tmp)_v\d+", token) or re.fullmatch(r"v\d+", token)
        ):
            token = names.setdefault(token, f"temporary_{len(names)}")
        result.append(token)
    return result


def compare(reference, candidate):
    files = ["kernel_config.py", "orchestration/_decode_csa_tp1_layer.cpp"]
    old = {str(p.relative_to(reference)) for p in (reference / "kernels").rglob("*.cpp")}
    new = {str(p.relative_to(candidate)) for p in (candidate / "kernels").rglob("*.cpp")}
    if not old or not new:
        raise ValueError("缺少实际核编译输入")
    files += sorted(old & new)
    rows = []
    for name in files:
        a, b = (reference / name).read_text(), (candidate / name).read_text()
        rows.append({"file": name, "exact_equal": a == b, "normalized_equal": normalize(a) == normalize(b)})
    return {
        "reference": str(reference.relative_to(HERE)),
        "candidate": str(candidate.relative_to(HERE)),
        "added_kernel_files": sorted(new - old),
        "removed_kernel_files": sorted(old - new),
        "files": rows,
        "passed": old == new and all(row["normalized_equal"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="qa_shape_gate_v1")
    args = parser.parse_args()
    if not args.candidate.isidentifier():
        raise ValueError("需要私有候选包名")
    root = HERE / "runs"
    result = {
        "scope": "生成代码等价检查，不代表编译二进制或设备执行性能相同",
        "normalization": (
            "仅注释/空白、inline数字标识符、phi/ssa/rv/tmp版本和vN临时名的一一重命名；"
            "不忽略算术、常数、类型、地址或实参"
        ),
        "off": compare(
            root / "ws_interval_v2/h131072_b16_interval_v1/build_output/_decode_csa_tp1_layer_r_cw0lhm",
            root / args.candidate / "cpu_shape_off_v1/kernel",
        ),
        "on": compare(
            root / "qa_single_m16_v2/cpu_preflight_v1/kernel",
            root / args.candidate / "cpu_shape_on_v1/kernel",
        ),
    }
    result["passed"] = result["off"]["passed"] and result["on"]["passed"]
    (HERE / "QA_SHAPE_GATE_CODEGEN.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {name: {k: v for k, v in result[name].items() if k != "files"} for name in ("off", "on")},
            ensure_ascii=False,
        )
    )
    for name in ("off", "on"):
        print(name, "differences", [r["file"] for r in result[name]["files"] if not r["normalized_equal"]])


if __name__ == "__main__":
    main()
