#!/usr/bin/env python3
"""只读现成正式基线及唯一 CPU 生成物，检查 DQ Tile 地址/同步/数学循环。"""

import ast
import collections
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
PAR = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018")
BASE_SOURCE = PAR / "hc_active_notifications_evidence/source_candidate"
BASE = PAR / "hc_active_notifications_h131072_b16_v1/build_output/decode_csa_spmd_3kuwaqat"
NEW = OUT / "cpu_v1/kernel"


def normalize_cpp(text):
    text = re.sub(r"//[^\n]*", "", text)
    names = {}
    text = re.sub(r"\bv\d+\b|\blabel\d+\b",
                  lambda m: names.setdefault(m[0], "ssa_" + str(len(names))), text)
    return re.sub(r"\s+", "", text)


def cpp_function(text, name):
    begin = text.index("AICORE void " + name + "(")
    end = text.find("AICORE void ", begin + 20)
    return text[begin:end if end >= 0 else len(text)]


def calls(text):
    return dict(collections.Counter(re.findall(
        r"(?m)^\s*(T[A-Z]\w*|wait_flag_dev|ffts_cross_core_sync|wait_flag|set_flag|pipe_barrier|dsb|dcci)\(",
        text)))


def sync_calls(text):
    return dict(collections.Counter(re.findall(
        r"(?m)^\s*((?:wait_flag|set_flag|pipe_barrier)\([^;]+\));", text)))


def dq_loop(pto):
    start = pto.index("scf.for %dq_work_")
    body = pto.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (pto[end] == "{") - (pto[end] == "}")
        end += 1
    return pto[start:end]


def normalize_pto(text):
    text = re.sub(r' loc\("[^"]*":\d+:\d+\)', "", text)
    names = {}
    def replace(match):
        name = match[0]
        if re.fullmatch(r"%c\d+_(?:index|i\d+)", name):
            return name  # 保留整数常量取值、类型，不仅比较形状。
        return names.setdefault(name, "%ssa_" + str(len(names)))
    text = re.sub(r"%[\w.$]+", replace, text)
    return re.sub(r"\s+", "", text)


def collect(root, source, label):
    pto_path = root / "ptoas/dynamic_csa_full.pto"
    cpp_path = root / "ptoas/dynamic_csa_full.cpp"
    pto, cpp = pto_path.read_text(), cpp_path.read_text()
    tree = ast.parse((source / "query_projection_stages.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q_dequant_rope_stage")
    allocations, dq_allocations, maxima = [], [], {}
    role = None
    for lineno, line in enumerate(pto.splitlines(), 1):
        found = re.search(r"func.func @(\w+)", line)
        if found:
            role = found[1]
        if "pto.alloc_tile" not in line:
            continue
        found = re.search(r"(%\w+) = pto.alloc_tile addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
        assert found, line
        name, addr, space, dtype, rows, cols = found.groups()
        size = int(rows) * int(cols) * int(re.search(r"\d+", dtype)[0]) // 8
        row = dict(name=name, role=role, space=space, dtype=dtype, rows=int(rows), cols=int(cols),
                   address=int(addr), bytes=size, end=int(addr)+size, pto_line=lineno)
        allocations.append(row)
        key = role + "/" + space
        maxima[key] = max(maxima.get(key, 0), row["end"])
        source_line = re.search(r'query_projection_stages.py":(\d+):', line)
        if role == "vector_schedule" and source_line and fn.lineno <= int(source_line[1]) <= fn.end_lineno:
            dq_allocations.append(row)
    cube, vector = cpp_function(cpp, "cube_schedule"), cpp_function(cpp, "vector_schedule")
    loop = dq_loop(pto)
    (OUT / (label + "_dq_loop.pto.txt")).write_text(loop + "\n")
    (OUT / (label + "_dq_allocations.json")).write_text(json.dumps(dq_allocations, ensure_ascii=False, indent=2) + "\n")
    return dict(pto=str(pto_path), cpp=str(cpp_path), upper_bounds=maxima,
                dq_allocations=dq_allocations, dq_vec_upper=max(x["end"] for x in dq_allocations if x["space"] == "vec"),
                full_calls=dict(cube=calls(cube), vector=calls(vector)),
                full_sync_calls=dict(cube=sync_calls(cube), vector=sync_calls(vector))), cube, loop


def main():
    reference = OUT / "source_reference"
    names = sorted(p.name for p in reference.glob("*.py"))
    assert len(names) == 33
    same = all((BASE_SOURCE / n).read_text() == (reference / n).read_text() for n in names)
    assert same
    cpu = json.loads((OUT / "cpu_v1/result.json").read_text())
    assert cpu["passed"] and cpu["single_mix"] and cpu["sync_start"] and not cpu["device_execution"]
    old, old_cube, old_loop = collect(BASE, BASE_SOURCE, "baseline")
    new, new_cube, new_loop = collect(NEW, OUT / "source_candidate", "candidate")
    allocation_key = lambda x: tuple(x[k] for k in ("space", "dtype", "rows", "cols", "address", "bytes"))
    allocation_equal = list(map(allocation_key, old["dq_allocations"])) == list(map(allocation_key, new["dq_allocations"]))
    loop_equal = normalize_pto(old_loop) == normalize_pto(new_loop)
    cube_equal = normalize_cpp(old_cube) == normalize_cpp(new_cube)
    checks = dict(baseline_source_33_direct_text_equal=same, dq_allocations_equal=allocation_equal,
                  whole_upper_bounds_equal=old["upper_bounds"] == new["upper_bounds"],
                  aic_cpp_normalized_equal=cube_equal, dq_loop_normalized_equal=loop_equal,
                  hardware_call_counts_equal=old["full_calls"] == new["full_calls"],
                  local_sync_event_counts_equal=old["full_sync_calls"] == new["full_sync_calls"])
    result = dict(passed=all(checks.values()), checks=checks, baseline=old, candidate=new,
                  scope="同一正式33源基线、同controls的已生成PTO/C++；不重编基线，不执行设备",
                  comparison_normalization="忽略源码路径/行号和SSA/label名称；DQ整数常量保留值和类型",
                  limits="CPU不证明设备exact或性能；组数改变仍可能影响到达时刻/缓存行为")
    (OUT / "generated_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (OUT / "cpu_result.json").write_text(json.dumps(cpu, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], checks=checks,
                         baseline_dq_ub_end=old["dq_vec_upper"], candidate_dq_ub_end=new["dq_vec_upper"],
                         baseline_upper_bounds=old["upper_bounds"], candidate_upper_bounds=new["upper_bounds"]),
                     ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

