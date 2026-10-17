#!/usr/bin/env python3
"""只读两份已有完整 lowering，核对 Tile、数学、同步与动态余数路径。"""

import ast
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "query_dequant_group_20261018"


def normalize_cpp(text):
    text = re.sub(r"//[^\n]*", "", text)
    names = {}
    text = re.sub(r"\bv\d+\b|\blabel\d+\b", lambda m: names.setdefault(m[0], "ssa_" + str(len(names))), text)
    return re.sub(r"\s+", "", text)


def normalize_pto(text):
    text = re.sub(r' loc\("[^"]*":\d+:\d+\)', "", text)
    names = {}
    def replace(match):
        name = match[0]
        if re.fullmatch(r"%c\d+_(?:index|i\d+)", name):
            return name
        return names.setdefault(name, "%ssa_" + str(len(names)))
    return re.sub(r"\s+", "", re.sub(r"%[\w.$]+", replace, text))


def cpp_function(text, name):
    start = text.index("AICORE void " + name + "(")
    end = text.find("AICORE void ", start + 20)
    return text[start:end if end >= 0 else len(text)]


def block(text, start):
    brace = text.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if text[end:].startswith(" else {"):
        _, end = block(text, end + 1)
    return text[start:end], end


def region(text, pattern):
    match = re.search(pattern, text)
    assert match, pattern
    return block(text, match.start())[0]


def event_records(text):
    return re.findall(r"(?m)^\s*((?:wait_flag|set_flag|wait_flag_dev|ffts_cross_core_sync|pipe_barrier|dsb|dcci|pto::comm::TWAIT|pto::comm::TNOTIFY)\([^;]+\));", text)


def collect(directory, side, label):
    kernel = directory / "cpu_v1/kernel/ptoas"
    pto_path, cpp_path = kernel / "dynamic_csa_full.pto", kernel / "dynamic_csa_full.cpp"
    pto, cpp = pto_path.read_text(), cpp_path.read_text()
    fn = next(n for n in ast.parse((directory / side / "query_projection_stages.py").read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "q_dequant_rope_stage")
    rows, dq_rows, bounds = [], [], {}
    role = None
    for line_number, line in enumerate(pto.splitlines(), 1):
        name = re.search(r"func.func @(\w+)", line)
        if name:
            role = name[1]
        if "pto.alloc_tile" not in line:
            continue
        match = re.search(r"(%\w+) = pto.alloc_tile addr = %c(\d+)_i64.*?loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", line)
        assert match, line
        name, address, space, dtype, m, n = match.groups()
        size = int(m) * int(n) * int(re.search(r"\d+", dtype)[0]) // 8
        item = dict(name=name, role=role, space=space, dtype=dtype, rows=int(m), cols=int(n),
                    address=int(address), bytes=size, end=int(address)+size, pto_line=line_number)
        rows.append(item)
        key = role + "/" + space
        bounds[key] = max(bounds.get(key, 0), item["end"])
        source_line = re.search(r'query_projection_stages.py":(\d+):', line)
        if role == "vector_schedule" and source_line and fn.lineno <= int(source_line[1]) <= fn.end_lineno:
            dq_rows.append(item)
    cube, vector = cpp_function(cpp, "cube_schedule"), cpp_function(cpp, "vector_schedule")
    loop = region(pto, r"scf.for %dq_work_")
    first_alloc = loop.index("%qr_scale_dq_t_")
    math_start = loop.rfind("scf.if ", 0, first_alloc)
    math = block(loop, math_start)[0]
    (ROOT / (label + "_dq_loop.pto.txt")).write_text(loop + "\n")
    (ROOT / (label + "_dq_math.pto.txt")).write_text(math + "\n")
    (ROOT / (label + "_dq_allocations.json")).write_text(json.dumps(dq_rows, ensure_ascii=False, indent=2) + "\n")
    calls = lambda s: dict(collections.Counter(re.findall(r"(?m)^\s*(T[A-Z]\w*|wait_flag_dev|ffts_cross_core_sync|wait_flag|set_flag|pipe_barrier|dsb|dcci)\(", s)))
    return dict(pto=str(pto_path), cpp=str(cpp_path), allocations=rows, dq_allocations=dq_rows,
                address_envelopes=bounds, dq_vec_end=max(x["end"] for x in dq_rows if x["space"] == "vec"),
                hardware_calls=dict(cube=calls(cube), vector=calls(vector)),
                event_counts=dict(cube=dict(collections.Counter(event_records(cube))),
                                  vector=dict(collections.Counter(event_records(vector))))), cube, vector, math, loop


def main():
    old_cpu = json.loads((BASE / "cpu_v1/result.json").read_text())
    cpu = json.loads((ROOT / "cpu_v1/result.json").read_text())
    assert cpu["passed"] and cpu["single_mix"] and cpu["sync_start"] and not cpu["device_execution"]
    names = sorted(p.name for p in (ROOT / "source_reference").glob("*.py"))
    same = len(names) == 33 and all((BASE / "source_candidate" / n).read_text() == (ROOT / "source_reference" / n).read_text() for n in names)
    old, old_cube, old_vector, old_math, old_loop = collect(BASE, "source_candidate", "baseline")
    new, new_cube, new_vector, new_math, new_loop = collect(ROOT, "source_candidate", "candidate")
    allocation = lambda x: tuple(x[k] for k in ("role", "space", "dtype", "rows", "cols", "address", "bytes"))
    checks = dict(baseline_source_33_direct_text_equal=same,
                  controls_equal=old_cpu["controls"] == cpu["controls"],
                  compiled_abi_equal=old_cpu["compiled_abi"] == cpu["compiled_abi"],
                  dq_allocations_equal=list(map(allocation, old["dq_allocations"])) == list(map(allocation, new["dq_allocations"])),
                  all_allocations_equal=list(map(allocation, old["allocations"])) == list(map(allocation, new["allocations"])),
                  address_envelopes_equal=old["address_envelopes"] == new["address_envelopes"],
                  aic_cpp_normalized_equal=normalize_cpp(old_cube) == normalize_cpp(new_cube),
                  full_and_tail_math_pto_equal=normalize_pto(old_math) == normalize_pto(new_math),
                  hardware_call_counts_equal=old["hardware_calls"] == new["hardware_calls"],
                  local_and_cross_core_event_order_normalized_equal=all(
                      normalize_cpp("\n".join(event_records(a))) == normalize_cpp("\n".join(event_records(b)))
                      for a,b in ((old_cube,new_cube),(old_vector,new_vector))))
    # pipeline长度保留为真实设备Scalar；主循环两头一组，余数由if保护。
    compact = re.sub(r' loc\("[^"]*":\d+:\d+\)', "", new_loop)
    pipeline_lines = [line for line in compact.splitlines() if
                      ("dq_heads_in_row" in line or "h_inner_" in line or
                       ("arith.divsi" in line and "%c2_index" in line))]
    (ROOT / "candidate_pipeline_control.txt").write_text("\n".join(pipeline_lines) + "\n")
    cpp_lines = (ROOT / "cpu_v1/kernel/ptoas/dynamic_csa_full.cpp").read_text().splitlines()
    # 只保存本次已读的原始C++片段，带原行号；复算不调用任何编译工具。
    spans = [(19408,19428),(48257,48276),(48765,48806)]
    (ROOT / "pipeline_one_head_cpp.txt").write_text("\n".join(
        f"{i}: {cpp_lines[i-1]}" for start,end in spans for i in range(start,end+1)) + "\n")
    result = dict(passed=all(checks.values()), checks=checks,
                  baseline={k:v for k,v in old.items() if k != "allocations"},
                  candidate={k:v for k,v in new.items() if k != "allocations"},
                  baseline_allocations=len(old["allocations"]), candidate_allocations=len(new["allocations"]),
                  raw_event_parameter_text_equal=old["event_counts"] == new["event_counts"],
                  raw_event_parameter_difference="DQ之后跨核事件参数的SSA编号顺延；归一SSA后的全部GM/local/FFTS事件顺序另作严格比较，保留event ID与pipe",
                  one_head_pipeline=dict(
                      pto_main_loop="bound=2*floor(heads_in_row/2)，step=2；余数==1时执行最后一个head",
                      cpp_main_bound_line=48263, cpp_zero_trip_branch_lines=[48269,48275],
                      cpp_events_initialized_before_main=[48265,48266],
                      cpp_remainder_condition_line=48779, cpp_single_head_load_line=48805,
                      reviewed="n=1时主循环为0，EVENT6/7预置并排空；仅执行余数头hg+0，没有hg+1访问",
                      limit="源/PTO/C++控制审查，未做设备长度1 probe"),
                  scope="复用正式9d5327e5已生成CPU，候选仅一次完整CPU；未编译/执行设备",
                  normalization="仅忽略源码路径行号、SSA/label名称；保留常量值、Tile尺寸、地址及调用参数",
                  limits="地址末端是max(addr+声明字节)，不等于运行时live峰值；CPU不能证明设备同步活性或性能")
    (ROOT / "generated_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "cpu_result.json").write_text(json.dumps(cpu, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], checks=checks,
                         dq_vec_ends=[old["dq_vec_end"],new["dq_vec_end"]],
                         address_envelopes=new["address_envelopes"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
