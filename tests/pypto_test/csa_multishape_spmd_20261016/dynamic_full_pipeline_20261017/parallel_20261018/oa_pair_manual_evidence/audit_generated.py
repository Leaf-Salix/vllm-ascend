"""只读完整CPU既有产物：资源、动态写回和两端owner；不重新编译或上卡。"""

import ast
import json
import re
from pathlib import Path


here = Path(__file__).resolve().parent
kernel = here / "cpu_v1/kernel"
ir_path = kernel / "passes_dump/38_after_AllocateMemoryAddr.py"
ir_text = ir_path.read_text()
ir = ast.parse(ir_text)
cube = next(n for n in ast.walk(ir) if isinstance(n, ast.FunctionDef) and n.name == "cube_schedule")
vector = next(n for n in ast.walk(ir) if isinstance(n, ast.FunctionDef) and n.name == "vector_schedule")
pair = next(n for n in ast.walk(cube) if isinstance(n, ast.If) and "pair_first_inline" in ast.unparse(n.test))


def memory_values(node):
    values = {}
    for n in ast.walk(node):
        if not isinstance(n, ast.AnnAssign) or not isinstance(n.target, ast.Name):
            continue
        for c in ast.walk(n.annotation):
            if not isinstance(c, ast.Call) or ast.unparse(c.func) != "pl.MemRef" or not isinstance(c.args[0], ast.Name):
                continue
            mem = c.args[0].id
            if mem.startswith("mem_ddr"):
                continue
            start = ast.literal_eval(c.args[1].args[0])
            size = ast.literal_eval(c.args[2])
            values[n.target.id] = {"space": mem.split("_")[1], "start": start, "bytes": size,
                                   "end": start+size, "line": n.lineno}
    return values


def memory_summary(values):
    answer = {}
    for value in values.values():
        item = answer.setdefault(value["space"], {"address_upper_bound": 0, "intervals": set()})
        item["address_upper_bound"] = max(item["address_upper_bound"], value["end"])
        item["intervals"].add((value["start"], value["end"]))
    for item in answer.values():
        item["intervals"] = sorted(item["intervals"])
    return answer


pair_values = memory_values(pair)
pair_summary = memory_summary(pair_values)
whole = {"cube": memory_summary(memory_values(cube)), "vector": memory_summary(memory_values(vector))}
assert pair_summary["mat"]["intervals"] == [(i*65536, (i+1)*65536) for i in range(1, 7)]
assert pair_summary["right"]["intervals"] == [(0, 32768), (32768, 65536)]
assert pair_summary["left"]["intervals"] == [(0, 32768), (32768, 65536)]
assert pair_summary["acc"]["intervals"] == [(0, 65536), (65536, 131072)]
assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.tile.move" for n in ast.walk(pair))
pair_text = "\n".join(ir_text.splitlines()[pair.lineno-1:pair.end_lineno])+"\n"
(here / "pair_allocated_ir.py.txt").write_text(pair_text)
assert "valid_rows_inline2201__ssa_v0: pl.Scalar[pl.INDEX] = tokens__ssa_v0" in pair_text
stores = [ast.unparse(n) for n in ast.walk(pair) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.tile.store"]
assert len(stores) == 2
predicate = "0 < tokens__ssa_v0 and tokens__ssa_v0 <= 96"
assert any(isinstance(n, ast.If) and ast.unparse(n.test) == predicate for n in ast.walk(cube))
wait_branch = next(n for n in ast.walk(vector) if isinstance(n, ast.If) and ast.unparse(n.test) == predicate)
wait_text = "\n".join(ir_text.splitlines()[wait_branch.lineno-1:wait_branch.body[-1].end_lineno])+"\n"
(here / "wait_allocated_ir.py.txt").write_text(wait_text)
wait_loops = [n for n in wait_branch.body if isinstance(n, ast.For)]
assert len(wait_loops) == 2 and [ast.unparse(n.iter.args[0]) for n in wait_loops] == ["3", "2"]
tail = [ast.unparse(n) for n in wait_branch.body[-3:]]
assert tail == ["pl.system.fence()", "pl.system.cacheinvalid()", "pl.system.fence()"]

cpp_path = kernel / "kernels/aic/cube_schedule.cpp"
cpp = cpp_path.read_text()
lines = cpp.splitlines()
names, pending, tracked = {}, None, []
for lineno, line in enumerate(lines, 1):
    comment = re.search(r"// pto: %(\w+)", line)
    if comment:
        pending = comment.group(1)
    value = re.search(r"\b(v\d+) = Tile<", line)
    if value and pending:
        names[value.group(1)] = pending
    op = re.search(r"\b(TLOAD|TEXTRACT)\((v\d+),", line)
    if op:
        source_name = names.get(op.group(2), "")
        if re.match(r"(a[0-3]|w[0-3][ab]|l[0-3][01]|r[0-3][01][ab])_inline", source_name):
            tracked.append({"line": lineno, "op": op.group(1), "source_name": source_name,
                            "cpp": line.strip()})
loads = {n["source_name"].split("_inline")[0]: n["line"] for n in tracked if n["op"] == "TLOAD"}
extracts = {n["source_name"].split("_inline")[0]: n["line"] for n in tracked if n["op"] == "TEXTRACT"}
assert len(loads) == 12
window_checks = []
for b in range(1, 4):
    ls = [loads[f"a{b}"], loads[f"w{b}a"], loads[f"w{b}b"]]
    current = extracts[f"l{b-1}0"]
    assert max(ls) < current
    window_checks.append({"prefetch_block": b, "loads": ls, "current_first_extract": current})
begin = next(n for n, line in enumerate(lines, 1) if "// pto: %seed0_inline2190" in line)
end = next(n for n, line in enumerate(lines, 1) if n>begin and "if (v4329)" in line)
(here / "pair_generated_cpp.cpp.txt").write_text("\n".join(lines[begin-1:end-1])+"\n")
events = [{"line": n, "text": line.strip()} for n, line in enumerate(lines, 1)
          if begin <= n < end and re.search(r"\b(TLOAD|TEXTRACT|TMATMUL(?:_ACC)?|TSTORE|set_flag|wait_flag)\(", line)]
(here / "pair_event_sequence.txt").write_text("\n".join(f"{x['line']}: {x['text']}" for x in events)+"\n")
assert "v4336.SetValidShape(v133, v258);" in cpp and "v4338.SetValidShape(v133, v258);" in cpp
assert "int64_t CSA_FULL_T = static_cast<int64_t>(x_fp32__ssa_v0_tensor->shapes[0]);" in cpp
result = json.loads((here / "cpu_v1/result.json").read_text())
assert result["passed"] and len(result["compiled_abi"]) == 130 and len(result["symbolic_dimensions"]) == 58
report = {
    "cpu_scope": result["controls"], "full_codegen_passed": True,
    "runtime_roots": 130, "dynamic_axes": 58, "single_mix": result["single_mix"],
    "sync_start": result["sync_start"], "sample_shapes_supplied": result["sample_shapes_supplied"],
    "whole_function_resources": whole, "pair_resources": pair_summary,
    "pair_mat_occupied_bytes": 393216, "pair_mat_address_upper_bound": 458752,
    "pair_value_allocations": pair_values, "pair_has_mat_copy": False,
    "producer_consumer_dynamic_predicate": predicate,
    "new_wait_loops": ["3*g+p, p=0..2", "(3*g+3+s)%24, s=0..1"],
    "wait_tail": tail, "stores": stores, "cpp_stage_operations": tracked,
    "prefetch_issue_window_checks": window_checks,
    "dynamic_store_cpp": "v133 is runtime CSA_FULL_T; both Acc SetValidShape(v133,128), both GM store views have runtime v133 rows",
    "event_review": "当前块只等自己的MTE2就绪事件；未来块TLOAD先发。Right复用保留M→MTE1，Acc存储前M→FIX。只能证明生成依赖，无设备重叠比例。",
    "limitations": ["未NPU，未设备精度/图/保护/代际测试", "完整CPU仅G6/QAfalse；其他controls会重新生成",
                    "既有pipeline降深提示保留，未以同控制新编baseline排除全部allocator差异",
                    "两个Mat三元组共384KiB但全函数地址上界不是384KiB"],
}
(here / "generated_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"pair_mat_bytes": 393216, "full_cube_mat_upper": whole["cube"]["mat"]["address_upper_bound"],
                  "roots": 130, "axes": 58, "full_cpu_pass": True}, ensure_ascii=False))
