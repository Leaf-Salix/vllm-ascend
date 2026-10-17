"""只读v2既有AST与C++，核对真实槽位/搬运与事件；不重新编译或启动设备。"""

import ast
import json
import re
from pathlib import Path


here = Path(__file__).resolve().parent
kernel = here / "cpu_v2/kernel"
allocated = kernel / "passes_dump/38_after_AllocateMemoryAddr.py"
tree = ast.parse(allocated.read_text())
allocations, values, moves, ops = {}, {}, [], []
for node in sorted(ast.walk(tree), key=lambda x: getattr(x, "lineno", 0)):
    if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
        continue
    name = node.target.id
    for call in (x for x in ast.walk(node.annotation) if isinstance(x, ast.Call)):
        if ast.unparse(call.func) != "pl.MemRef" or not isinstance(call.args[0], ast.Name):
            continue
        mem_name = call.args[0].id
        if mem_name.startswith("mem_ddr"):
            continue
        offset = ast.literal_eval(call.args[1].args[0])
        size = ast.literal_eval(call.args[2])
        item = {"memory": mem_name, "space": mem_name.split("_")[1], "offset": offset,
                "bytes": size, "end": offset + size}
        allocations[mem_name] = item
        values[name] = item
    if isinstance(node.value, ast.Call):
        fn = ast.unparse(node.value.func)
        if fn == "pl.tile.move":
            moves.append(ast.unparse(node))
        if fn in ("pl.tile.load", "pl.tile.extract", "pl.tile.matmul_acc"):
            ops.append({"line": node.lineno, "result": name, "op": fn,
                        "args": [ast.unparse(a) for a in node.value.args]})
peaks = {}
for v in allocations.values():
    peaks[v["space"]] = max(peaks.get(v["space"], 0), v["end"])
assert peaks == {"acc": 131072, "mat": 393216, "left": 65536, "right": 65536}
assert not moves
assert sorted(v["offset"] for v in allocations.values() if v["space"] == "right") == [0, 32768]
source = ast.parse((here / "pair_four_k256.py").read_text())
loops = [n for n in ast.walk(source) if isinstance(n, ast.For)]
assert len(loops) == 1 and ast.unparse(loops[0].iter.func) == "pl.range"
assert ast.unparse(loops[0].target) == "(outer, (acc0, acc1))"
init_values = next(k.value for k in loops[0].iter.keywords if k.arg == "init_values")
assert ast.unparse(init_values) == "(valid0, valid1)"
calls = [n for n in ast.walk(source) if isinstance(n, ast.Call)]
assert not any(ast.unparse(n.func) == "pl.pipeline" for n in calls)
mma = sorted((n for n in ast.walk(loops[0]) if isinstance(n, ast.AnnAssign) and
              isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "pl.tile.matmul_acc"),
             key=lambda n: n.lineno)
# 实际源码使用普通Assign；核对16次显式MMA，只有首K128的两输出有初始化条件。
mma_source = sorted((n for n in calls if ast.unparse(n.func) == "pl.tile.matmul_acc"), key=lambda n: n.lineno)
assert len(mma_source) == 16
initializers = [ast.unparse(next(k.value for k in n.keywords if k.arg == "init_cond")) for n in mma_source]
assert initializers == ["outer == 0", "outer == 0"] + ["False"] * 14
cpp = kernel / "kernels/aic/oa_pair_four_k256.cpp"
text = cpp.read_text()
events = [{"line": n, "text": line.strip()} for n, line in enumerate(text.splitlines(), 1)
          if re.search(r"\b(TLOAD|TEXTRACT|TMATMUL(?:_ACC)?|set_flag|wait_flag|TSTORE)\(", line)]
(here / "event_sequence.txt").write_text("\n".join(f"{x['line']}: {x['text']}" for x in events) + "\n")
def line_of(fragment):
    return next(n for n, line in enumerate(text.splitlines(), 1) if fragment in line)
checks = {
    "block1_loads": [line_of("TLOAD(v106,"), line_of("TLOAD(v121,"), line_of("TLOAD(v133,")],
    "block0_first_extract": line_of("TEXTRACT(v148,"),
    "block2_loads": [line_of("TLOAD(v173,"), line_of("TLOAD(v188,"), line_of("TLOAD(v200,")],
    "block1_first_extract": line_of("TEXTRACT(v214,"),
    "block3_loads": [line_of("TLOAD(v238,"), line_of("TLOAD(v253,"), line_of("TLOAD(v265,")],
    "block2_first_extract": line_of("TEXTRACT(v279,"),
}
assert max(checks["block1_loads"]) < checks["block0_first_extract"]
assert max(checks["block2_loads"]) < checks["block1_first_extract"]
assert max(checks["block3_loads"]) < checks["block2_first_extract"]
report = {
    "scope": "v2单次CPU产物只读审计；不是设备精度/性能结论",
    "full_codegen_passed": json.loads((here / "cpu_v2/result.json").read_text())["full_codegen_passed"],
    "memory_peaks_bytes": peaks, "allocations": allocations, "value_allocations": values,
    "mat_moves": moves, "range_loop_carried": ["Acc0", "Acc1"],
    "explicit_mma_nodes_per_k1024": len(mma_source), "initializers": initializers,
    "designed_per_accumulator_k_offsets": list(range(0, 4096, 128)),
    "matmul_ops_from_allocated_ir": [x for x in ops if x["op"] == "pl.tile.matmul_acc"],
    "load_ops_from_allocated_ir": [x for x in ops if x["op"] == "pl.tile.load"],
    "cpp_issue_order_checks": checks,
    "event_window_interpretation": "块0只等本块MTE2事件0/1/2；块1的3/4/5尚不等待。后续块亦按本块就绪事件消费，代码没有将未来块load完成提前绑定到当前块第一次extract。真实重叠比例未测。",
    "cpp_counts": {name: len(re.findall(r"\b" + name + r"\(", text))
                   for name in ["TLOAD", "TEXTRACT", "TMATMUL", "TMATMUL_ACC", "TSTORE", "set_flag", "wait_flag"]},
    "device_execution": False,
}
(here / "generated_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"full_codegen_passed": report["full_codegen_passed"], "peaks": peaks,
                  "move_count": len(moves), "checks": checks}, ensure_ascii=False))
