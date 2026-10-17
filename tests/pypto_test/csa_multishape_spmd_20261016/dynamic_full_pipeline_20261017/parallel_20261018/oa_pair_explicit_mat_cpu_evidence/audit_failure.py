"""只读取本目录唯一CPU尝试，提取资源与Mat回边；不导入PyPTO或创建设备。"""

import ast
import json
from pathlib import Path


here = Path(__file__).resolve().parent
allocated = here / "cpu_v1/kernel/passes_dump/38_after_AllocateMemoryAddr.py"
tree = ast.parse(allocated.read_text())
allocations = {}
values = {}
copies = []
for node in sorted(ast.walk(tree), key=lambda x: getattr(x, "lineno", 0)):
    if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
        continue
    name = node.target.id
    annotation = node.annotation
    calls = [x for x in ast.walk(annotation) if isinstance(x, ast.Call)]
    for call in calls:
        if ast.unparse(call.func) != "pl.MemRef" or not isinstance(call.args[0], ast.Name):
            continue
        mem_name = call.args[0].id
        if mem_name.startswith("mem_ddr"):
            continue
        offset = ast.literal_eval(call.args[1].args[0])
        size = ast.literal_eval(call.args[2])
        space = mem_name.split("_")[1]
        item = {"memory": mem_name, "space": space, "offset": offset, "bytes": size, "end": offset + size}
        allocations[mem_name] = item
        values[name] = item
    if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "pl.tile.move":
        src = ast.unparse(node.value.args[0])
        copies.append({"source": src, "destination": name, "source_allocation": values[src],
                       "destination_allocation": values[name], "line": node.lineno})

peaks = {}
for value in allocations.values():
    peaks[value["space"]] = max(peaks.get(value["space"], 0), value["end"])
assert peaks == {"acc": 131072, "mat": 393216, "left": 65536, "right": 65536}, peaks
assert len(copies) == 3 and all(x["source_allocation"]["space"] == "mat" and
                                x["destination_allocation"]["space"] == "mat" for x in copies)
first_move = next(f for f in sorted(allocated.parent.glob("*.py")) if "pl.tile.move(" in f.read_text())
source = ast.parse((here / "pair_explicit.py").read_text())
pipeline_calls = [n for n in ast.walk(source) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pl.pipeline"]
assert not pipeline_calls
result = json.loads((here / "cpu_v1/result.json").read_text())
assert result["source_imported"] and not result["full_codegen_passed"]
assert "supported tmov address-space pair" in result["error"]
pto = here / "cpu_v1/kernel/ptoas/oa_pair_explicit_mat.pto"
tmov_lines = [{"line": n, "text": text} for n, text in enumerate(pto.read_text().splitlines(), 1) if "pto.tmov" in text]
cpp = list((here / "cpu_v1/kernel/kernels").rglob("*.cpp"))
report = {
    "scope": "仅审计一次失败CPU生成，不是设备数值或流水证明",
    "full_codegen_passed": False, "device_execution": False, "source_pipeline_calls": 0,
    "actual_memory_peaks_bytes": peaks, "allocations": allocations,
    "actual_right_slot_offsets": sorted(v["offset"] for v in allocations.values() if v["space"] == "right"),
    "mat_backedge_copies": copies, "first_inserted_pass": str(first_move),
    "mat_copy_bytes_per_iteration": sum(x["source_allocation"]["bytes"] for x in copies),
    "mat_copy_iterations": 15, "pto_tmov": tmov_lines, "kernel_cpp_files": [str(f) for f in cpp],
    "actual_event_sequence_auditable": bool(cpp),
    "designed_per_accumulator_k_offsets": list(range(0, 4096, 128)),
    "boundary": "K顺序是源码设计核对；后端在非法Mat→Mat处失败，不能称完整数学或硬件事件通过",
}
(here / "failure_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"peaks": peaks, "mat_copies": len(copies), "new_compilation": False}, ensure_ascii=False))
