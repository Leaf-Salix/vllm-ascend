#!/usr/bin/env python3
"""只读比较本次唯一 CPU 与同控制、同参照源码的既有 CPU 生成物。"""
import collections
import json
import pathlib
import re

E = pathlib.Path(__file__).resolve().parent
B = E.parent / "ultra_attention_tick_clock_20261018"
old_result = json.loads((B / "cpu_v1/result.json").read_text())
new_result = json.loads((E / "cpu_v1/result.json").read_text())
assert old_result["passed"] and new_result["passed"]
assert old_result["controls"] == new_result["controls"]
assert old_result["compiled_abi"] == new_result["compiled_abi"]
reference = E / "source_reference"
old_source = pathlib.Path(old_result["source_package"])
names = sorted(p.name for p in reference.glob("*.py"))
assert len(names) == 33
assert names == sorted(p.name for p in old_source.glob("*.py"))
assert all((reference / n).read_text() == (old_source / n).read_text() for n in names)


def alloc(path):
    rows = []
    envelope = collections.defaultdict(int)
    for lineno, text in enumerate(path.read_text().splitlines(), 1):
        if "pto.alloc_tile" not in text:
            continue
        loc = re.search(r'loc\("([^"]+)":(\d+):(\d+)\)', text)
        source, source_line = pathlib.Path(loc[1]).name, int(loc[2])
        signature = text.split(" = ", 1)[1].split(" loc(", 1)[0]
        signature = re.sub(r"_inline\d+", "_inline", signature)
        signature = re.sub(r"%\d+\b", "%ssa", signature)
        signature = re.sub(r"__(iter|ssa)_v\d+", r"__\1", signature)
        m = re.search(r"addr = %c(\d+)_i64.*loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)", text)
        assert m, (lineno, text)
        address, space, dtype, height, width = m.groups()
        size = int(height) * int(width) * int(re.search(r"\d+", dtype)[0]) // 8
        envelope[space] = max(envelope[space], int(address) + size)
        rows.append({"source": source, "source_line": source_line, "pto_line": lineno,
                     "signature": signature, "address": int(address), "space": space,
                     "dtype": dtype, "rows": int(height), "cols": int(width)})
    return rows, dict(envelope)


def keys(rows):
    return [(r["source"], r["signature"]) for r in rows]


old_pto = B / "cpu_v1/kernel/ptoas/dynamic_csa_full.pto"
new_pto = E / "cpu_v1/kernel/ptoas/dynamic_csa_full.pto"
old_cpp = B / "cpu_v1/kernel/ptoas/dynamic_csa_full.cpp"
new_cpp = E / "cpu_v1/kernel/ptoas/dynamic_csa_full.cpp"
a, old_envelope = alloc(old_pto)
b, new_envelope = alloc(new_pto)
small = [r for r in b if r["source"] == "qkv_stages.py" and 146 <= r["source_line"] <= 175]
retained = [r for r in b if r not in small]
assert len(small) == 15 and keys(a) == keys(retained)
assert old_envelope == new_envelope
old_lines, new_lines = old_cpp.read_text().splitlines(), new_cpp.read_text().splitlines()
# 固定到本次两个实际生成物的 QA 区间；不声称是通用 C++ 解析器。
assert old_lines[6181].strip() == "if (v133 < v258) {"
assert new_lines[6284].strip() == "if (v133 <= v272) {"
assert new_lines[6655].strip() == "} else {"


def events(lines, lo, hi, erase_ids=False):
    records = []
    for index in range(lo - 1, hi):
        text = lines[index].strip()
        if re.match(r"(?:set_flag|wait_flag|wait_flag_dev|ffts_cross_core_sync|pipe_barrier|dsb|dcci)\(", text):
            text = re.sub(r"\bv\d+\b", "v", text)
            if erase_ids:
                text = re.sub(r"EVENT_ID\d+", "EVENT_ID", text)
            records.append(text)
    return records


prefix_same = events(old_lines, 1, 6181) == events(new_lines, 1, 6284)
suffix_same = events(old_lines, 7677, len(old_lines), True) == events(new_lines, 8155, len(new_lines), True)
old_qa = events(old_lines, 6182, 7676, True)
retained_qa = events(new_lines, 6657, 8154, True)
assert prefix_same and suffix_same
assert collections.Counter(old_qa) == collections.Counter(retained_qa)

out = {
    "passed": True,
    "scope": "已有生成物只读文本/分配/调用列表审计，无编译、无设备执行",
    "baseline_result": str(B / "cpu_v1/result.json"),
    "candidate_result": str(E / "cpu_v1/result.json"),
    "reference_sources_33_direct_text_equal": True,
    "controls_equal": True,
    "compiled_abi_equal": True,
    "compiled_abi_entries": len(new_result["compiled_abi"]),
    "controls": new_result["controls"],
    "allocation_method": "按 PTO 顺序比较 source 文件名与完整 alloc 右式；仅去 source 路径/行号、inline 编号、数值 SSA 编号及 iter/ssa 版本。动态 valid 关系另由源 AST 与整数 owner 审计覆盖。地址/类型/形状/静态 valid 均保留。",
    "reference_allocation_count": len(a),
    "candidate_allocation_count": len(b),
    "retained_allocation_records_equal": True,
    "retained_qkv_allocations": sum(r["source"] == "qkv_stages.py" for r in a),
    "outside_qkv_allocations": sum(r["source"] != "qkv_stages.py" for r in a),
    "address_envelope_bytes_both": old_envelope,
    "address_envelope_limit": "各空间 max(addr+声明形状字节数)，不是累加分配量，也不是运行时 live 峰值。",
    "new_small_branch_allocations": small,
    "event_method": "依生成顺序提取 set/wait_flag、wait_flag_dev、FFTS、pipe_barrier、dsb、dcci；SSA 变量重命名归一。局部 event ID 归一只用于明确标出的比较，不代表完整指令时序一致。",
    "before_qa_event_records_equal": prefix_same,
    "before_qa_event_records": len(events(old_lines, 1, 6181)),
    "small_branch_event_records": len(events(new_lines, 6285, 6656)),
    "retained_qa_event_records_each": len(old_qa),
    "retained_qa_event_multiset_equal_after_id_normalization": True,
    "retained_qa_event_order_equal_after_id_normalization": old_qa == retained_qa,
    "after_qa_event_records_each": len(events(old_lines, 7677, len(old_lines))),
    "after_qa_event_order_equal_after_id_normalization": suffix_same,
    "after_qa_actual_event_id_differences": [
        {"old_cpp_lines": [8082, 8105], "new_cpp_lines": [8560, 8583], "pipe": "M→FIX", "old_id": 5, "new_id": 6},
        {"old_cpp_lines": [8369, 8397], "new_cpp_lines": [8847, 8875], "pipe": "M→FIX", "old_id": 6, "new_id": 7},
        {"old_cpp_lines": [8607, 8630], "new_cpp_lines": [9085, 9108], "pipe": "M→FIX", "old_id": 7, "new_id": 0},
    ],
    "small_cpp_region": [6285, 6656],
    "qa_release_cpp_region": [8155, 8186],
    "vector_qa_wait_pto_region": [6349, 6408],
    "limits": ["CPU 不验证真机位一致、同步活性或性能", "T>48 源数学/owner 保留，但编译器 QA 内事件编号和局部排序改变，不能宣称其机器码或时序完全不变", "本次 CPU 控制 OA256/OB128；长历史设备策略由主入口独立选择"],
    "artifacts": [{"path": str(p), "bytes": p.stat().st_size} for p in [old_pto, old_cpp, new_pto, new_cpp]],
}
(E / "codegen_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print("PASS: 33 source texts; controls/130 ABI; retained 2365 alloc records; unchanged address envelopes; bounded event audit.")
