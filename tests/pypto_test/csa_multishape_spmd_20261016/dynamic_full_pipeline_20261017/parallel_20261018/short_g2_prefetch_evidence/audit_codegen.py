#!/usr/bin/env python3
"""只读唯一一次 CPU 的既有产物，摘录实际地址和六个 panel 的活跃操作数。

不调用 PyPTO、不生成内核、不执行设备；此审计不替代 exact 或性能验证。
"""

import json
import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    cpu = root / "cpu_medium_v1"
    result = json.loads((cpu / "result.json").read_text())
    expected = json.loads((root / "controls_medium.json").read_text())
    assert result["passed"] and result["controls"] == expected
    assert result["source_package"] == str(root / "source_candidate")
    assert not result["sample_shapes_supplied"] and not result["device_execution"]
    changed = []
    for source in sorted((root / "source_candidate").glob("*.py")):
        assert source.read_text() == (cpu / "source" / source.name).read_text()
        if source.read_text() != (root / "source_reference_d253" / source.name).read_text():
            changed.append(source.name)
    assert changed == ["policy.py"]

    pto = cpu / "kernel/ptoas/dynamic_csa_full.pto"
    cpp = cpu / "kernel/ptoas/dynamic_csa_full.cpp"
    lines = pto.read_text().splitlines()
    allocation_pattern = re.compile(
        r"(%\S+) = pto.alloc_tile addr = %c(\d+)_i64.*"
        r"!pto.tile_buf<loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)"
    )
    widths = {"i8": 1, "i16": 2, "f16": 2, "bf16": 2, "i32": 4, "ui32": 4, "f32": 4}
    allocations = {}
    all_allocations = []
    for lineno, line in enumerate(lines, 1):
        match = allocation_pattern.search(line)
        if match:
            name, address, space, dtype, rows, cols = match.groups()
            size = int(rows) * int(cols) * widths[dtype]
            allocations[name] = dict(
                line=lineno, name=name, address=int(address), end=int(address) + size,
                space=space, dtype=dtype, rows=int(rows), cols=int(cols),
            )
            # 不同函数可复用数字 SSA 名；总空间末端不能用跨函数 name 字典统计。
            all_allocations.append(allocations[name])

    qk, ws, key_extracts = [], [], []
    matmul_pattern = re.compile(r"pto.tmatmul ins\((%[^,]+), (%[^ :]+) :.*outs\((%[^ :]+) :")
    for lineno, line in enumerate(lines, 1):
        match = matmul_pattern.search(line)
        if match:
            left, right, output = match.groups()
            record = dict(line=lineno, left=allocations[left], right=allocations[right], output=allocations[output])
            if output.startswith("%buf_scores_inline"):
                qk.append(record)
            elif output.startswith(("%buf_reduced_pair_inline", "%buf_last_scores_inline")):
                ws.append(record)
        if "pto.textract" in line:
            match = re.search(r"outs\((%buf_(?:current|prefetched)_key[^ :]+) :", line)
            if match:
                key_extracts.append(dict(line=lineno, tile=allocations[match.group(1)]))
    assert len(qk) == len(ws) == len(key_extracts) == 6
    panels = []
    for panel, operation in enumerate(qk):
        current = operation["right"]
        assert current["address"] == key_extracts[panel]["tile"]["address"]
        next_key = key_extracts[panel + 1]["tile"] if panel < 5 else None
        if next_key:
            assert current["end"] <= next_key["address"] or next_key["end"] <= current["address"]
        previous_ws = ws[panel - 1] if panel else None
        if previous_ws:
            for key in (current, next_key):
                if key:
                    assert previous_ws["right"]["end"] <= key["address"]
            assert operation["output"]["end"] <= previous_ws["output"]["address"]
        panels.append(dict(panel=panel, qk=operation, next_key=next_key, previous_ws=previous_ws))

    cpp_lines = cpp.read_text().splitlines()
    markers = (
        "TEXTRACT(v3367, v3365, v3369, v3370);",
        "set_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);",
        "wait_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);",
    )
    reuse = []
    for lineno, line in enumerate(cpp_lines, 1):
        if any(marker in line for marker in markers) or ("TINSERT<" in line and "(v3373, v3379," in line):
            reuse.append(dict(line=lineno, text=line.strip()))
    assert len(reuse) == 4
    maxima = {
        space: max(x["end"] for x in all_allocations if x["space"] == space)
        for space in sorted({x["space"] for x in all_allocations})
    }
    bounds = dict(mat=524288, left=65536, right=65536, acc=131072, vec=188416)
    assert all(maxima[space] <= bounds[space] for space in maxima)
    summary = dict(
        cpu_passed=True, compile_attempts=1, controls_confirmed=expected,
        dynamic_single_mix=result["single_mix"], sync_start=result["sync_start"],
        supplied_sample_shapes=False, source_copy_equal=True, only_modified_files=changed,
        pto=str(pto), cpp=str(cpp), full_mix_maximum_allocation_end_bytes=maxima,
        indexer_qk_calls=6, indexer_ws_calls=6, actual_key_extracts=6,
        six_panels=panels, tail_l1_reuse_cpp= reuse,
        limitations="已有生成文本的操作数/地址与局部等待检查；非机器码、设备精度或性能证明。",
    )
    (root / "codegen_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    excerpt = []
    for start, stop in ((11865, 11891), (12265, 12311), (12704, 12774), (14202, 14269)):
        excerpt.append(f"{cpp}:{start}\n")
        excerpt.extend(f"{n}: {cpp_lines[n-1]}\n" for n in range(start, stop + 1))
    (root / "codegen_excerpt.txt").write_text("".join(excerpt))
    print(json.dumps({"cpu_passed": True, "six_panel_operand_check": True, "maximum_allocation_end_bytes": maxima,
                      "tail_reuse_lines": [x["line"] for x in reuse]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
