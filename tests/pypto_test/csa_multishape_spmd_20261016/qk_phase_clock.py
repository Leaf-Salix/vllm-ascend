# SPDX-License-Identifier: Apache-2.0
"""QK/PV AIV分段SYS_CNT诊断，不增加阶段内pipe屏障。

记录Scalar执行四段代码的墙钟时间，可能包含现有等待，也允许其他pipe
重叠运行；不是四条流水的独占忙时。插桩结果禁止参与700μs目标。
"""

import contextlib
import json
import re
import statistics
from pathlib import Path


def close_brace(source, opening):
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("生成代码括号不闭合")


def phase_regions(segment):
    # 最内层同时包含softmax和最终除法的for，就是跨query的qk_tick循环。
    loops = []
    for match in re.finditer(r"^([ ]+)for \([^\n]+\) \{", segment, re.M):
        start, end = match.end() - 1, close_brace(segment, match.end() - 1)
        text = segment[start:end]
        if "TROWMAX(" in text and "TROWEXPANDDIV(" in text and "TGATHER(" in text:
            loops.append((end - start, start, end, len(match.group(1))))
    if not loops:
        raise ValueError("未找到QK tick循环")
    _, start, end, indent = min(loops)

    def children(left, right, spaces):
        result = []
        for match in re.finditer(r"^([ ]+)if \([^\n]+\) \{", segment[left:right], re.M):
            if len(match.group(1)) == spaces:
                opening = left + match.end() - 1
                closing = close_brace(segment, opening)
                result.append((opening, closing, spaces))
        return result

    outer = children(start + 1, end, indent + 2)
    softmax = [r for r in outer if "TROWMAX(" in segment[r[0]:r[1]]]
    kv = [r for r in outer if "TGATHER(" in segment[r[0]:r[1]]
          and "TROWEXPANDDIV(" not in segment[r[0]:r[1]]]
    pv = [r for r in outer if "TROWEXPANDDIV(" in segment[r[0]:r[1]]]
    if [len(softmax), len(kv), len(pv)] != [1, 1, 1]:
        raise ValueError("QK阶段模式不唯一")
    inner = children(pv[0][0] + 1, pv[0][1], indent + 4)
    publish = [r for r in inner if "TROWEXPANDDIV(" in segment[r[0]:r[1]]]
    merge = [r for r in inner if "wait_flag_dev(" in segment[r[0]:r[1]]]
    if len(publish) != 1 or len(merge) != 1:
        raise ValueError("PV合并和输出边界不唯一")
    return {"kv": kv[0], "softmax": softmax[0], "pv_merge": merge[0], "publish": publish[0]}


def instrument_cpp(source, layout, leaf_count):
    if layout["protocol"]["progress_words_per_core"] != 128:
        raise ValueError("探针要求512B独占进度区")
    functions = list(re.finditer(r"^AICORE void csa_static_aiv\(([^\n]+)\) \{", source, re.M))
    if len(functions) != 1:
        raise ValueError("AIV函数不唯一")
    match = functions[0]
    declarations = match.group(1).split(",")
    params = [p.strip().split()[-1].lstrip("*") for p in declarations]
    if len(params) != leaf_count + 3 or "__gm__ int32_t*" not in declarations[leaf_count - 3]:
        raise ValueError("进度区ABI改变")
    progress = params[leaf_count - 3]
    core = f"(24 + 2 * {params[leaf_count]} + {params[-1]})"
    end = source.find("\n}", match.end())
    body = source[match.end():end]
    blocks = list(re.finditer(r"^  if \([^\n]+\) \{\n", body, re.M))
    if len(blocks) != len(layout["batches"]["aiv"]):
        raise ValueError("阶段布局不符")
    selected = [i for i, b in enumerate(layout["batches"]["aiv"])
                if {r["kernel_name"] for r in b["members"]} == {"qk_pv_aiv"}]
    if len(selected) != 1:
        raise ValueError("QK/PV AIV阶段不唯一")
    block = blocks[selected[0]]
    close = body.find("\n  }", block.end())
    segment = body[block.end():close]
    sites = list(re.finditer(
        r"^    dcci\(\(__gm__ void\*\)0, cache_line_t::ENTIRE_DATA_CACHE\);\n"
        r"(?:    pipe_barrier\([^\n]+\);\n)*    dsb\(DSB_DDR\);\n", segment, re.M))
    if len(sites) != 2:
        raise ValueError("原缓存维护边界不唯一")
    regions = phase_regions(segment)
    insertions = [(sites[0].end(), "    uint64_t qkp_begin = get_sys_cnt();\n"
                   + "".join(f"    uint64_t qkp_{name}_ticks = 0; uint32_t qkp_{name}_count = 0;\n"
                             for name in regions)),
                  (sites[1].end(), "    uint64_t qkp_end = get_sys_cnt();\n")]
    audit = []
    for slot, (name, (opening, closing, indent)) in enumerate(regions.items()):
        spacing = " " * (indent + 2)
        insertions.append((opening + 1, f"\n{spacing}uint64_t qkp_{name}_before = get_sys_cnt();\n"
                           f"{spacing}__asm__ __volatile__(\"\" ::: \"memory\");"))
        insertions.append((closing, f"  __asm__ __volatile__(\"\" ::: \"memory\");\n"
                           f"{spacing}qkp_{name}_ticks += get_sys_cnt() - qkp_{name}_before;\n"
                           f"{spacing}qkp_{name}_count += 1;\n" + " " * indent))
        audit.append({"phase": name, "slot": slot, "body_bytes": closing - opening,
                      "source_begin": segment[opening + 1:opening + 150].strip()})
    for position, text in sorted(insertions, reverse=True):
        segment = segment[:position] + text + segment[position:]
    for slot, name in enumerate(regions):
        segment += (f"    volatile __gm__ uint32_t* qkp_{name}_dst = "
                    f"(volatile __gm__ uint32_t*)({progress} + {core} * 128 + {16 + slot * 16});\n"
                    f"    qkp_{name}_dst[0] = (uint32_t)qkp_{name}_ticks;\n"
                    f"    qkp_{name}_dst[1] = qkp_{name}_count;\n")
    segment += (f"    volatile __gm__ uint32_t* qkp_total = "
                f"(volatile __gm__ uint32_t*)({progress} + {core} * 128 + 80);\n"
                "    qkp_total[0] = (uint32_t)qkp_begin;\n    qkp_total[1] = (uint32_t)qkp_end;\n")
    body = body[:block.end()] + segment + body[close:]
    tail = body.rfind("  #endif")
    if tail < 0:
        raise ValueError("AIV核尾部不唯一")
    body = body[:tail] + "  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);\n  dsb(DSB_DDR);\n" + body[tail:]
    return source[:match.end()] + body + source[end:], audit


@contextlib.contextmanager
def compiler_probe(layout, leaf_count, output):
    from pypto.backend import pto_backend

    original = pto_backend._compile_pto_module
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)

    def compile_with_clock(pto_code, unit_name, output_dir, *args, **kwargs):
        source = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != "csa_static_group":
            return source
        transformed, audit = instrument_cpp(source, layout, leaf_count)
        (directory / "original.cpp").write_text(source)
        (directory / "instrumented.cpp").write_text(transformed)
        (directory / "instrumentation.json").write_text(json.dumps(audit, indent=2) + "\n")
        return transformed

    pto_backend._compile_pto_module = compile_with_clock
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original


def export_clock(progress, output):
    flat = progress.cpu().tolist()
    rows = []
    for core in range(24, 72):
        begin, end = [v & 0xFFFFFFFF for v in flat[core * 128 + 80:][:2]]
        duration = ((end - begin) & 0xFFFFFFFF) / 50
        row = {"core": core, "stage_us": duration}
        for slot, name in enumerate(("kv", "softmax", "pv_merge", "publish")):
            ticks, count = [v & 0xFFFFFFFF for v in flat[core * 128 + 16 + slot * 16:][:2]]
            if not count or not duration or ticks / 50 > duration:
                raise ValueError(f"分段时戳无效：{core}/{name}")
            row[name + "_us"], row[name + "_count"] = ticks / 50, count
        rows.append(row)
    medians = {k: statistics.median(r[k] for r in rows) for k in rows[0] if k != "core"}
    report = {"eligible_for_700us_target": False,
              "scope": "末次重放Scalar分段墙钟，不增加中间pipe屏障；不是各pipe独占忙时，插桩扰动流水",
              "medians": medians, "rows": rows}
    (Path(output) / "phases.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "rows"}
