# SPDX-License-Identifier: Apache-2.0
"""测试专用：给生成的 C++ 阶段加 SYS_CNT 时戳，不修改 PyPTO/Simpler 源码。

原 L1 只覆盖融合任务首尾。本探针区分等待、数学/缓存维护、发布三段，
有额外开销，只用于归因，绝不拿插桩时间证明 700 us 性能目标。
时间记录使用每核既有 512 B 进度区的空闲行，信号独占第一条 64 B 行。
"""

import contextlib
import json
import re
from pathlib import Path


def instrument_cpp(source, layout, leaf_count):
    # 一条记录四个 uint32，50 MHz 下约 85 秒回绕。一次 CSA 远小于该范围，
    # 解析时按模 2^32 求差。每核单写者，始终在本核的 512 B 区域内。
    limit = max(row["dispatch_seq"] for batches in layout["batches"].values()
                for batch in batches for row in batch["members"])
    if 16 + limit * 4 > 128:
        raise ValueError("当前进度区空闲行不足以存储全部阶段时戳")
    spans = list(re.finditer(r"^AICORE void csa_static_(aic|aiv)\(([^\n]+)\) \{", source, re.M))
    if len(spans) != 2:
        raise ValueError("无法唯一定位融合的两个物理函数")
    audit = {}
    for match in reversed(spans):
        role = match.group(1)
        end = source.find("\n}", match.end())
        if end < 0:
            raise ValueError("找不到物理函数结束")
        params = [part.strip().split()[-1].lstrip("*") for part in match.group(2).split(",")]
        # 叶子 ABI 的最后三个 Tensor 固定为 progress/records/waits，后接
        # 显式的 block_idx/block_num，AIV 再接 subblock_idx。逐项检查防止错绑。
        declarations = match.group(2).split(",")
        if "__gm__ int32_t*" not in declarations[leaf_count - 3] or "__gm__ int64_t*" not in declarations[leaf_count - 2]:
            raise ValueError("进度/调度 Tensor 的 C++ ABI 改变")
        if len(params) != leaf_count + (2 if role == "aic" else 3):
            raise ValueError("运行时核编号参数数量改变")
        progress, records, group = params[leaf_count - 3], params[leaf_count - 2], params[leaf_count]
        core = group if role == "aic" else f"(24 + 2 * {group} + {params[-1]})"
        body = source[match.end():end]
        blocks = list(re.finditer(r"^  if \([^\n]+\) \{\n", body, re.M))
        expected = layout["batches"][role]
        if len(blocks) != len(expected):
            raise ValueError(f"{role} 顶层控制块数量不匹配：{len(blocks)} != {len(expected)}")
        for index in reversed(range(len(blocks))):
            block = blocks[index]
            close = body.find("\n  }", block.end())
            segment = body[block.end():close]
            cache_sites = list(re.finditer(
                r"^    dcci\(\(__gm__ void\*\)0, cache_line_t::ENTIRE_DATA_CACHE\);\n"
                r"(?:    pipe_barrier\([^\n]+\);\n)*    dsb\(DSB_DDR\);\n", segment, re.M))
            if len(cache_sites) != 2:
                raise ValueError(f"{role}/{index} 的两处 payload 维护边界不唯一")
            record_offset = expected[index]["record_offset_words"]
            width = layout["protocol"]["record_width"]
            stamp_base = f"({core} * 128 + 16 + 4 * ({records}[{record_offset} + {core} * {width} + 2] - 1))"
            declaration = (f"    // CSA 阶段探针 {role}/{index}；不参与正式性能达标判定。\n"
                           f"    volatile __gm__ uint32_t* csa_clock_{index} = "
                           f"(volatile __gm__ uint32_t*)({progress} + {stamp_base});\n")

            def stamp(point):
                return ("    __asm__ __volatile__(\"\" ::: \"memory\");\n"
                        f"    csa_clock_{index}[{point}] = (uint32_t)get_sys_cnt();\n")

            # 先从后往前插入，保持前面缓存维护边界的字符串偏移有效。
            for point, site in reversed(list(enumerate(cache_sites, 1))):
                segment = segment[:site.end()] + stamp(point) + segment[site.end():]
            segment = declaration + stamp(0) + segment + "\n" + stamp(3).rstrip("\n")
            body = body[:block.end()] + segment + body[close:]
        # 最后一处时戳写在本轮发布之后，单独清理本核 DCache 并排空写入。
        # 此处只有本核屏障，无全核集合，也不改变信号所在第一条缓存行。
        tail = body.rfind("  #endif")
        if tail < 0:
            raise ValueError("找不到物理核条件编译边界")
        body = body[:tail] + ("  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);\n"
                             "  dsb(DSB_DDR);\n") + body[tail:]
        source = source[:match.end()] + body + source[end:]
        audit[role] = {"batches": len(blocks), "timestamp_slots_per_core": limit * 4,
                       "signal_line_untouched": True}
    return source, audit


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
        (directory / "original.cpp").write_text(source)
        transformed, audit = instrument_cpp(source, layout, leaf_count)
        (directory / "instrumented.cpp").write_text(transformed)
        (directory / "instrumentation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        return transformed

    pto_backend._compile_pto_module = compile_with_clock
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original


def export_stage_clock(progress, layout, output):
    flat = progress.cpu().tolist()
    entries = []
    for role, batches in layout["batches"].items():
        for index, batch in enumerate(batches):
            for member in batch["members"]:
                base = member["core"] * 128 + 16 + 4 * (member["dispatch_seq"] - 1)
                stamps = [value & 0xFFFFFFFF for value in flat[base:base + 4]]
                entries.append({"core": member["core"], "role": role, "stage": index,
                                "sequence": member["dispatch_seq"], "kernel": batch["kernel"],
                                "task": batch["task"], "cycles": stamps})
    # 起点在单次调用内统一展开，兼容低 32 位计数器恰好回绕。
    anchor = entries[0]["cycles"][0]
    for row in entries:
        offsets = [((value - anchor + 2**31) % 2**32 - 2**31) / 50 for value in row["cycles"]]
        if any(b < a for a, b in zip(offsets, offsets[1:])) or offsets[-1] - offsets[0] > 1000000:
            raise ValueError(f"阶段时戳顺序异常：{row}")
        row["times_us"] = offsets
        row["wait_us"], row["math_and_fence_us"], row["notify_us"] = [b - a for a, b in zip(offsets, offsets[1:])]
    earliest = min(row["times_us"][0] for row in entries)
    trace = []
    for row in entries:
        for phase, start, end in zip(("等待与读侧维护", "数学与写侧维护", "完成发布"), row["times_us"], row["times_us"][1:]):
            trace.append({"name": row["kernel"] + " / " + phase, "cat": phase, "ph": "X", "pid": 0,
                          "tid": row["core"], "ts": start - earliest, "dur": end - start,
                          "args": {"sequence": row["sequence"], "task": row["task"]}})
    output = Path(output)
    (output / "stages.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n")
    (output / "stages_trace.json").write_text(json.dumps({"traceEvents": trace}, ensure_ascii=False) + "\n")
    return {"members": len(entries), "clock_freq_hz": 50000000,
            "instrumented_span_us": max(row["times_us"][-1] for row in entries) - earliest,
            "eligible_for_700us_target": False}
