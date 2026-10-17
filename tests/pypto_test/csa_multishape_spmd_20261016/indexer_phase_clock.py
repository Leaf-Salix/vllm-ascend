# SPDX-License-Identifier: Apache-2.0
"""Indexer AIC 的 Scalar 提交区间探针，不测量各异步 pipe 的独占忙时。

记录原页表读取到 Key DMA 提交的代码区间，以及每个 1024 分数块的提交
区间。普通 wait_flag 在目标 pipe 上生效，不能用 Scalar 时间戳把它解释
成该 pipe 的等待耗时。只另计已有 wait_flag_dev；不增加中间流水屏障。
"""

import contextlib
import json
import re
import statistics
from pathlib import Path

from qk_phase_clock import close_brace


def instrument_cpp(source, layout, contract):
    if layout["protocol"]["progress_words_per_core"] != 128:
        raise ValueError("探针要求已有512B独占进度区")
    leaf_count = contract["leaf_tensor_count"]
    table_ids = [i for i, row in enumerate(contract["ordered_inputs"])
                 if row["reference"].get("allocation") == "argument:idx_block_table"]
    if len(table_ids) != 1:
        raise ValueError("Indexer页表的ABI绑定不唯一")
    match = re.search(r"^AICORE void csa_static_aic\(([^\n]+)\) \{", source, re.M)
    if match is None:
        raise ValueError("未找到AIC入口")
    declarations = match.group(1).split(",")
    params = [part.strip().split()[-1].lstrip("*") for part in declarations]
    if (len(params) != leaf_count + 2
            or "__gm__ int32_t*" not in declarations[leaf_count - 3]
            or "__gm__ int32_t*" not in declarations[table_ids[0]]):
        raise ValueError("进度/页表ABI改变")
    progress, core, table = params[leaf_count - 3], params[leaf_count], params[table_ids[0]]
    end = source.find("\n}", match.end())
    body = source[match.end():end]
    blocks = list(re.finditer(r"^  if \([^\n]+\) \{\n", body, re.M))
    batches = layout["batches"]["aic"]
    if len(blocks) != len(batches):
        raise ValueError("阶段布局改变")
    selected = [i for i, batch in enumerate(batches)
                if {row["kernel_name"] for row in batch["members"]}
                == {"indexer_score_topk_native_pair_aic"}]
    if len(selected) != 1:
        raise ValueError("Indexer AIC阶段不唯一")
    block = blocks[selected[0]]
    stop = body.find("\n  }", block.end())
    segment = body[block.end():stop]
    cache = list(re.finditer(
        r"^    dcci\(\(__gm__ void\*\)0, cache_line_t::ENTIRE_DATA_CACHE\);\n"
        r"(?:    pipe_barrier\([^\n]+\);\n)*    dsb\(DSB_DDR\);\n", segment, re.M))
    if len(cache) != 2:
        raise ValueError("原业务缓存维护边界不唯一")
    page_read = re.compile(r"^([ ]+)int32_t \w+ = \(" + re.escape(table)
                           + r"\)\[[^\n]+\];", re.M)
    loops = []
    for loop in re.finditer(r"^([ ]+)for \([^\n]+\) \{", segment, re.M):
        opening = loop.end() - 1
        closing = close_brace(segment, opening)
        contents = segment[opening:closing]
        if len(page_read.findall(contents)) == 32 and contents.count("TMATMUL(") == 32:
            loops.append((closing - opening, opening, closing, len(loop.group(1))))
    if not loops:
        raise ValueError("未找到原N64的1024分数块循环")
    _, opening, closing, indent = min(loops)
    reads = list(page_read.finditer(segment, opening, closing))
    ranges = []
    for index, read in enumerate(reads):
        branch = re.search(r"if \([^\n]+\) \{", segment[read.end():closing])
        if branch is None:
            raise ValueError("页表读取后没有奇偶地址分支")
        left = read.end() + branch.end() - 1
        right = close_brace(segment, left)
        otherwise = re.match(r"\s*else \{", segment[right + 1:])
        if otherwise is None:
            raise ValueError("页加载缺少另一基址分支")
        finish = close_brace(segment, right + otherwise.end()) + 1
        piece = segment[left:finish]
        if piece.count("TLOAD(") != 2 or "TMATMUL(" in piece:
            raise ValueError("页加载窗口不再是两个互斥TLOAD")
        if index + 1 < len(reads) and finish > reads[index + 1].start():
            raise ValueError("页加载提交区间重叠")
        ranges.append((read.start(), finish, len(read.group(1))))
    if len(ranges) != 32:
        raise ValueError("页加载数量改变")

    inserts = [(cache[0].end(),
                "    uint64_t ik_begin = get_sys_cnt();\n"
                "    uint64_t ik_key_ticks = 0, ik_step_ticks = 0, ik_ffts_ticks = 0;\n"
                "    uint32_t ik_key_count = 0, ik_step_count = 0, ik_ffts_count = 0;\n"),
               (cache[1].end(), "    uint64_t ik_end = get_sys_cnt();\n")]
    spacing = " " * (indent + 2)
    inserts.extend([(opening + 1, f"\n{spacing}uint64_t ik_step_begin = get_sys_cnt();"),
                    (closing, f"{spacing}ik_step_ticks += get_sys_cnt() - ik_step_begin;\n"
                              f"{spacing}++ik_step_count;\n" + " " * indent)])
    for index, (start, finish, spaces) in enumerate(ranges):
        spacing = " " * spaces
        inserts.extend([(start, f"{spacing}uint64_t ik_key_begin_{index} = get_sys_cnt();\n"),
                        (finish, f"\n{spacing}ik_key_ticks += get_sys_cnt() - ik_key_begin_{index};\n"
                                 f"{spacing}++ik_key_count;")])
    # wait_flag_dev 的时戳只包围已有调用；不测普通异步 wait_flag 的“等待”。
    waits = list(re.finditer(r"^([ ]+)(wait_flag_dev\([^\n]+\);)$", segment, re.M))
    if len(waits) != 2:
        raise ValueError("原Indexer FFTS等待点数量改变")
    for index, wait in enumerate(waits):
        spacing = wait.group(1)
        inserts.extend([(wait.start(), f"{spacing}uint64_t ik_ffts_begin_{index} = get_sys_cnt();\n"),
                        (wait.end(), f"\n{spacing}ik_ffts_ticks += get_sys_cnt() - ik_ffts_begin_{index};\n"
                                     f"{spacing}++ik_ffts_count;")])
    for position, text in sorted(inserts, reverse=True):
        segment = segment[:position] + text + segment[position:]
    segment += (f"\n    volatile __gm__ uint32_t* ik_dst = "
                f"(volatile __gm__ uint32_t*)({progress} + {core} * 128 + 16);\n"
                "    ik_dst[0] = (uint32_t)ik_begin; ik_dst[1] = (uint32_t)ik_end;\n"
                "    ik_dst[2] = (uint32_t)ik_key_ticks; ik_dst[3] = ik_key_count;\n"
                "    ik_dst[4] = (uint32_t)ik_step_ticks; ik_dst[5] = ik_step_count;\n"
                "    ik_dst[6] = (uint32_t)ik_ffts_ticks; ik_dst[7] = ik_ffts_count;\n")
    body = body[:block.end()] + segment + body[stop:]
    tail = body.rfind("  #endif")
    if tail < 0:
        raise ValueError("AIC结束边界改变")
    body = body[:tail] + "  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);\n  dsb(DSB_DDR);\n" + body[tail:]
    audit = {"stage": selected[0], "page_read_static_sites": len(ranges),
             "score_step_static_loops": 1, "ffts_wait_static_sites": len(waits),
             "ordinary_async_waits_not_timed": True, "middle_pipe_barriers_added": 0,
             "progress_first_signal_line_untouched": True, "table_leaf": table_ids[0]}
    return source[:match.end()] + body + source[end:], audit


@contextlib.contextmanager
def compiler_probe(layout, contract, output):
    from pypto.backend import pto_backend

    original = pto_backend._compile_pto_module
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)

    def compile_with_clock(pto_code, unit_name, output_dir, *args, **kwargs):
        source = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != "csa_static_group":
            return source
        transformed, audit = instrument_cpp(source, layout, contract)
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
    for core in range(24):
        values = [v & 0xFFFFFFFF for v in flat[core * 128 + 16:core * 128 + 24]]
        begin, end, key, key_count, step, step_count, ffts, ffts_count = values
        duration = ((end - begin) & 0xFFFFFFFF) / 50
        if (not duration or not step_count or key_count != 32 * step_count
                or key > step or step / 50 > duration or ffts / 50 > duration):
            raise ValueError(f"Indexer提交区间记录无效：core={core}, values={values}")
        rows.append({"core": core, "stage_us": duration,
                     "key_submit_us": key / 50, "key_count": key_count,
                     "score_step_us": step / 50, "step_count": step_count,
                     "ffts_us": ffts / 50, "ffts_count": ffts_count})
    report = {"eligible_for_700us_target": False,
              "scope": "末次重放的Scalar提交区间；异步pipe可以重叠，插桩会改变流水，不是各pipe独占耗时",
              "medians": {k: statistics.median(r[k] for r in rows) for k in rows[0] if k != "core"},
              "rows": rows}
    (Path(output) / "phases.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "rows"}
