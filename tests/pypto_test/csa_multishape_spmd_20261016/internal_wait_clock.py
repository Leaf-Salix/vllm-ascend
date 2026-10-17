# SPDX-License-Identifier: Apache-2.0
"""只诊断Indexer/QK内部FFTS等待；插桩耗时不能参与性能目标。

SYS_CNT包围wait_flag_dev，记录Scalar停在该等待上的累计时间和次数。
其他流水可能同时执行，因此它不等于整核空闲或可直接消除的耗时。
使用每核原512B进度区的第2/3条独占缓存行，不碰首行通知信号。
"""

import contextlib
import json
import re
import statistics
from pathlib import Path


KERNELS = {"indexer_score_topk_native_pair_aic": 0,
           "indexer_score_topk_native_pair_aiv": 0,
           "qk_pv_aic": 1, "qk_pv_aiv": 1}


def instrument_cpp(source, layout, leaf_count):
    if layout["protocol"]["progress_words_per_core"] != 128:
        raise ValueError("探针要求已有512B独占进度区")
    functions = list(re.finditer(r"^AICORE void csa_static_(aic|aiv)\(([^\n]+)\) \{", source, re.M))
    if len(functions) != 2:
        raise ValueError("两个物理函数不唯一")
    audit = []
    for match in reversed(functions):
        role = match.group(1)
        end = source.find("\n}", match.end())
        declarations = match.group(2).split(",")
        params = [part.strip().split()[-1].lstrip("*") for part in declarations]
        if (len(params) != leaf_count + (2 if role == "aic" else 3)
                or "__gm__ int32_t*" not in declarations[leaf_count - 3]):
            raise ValueError("进度区ABI改变")
        progress, group = params[leaf_count - 3], params[leaf_count]
        core = group if role == "aic" else f"(24 + 2 * {group} + {params[-1]})"
        body = source[match.end():end]
        blocks = list(re.finditer(r"^  if \([^\n]+\) \{\n", body, re.M))
        batches = layout["batches"][role]
        if len(blocks) != len(batches):
            raise ValueError("生成阶段与布局不匹配")
        role_hits = []
        for index in reversed(range(len(blocks))):
            names = {r["kernel_name"] for r in batches[index]["members"]}
            targets = names.intersection(KERNELS)
            if not targets:
                continue
            if len(names) != 1 or len(targets) != 1:
                raise ValueError("探针阶段混入其他数学核")
            kernel = next(iter(targets))
            slot = KERNELS[kernel]
            role_hits.append(slot)
            block = blocks[index]
            close = body.find("\n  }", block.end())
            segment = body[block.end():close]
            sites = list(re.finditer(
                r"^    dcci\(\(__gm__ void\*\)0, cache_line_t::ENTIRE_DATA_CACHE\);\n"
                r"(?:    pipe_barrier\([^\n]+\);\n)*    dsb\(DSB_DDR\);\n", segment, re.M))
            if len(sites) != 2:
                raise ValueError("两处原业务缓存边界不唯一")
            begin, finish = sites[0].end(), sites[1].end()
            tag = f"csa_iwait_{slot}"
            counter = {"sites": 0}

            def wrap(wait):
                counter["sites"] += 1
                indent, call = wait.group(1), wait.group(2)
                return (f"{indent}{{\n{indent}  uint64_t before = get_sys_cnt();\n"
                        f"{indent}  __asm__ __volatile__(\"\" ::: \"memory\");\n"
                        f"{indent}  {call}\n"
                        f"{indent}  __asm__ __volatile__(\"\" ::: \"memory\");\n"
                        f"{indent}  {tag}_ticks += get_sys_cnt() - before;\n"
                        f"{indent}  {tag}_count += 1;\n{indent}}}")

            middle = re.sub(r"^([ ]+)(wait_flag_dev\([^\n]+\);)$", wrap,
                            segment[begin:finish], flags=re.M)
            if counter["sites"] == 0:
                raise ValueError("阶段没有内部wait_flag_dev")
            declaration = (f"    uint64_t {tag}_ticks = 0;\n    uint32_t {tag}_count = 0;\n"
                           f"    uint64_t {tag}_begin = get_sys_cnt();\n")
            ending = f"    uint64_t {tag}_end = get_sys_cnt();\n"
            # 在该阶段原通知之后写第二/三条行；仅本核单写者。
            stores = (f"    volatile __gm__ uint32_t* {tag}_dst = "
                      f"(volatile __gm__ uint32_t*)({progress} + {core} * 128 + {16 + slot * 16});\n"
                      f"    {tag}_dst[0] = (uint32_t){tag}_begin;\n"
                      f"    {tag}_dst[1] = (uint32_t){tag}_end;\n"
                      f"    {tag}_dst[2] = (uint32_t){tag}_ticks;\n"
                      f"    {tag}_dst[3] = {tag}_count;\n")
            segment = segment[:begin] + declaration + middle + ending + segment[finish:] + stores
            body = body[:block.end()] + segment + body[close:]
            audit.append({"role": role, "kernel": kernel, "batch": index,
                          "slot": slot, "wait_static_sites": counter["sites"]})
        if sorted(role_hits) != [0, 1]:
            raise ValueError("探针要求每侧各一份Indexer和QK/PV")
        tail = body.rfind("  #endif")
        if tail < 0:
            raise ValueError("没有物理核尾部")
        body = body[:tail] + ("  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);\n"
                             "  dsb(DSB_DDR);\n") + body[tail:]
        source = source[:match.end()] + body + source[end:]
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
    for core in range(72):
        for slot, name in enumerate(("indexer", "qk_pv")):
            begin, end, ticks, count = [v & 0xFFFFFFFF for v in flat[core * 128 + 16 + slot * 16:][:4]]
            duration = (end - begin) & 0xFFFFFFFF
            if not count or not duration or ticks > duration:
                raise ValueError(f"内部等待探针无效：{core}/{name}")
            rows.append({"core": core, "role": "aic" if core < 24 else "aiv",
                         "stage": name, "elapsed_us": duration / 50,
                         "scalar_wait_us": ticks / 50, "wait_count": count,
                         "scalar_wait_fraction": ticks / duration})
    groups = []
    for stage in ("indexer", "qk_pv"):
        for role in ("aic", "aiv"):
            group = [r for r in rows if (r["stage"], r["role"]) == (stage, role)]
            groups.append({"stage": stage, "role": role, "cores": len(group),
                           **{f"median_{key}": statistics.median(r[key] for r in group)
                              for key in ("elapsed_us", "scalar_wait_us", "wait_count", "scalar_wait_fraction")}})
    report = {"eligible_for_700us_target": False, "clock_hz": 50000000,
              "scope": "末次图重放的Scalar内部FFTS等待；不是整核空闲或可消除耗时，插桩会改变流水",
              "groups": groups, "rows": rows}
    (Path(output) / "waits.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "rows"}
