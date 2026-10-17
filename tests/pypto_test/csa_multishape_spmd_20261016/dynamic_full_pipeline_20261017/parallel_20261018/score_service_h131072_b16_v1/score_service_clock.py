# SPDX-License-Identifier: Apache-2.0
"""正式 668 长 G6 Score 的测试专用 Scalar 服务区间时钟。

整体思路：只改本次编译返回的 C++，不改 PyPTO 数学、原分支、事件或 ABI。
全体真实 score step 和 FIFO ACK 累计；细分仅采样固定第九个 N64 面板，
分别观察页/Key 发射、QK+FIX、WS 及其原局部 wait。各层独立 start，避免
body→step→segment→wait 的嵌套覆盖。所有累计在本次 kernel 入口清零，
结束时仅写 phase34/AIC0..23 的 32..63 项，使用紧随其后的原 release。

该记录不能分离异步引擎的纯耗时，不能乘 16 外推；新增 Scalar 工作会扰动
流水。禁止同时启用 Attention tick、FFTS 20/21 插桩。本模块只复用既有
phase_clock 的 16..19；后者自己的既有记录维护不算本 hook 新增 fence。
import 不启动编译/设备。仅 compilation_probe 上下文临时安装后端返回值钩子。
"""

import contextlib
import json
import re
import statistics
from pathlib import Path

PHASE = 34
CORES = 72
AIC_CORES = 24
WORDS = 128
SCHEMA = 0x53430108  # SC、版本1、固定panel8；不是输入shape编码。
SAMPLE_PANEL = 8
LOCAL_WAIT_SITES = 10  # 固定生成面板内的语句数；不涉及T/B或运行时step数。
TICK_NAMES = (
    "body_total", "step_total", "ack_issue", "page_key_sample",
    "qk_fix_sample", "ws_sample", "local_wait_sample",
)
TAG = re.compile(r"/\* CSA_SCORE_PROBE_BEGIN \*/.*?/\* CSA_SCORE_PROBE_END \*/", re.S)
SYNC = re.compile(
    r"\b(?:pto::comm::TNOTIFY|pto::comm::TWAIT|CSA_WAIT_LOCAL_GM|"
    r"wait_flag_dev|ffts_cross_core_sync|wait_flag|set_flag|pipe_barrier|dsb|dcci)\([^\n]+\);"
)


def _block(code):
    return "/* CSA_SCORE_PROBE_BEGIN */\n" + code + "\n/* CSA_SCORE_PROBE_END */"


def _close(text, opening):
    depth = 1
    for pos in range(opening + 1, len(text)):
        depth += (text[pos] == "{") - (text[pos] == "}")
        if depth == 0:
            return pos
    raise ValueError("生成 C++ 大括号不完整")


def _function(text, name):
    # raw返回值用AICORE，既有wrapper已展开为static __aicore__；两者分开审计。
    matches = list(re.finditer(r"(?m)^(?:AICORE|static __aicore__) void " + name + r"\(", text))
    if len(matches) != 1:
        raise ValueError("函数不唯一：" + name)
    start = matches[0].start()
    opening = text.index("{", start)
    end = _close(text, opening) + 1
    return start, end, text[start:end]


def _one(matches, name):
    matches = list(matches)
    if len(matches) != 1:
        raise ValueError(f"锚点 {name} 命中 {len(matches)} 次，拒绝猜测")
    return matches[0]


def _previous_marker(text, prefix, before, lower):
    matches = list(re.finditer(r"(?m)^ *// pto: %" + prefix + r"[^\n]+", text[lower:before]))
    if not matches:
        raise ValueError("缺失数学注释锚点：" + prefix)
    # Tile 的声明/地址各重复一次注释，取同一SSA名字的第一处。
    name = matches[-1][0].strip()
    same = [m for m in matches if m[0].strip() == name]
    return lower + same[0].start()


def _line_end(text, pos):
    end = text.find("\n", pos)
    return len(text) if end < 0 else end + 1


def _after_flags(text, pos):
    """只跨过原紧邻 set_flag；不越过下一块数学或 wait。"""
    while True:
        match = re.match(r"\s*set_flag\([^\n]+\);", text[pos:])
        if not match:
            return pos
        pos += match.end()


def instrument_cpp(raw):
    """返回标记 C++/审计；只接受已经核对的 G6/N64/1024 一份 Score。"""
    if "CSA_SCORE_PROBE_BEGIN" in raw or "CSA_ATTN_PROBE_BEGIN" in raw:
        raise ValueError("不得重复插入 Score 或混入 Attention tick")
    start, end, source = _function(raw, "cube_schedule")
    _vs, _ve, vector_before = _function(raw, "vector_schedule")
    guard = _one(re.finditer(r"#if defined\(__DAV_CUBE__\)\n", source), "Cube guard")
    notify = _one(list(re.finditer(r"pto::comm::TNOTIFY\((\w+), (\w+), (\w+)\);", source))[:1], "BEGIN notify")
    own_phase0, generation = notify[1], notify[2]

    # 先按嵌套结构锁定唯一step，不依赖编译器临时变量序号或历史行号。
    loops = []
    for match in re.finditer(r"for \(int64_t (\w+) = [^\n]+\) \{", source):
        last = _close(source, match.end() - 1)
        loops.append((match, last))
    step, step_end = _one(
        ((m, last) for m, last in loops
         if "buf_scores_inline" in source[m.end():last]
         and source[m.end():last].count("ffts_cross_core_sync(") == 1
         and source[m.end():last].count("wait_flag_dev(") == 1),
        "Score step",
    )
    outers = [(m, last) for m, last in loops if m.start() < step.start() and last > step_end]
    outer, outer_end = min(outers, key=lambda item: item[1] - item[0].start())
    step_text = source[step.end():step_end]
    # type与matmul的定义/使用关系同时校验，不能只凭注释中的buf名字。
    qk_defs = set(re.findall(r"(v\d+) = Tile<TileType::Acc, int32_t, 384, 64,", step_text))
    ws_defs = set(re.findall(r"(v\d+) = Tile<TileType::Acc, float, 16, 64,", step_text))
    calls = list(re.finditer(r"TMATMUL\((v\d+), (v\d+), (v\d+)\);", source[step.end():step_end]))
    qks = [m for m in calls if m[1] in qk_defs]
    wss = [m for m in calls if m[1] in ws_defs]
    if len(qks) != 16 or len(wss) != 16 or len(calls) != 32:
        raise ValueError("只接受 G6/N64/1024 的16 QK+16 WS，禁止静默套用其它策略")
    abspos = lambda m: step.end() + m.start()
    qk, ws = qks[SAMPLE_PANEL], wss[SAMPLE_PANEL - 1]
    qk_pos, ws_pos = abspos(qk), abspos(ws)
    fix = _one(re.finditer(r"(?m)^ *TINSERT<[^\n]+\([^\n]*, " + qk[1] + r",[^\n]*\);", source[qk_pos:step_end]), "sample FIX")
    fix_pos, fix_end = qk_pos + fix.start(), qk_pos + fix.end()
    store = _one(re.finditer(r"TSTORE\(\w+, " + ws[1] + r"\);", source[ws_pos:step_end]), "sample WS store")
    store_end = ws_pos + store.end()
    qk_begin = _previous_marker(source, "buf_scores_inline", qk_pos, step.end())
    ws_math_begin = _previous_marker(source, "buf_reduced_pair_inline", ws_pos, qk_pos)
    ws_extract_begin = _previous_marker(source, "buf_previous_pair_inline", qk_pos, step.end())
    key_begin = _previous_marker(source, "buf_panel_keys_inline", ws_extract_begin, step.end())
    extracts = list(re.finditer(r"TEXTRACT\([^\n]+\);", source[key_begin:ws_extract_begin]))
    key_end = _after_flags(source, key_begin + _one(extracts, "sample Key TEXTRACT").end())
    wait_fix = _one(re.finditer(r"wait_flag\(PIPE_M, PIPE_FIX, [^\n]+\);", source[ws_pos + len(ws[0]):fix_pos]), "sample QK→FIX wait")
    fix_begin = ws_pos + len(ws[0]) + wait_fix.start()
    fix_after = _after_flags(source, fix_end)
    ws_store_begin = source.index(ws[1] + ".SetValidShape(", fix_after, store_end)
    ws_store_end = _after_flags(source, store_end)
    # last DSB是调用helper之前原有acquire的末端；完整保留它，不把输入等待计入body。
    before = source.rfind("dsb(DSB_DDR);", 0, outer.start())
    if before < 0 or outer.start() - before > 4000:
        raise ValueError("无法定位 Score 输入 acquire 后的原始边界")
    body_begin = before + len("dsb(DSB_DDR);")
    release = source.index("pipe_barrier(PIPE_MTE2);", outer_end)
    next_notify = re.search(r"pto::comm::TNOTIFY\([^\n]+\);", source[release:])
    if not next_notify or "dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);" not in source[release:release + next_notify.start()]:
        raise ValueError("结束必须位于原整cache release之前")
    if not re.search(r"const int64_t \w+ = 313344;", source):
        raise ValueError("原phase34=34*72*128常量缺失")

    insertions, anchors = [], []
    def add(pos, code, name):
        insertions.append((pos, _block(code)))
        anchors.append({"name": name, "function_line": source[:pos].count("\n") + 1,
                        "context": source[max(0, pos - 80):pos + 110]})
    serial = 0
    def measure(name, left, right, count=None):
        nonlocal serial
        if not left < right:
            raise ValueError("时钟区间倒序：" + name)
        ident = serial
        serial += 1
        var = f"csa_score_start_{ident}"
        add(left, f'  __asm__ __volatile__("" ::: "memory");\n  const uint64_t {var} = get_sys_cnt();\n  __asm__ __volatile__("" ::: "memory");', name + ":begin")
        code = (f'  __asm__ __volatile__("" ::: "memory");\n'
                f'  csa_score_{name} += get_sys_cnt() - {var};\n'
                '  __asm__ __volatile__("" ::: "memory");')
        if count:
            code += f"\n  csa_score_{count} += 1;"
        add(right, code, name + ":end")

    decl = "\n".join("  uint64_t csa_score_" + name + " = 0;" for name in TICK_NAMES)
    decl += "\n" + "\n".join("  uint32_t csa_score_" + name + " = 0;" for name in ("leaves", "steps", "acks", "samples", "local_waits"))
    add(guard.end(), decl, "per_invocation_zero")
    measure("body_total", body_begin, release)
    ready = _one(re.finditer(r"ffts_cross_core_sync\([^\n]+\);", step_text), "Score READY")
    measure("step_total", step.end(), step.end() + ready.end(), "steps")
    ack_sites = list(re.finditer(r"wait_flag_dev\([^\n]+\);", source[outer.start():outer_end]))
    if len(ack_sites) != 2:
        raise ValueError("Score必须保留step复用和leaf排空两个ACK站点")
    for match in ack_sites:
        measure("ack_issue", outer.start() + match.start(), outer.start() + match.end(), "acks")
    query = _previous_marker(source, "buf_query_vector_inline", step.start(), outer.start())
    add(query, "  csa_score_leaves += 1;", "valid_leaf_count")
    add(key_begin, "  csa_score_samples += 1;", "sample_count")
    spans = [("page_key_sample", key_begin, key_end),
             ("qk_fix_sample", qk_begin, ws_math_begin),
             ("qk_fix_sample", fix_begin, fix_after),
             ("ws_sample", ws_extract_begin, qk_begin),
             ("ws_sample", ws_math_begin, fix_begin),
             ("ws_sample", ws_store_begin, ws_store_end)]
    local_sites = []
    for name, left, right in spans:
        measure(name, left, right)
        for match in re.finditer(r"wait_flag\([^\n]+\);", source[left:right]):
            a, b = left + match.start(), left + match.end()
            local_sites.append((a, b))
            measure("local_wait_sample", a, b, "local_waits")
    if len(set(local_sites)) != len(local_sites) or len(local_sites) != LOCAL_WAIT_SITES:
        raise ValueError("采样段重叠或缺少局部wait")
    # 单一退出点：在既有release之前一次写padding。generation最后写，不新增DCCI。
    stores = [f"  volatile __gm__ uint32_t* csa_score_dst = reinterpret_cast<volatile __gm__ uint32_t*>({own_phase0}.data()) + {PHASE * CORES * WORDS + 32};",
              f"  csa_score_dst[0] = {SCHEMA}u;",
              "  csa_score_dst[2] = csa_score_leaves;", "  csa_score_dst[3] = csa_score_steps;"]
    for i, name in enumerate(TICK_NAMES):
        stores += [f"  csa_score_dst[{4 + 2*i}] = static_cast<uint32_t>(csa_score_{name});",
                   f"  csa_score_dst[{5 + 2*i}] = static_cast<uint32_t>(csa_score_{name} >> 32);"]
    for pos, expression in [(18, "csa_score_acks"), (19, "csa_score_samples"), (20, "csa_score_local_waits"), (21, str(len(local_sites)) + "u")]:
        stores.append(f"  csa_score_dst[{pos}] = {expression};")
    stores += [f"  csa_score_dst[{i}] = 0;" for i in range(22, 32)]
    stores.append(f"  csa_score_dst[1] = static_cast<uint32_t>({generation});")
    add(release, "\n".join(stores), "single_padding_write_before_original_release")
    changed = source
    # 相同offset时保留登记顺序：body结束先计时，再输出counter；嵌套start各自独立。
    for index, (pos, code) in sorted(enumerate(insertions), key=lambda x: (x[1][0], x[0]), reverse=True):
        changed = changed[:pos] + code + changed[pos:]
    marked = raw[:start] + changed + raw[end:]
    if TAG.sub("", marked) != raw:
        raise ValueError("删除标记后不能逐字恢复原CPP")
    if SYNC.findall(marked) != SYNC.findall(raw):
        raise ValueError("原同步序列发生变化")
    if _function(marked, "vector_schedule")[2] != vector_before:
        raise ValueError("Vector函数必须逐字不变")
    audit = {"original_source_recovered": True, "original_sync_sequence_equal": True,
             "vector_function_unchanged": True, "cube_active_guard_only": True,
             "added_hardware_fences": 0, "local_wait_sites_per_sample": len(local_sites),
             "qk_sites": 16, "ws_sites": 16, "sample_panel": SAMPLE_PANEL,
             "counter_start_variables_unique": serial,
             "phase0_tensor": own_phase0, "generation_variable": generation,
             "signal_phase": PHASE, "signal_columns": [32, 63], "writers": [0, 23],
             "points": anchors, "segments": [{"name": name, "start_function_line": source[:a].count("\n")+1,
                                                 "end_function_line": source[:b].count("\n")+1} for name,a,b in spans],
             "limitations": ["Scalar原位区间而非异步引擎纯耗时", "nested区间不能相加", "固定j8不可乘16外推", "未验证最终CCEC时钟顺序/spill"]}
    return marked, audit


@contextlib.contextmanager
def compilation_probe(directory):
    """沿既有测试入口临时包裹一次返回值；退出时恢复后端，既不安装也不自发编译。"""
    from pypto.backend import pto_backend
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.wait_codegen import rewrite_wait_source
    from compile_diagnostics import instrument_clock

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    original = pto_backend._compile_pto_module
    def hooked(pto_code, unit_name, output_dir, *args, **kwargs):
        raw = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != "dynamic_csa_full":
            return raw
        marked, audit = instrument_cpp(raw)
        marked, audit["wait_backoff"] = rewrite_wait_source(marked)
        marked, audit["phase_clock"] = instrument_clock(marked, wait_backoff=True)
        audit.update(original_cpp_bytes=len(raw.encode()), instrumented_cpp_bytes=len(marked.encode()),
                     output_dir=str(output_dir),
                     raw_cpp_path_candidates=[str(p) for p in Path(output_dir).rglob(unit_name + ".cpp")],
                     note="不臆造raw_cpp路径；候选列表按当时实际文件登记，wrapper由正常构建输出保存。")
        (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
        return marked
    pto_backend._compile_pto_module = hooked
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original


def _slots(signal):
    return signal.tolist() if hasattr(signal, "tolist") else signal


def check_probe_padding(signal, generation):
    """只返回bool；严格检查reserved、当前schema/epoch和实际计数，不固定22步。"""
    try:
        slots = _slots(signal)
        if len(slots) <= PHASE or generation <= 0:
            return False
        for phase, cores in enumerate(slots):
            if len(cores) != CORES:
                return False
            for core, values in enumerate(cores):
                if len(values) != WORDS:
                    return False
                vals = [int(v) & 0xffffffff for v in values]
                allowed = set(range(16, 20))
                active = phase == PHASE and core < AIC_CORES
                if active:
                    allowed.update(range(32, 64))
                if any(vals[col] != 0 for col in range(1, WORDS) if col not in allowed):
                    return False
                if phase == 0 and vals[0] != generation:
                    return False
                if active:
                    row = vals[32:64]
                    if vals[0] != generation or row[0] != SCHEMA or row[1] != generation:
                        return False
                    leaves, steps, acks, samples, waits, sites = row[2], row[3], row[18], row[19], row[20], row[21]
                    if not (leaves <= steps and acks == steps and samples == steps and sites == LOCAL_WAIT_SITES and waits == sites * samples):
                        return False
                    if (steps == 0) != (leaves == 0) or any(row[22:]):
                        return False
                    ticks = [row[4+2*i] | (row[5+2*i] << 32) for i in range(7)]
                    # leaf尾ACK在step外，不能误要求step累计覆盖全部ACK时间。
                    if ticks[0] < ticks[1] or ticks[0] < ticks[2]:
                        return False
                    if sum(ticks[3:6]) > ticks[1] or ticks[6] > sum(ticks[3:6]):
                        return False
                    if steps == 0 and any(ticks[1:]):
                        return False
        return True
    except (TypeError, ValueError, IndexError, OverflowError):
        return False


def export_clock(signal, destination, phases):
    """复用原阶段时钟导出，另存24核累计；不减nested wait、不生成性能结论。"""
    from compile_diagnostics import export_clock as phase_export
    slots = _slots(signal)
    generation = int(slots[0][0][0])
    if not check_probe_padding(slots, generation):
        raise ValueError("Score probe的padding/schema/epoch/动态计数合同未通过")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    report = phase_export(signal, destination, phases)
    rows = []
    for core in range(AIC_CORES):
        vals = [int(v) & 0xffffffff for v in slots[PHASE][core][32:64]]
        ticks = {name: vals[4+2*i] | (vals[5+2*i] << 32) for i,name in enumerate(TICK_NAMES)}
        rows.append({"core": core, "generation": generation, "leaf_count": vals[2], "step_count": vals[3],
                     "ack_count": vals[18], "sample_count": vals[19], "local_wait_count": vals[20],
                     "local_wait_sites_per_sample": vals[21], "ticks": ticks,
                     "interval_us": {name: value / 50 for name,value in ticks.items()}, "raw_words": vals})
    result = {"eligible_for_performance": False, "clock_freq_hz": 50000000, "schema": SCHEMA,
              "generation": generation, "sample_panel": SAMPLE_PANEL, "records": rows,
              "median_core_total_us": {name: statistics.median(r["interval_us"][name] for r in rows) for name in TICK_NAMES},
              "warning": "嵌套区间不可相加；j8不可乘16；Scalar原位区间含原阻塞与插桩扰动，不是异步纯硬件耗时。"}
    path = destination / "score_service_summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    report.update(score_service=str(path), score_records=24, score_padding_passed=True, eligible_for_performance=False)
    return report
