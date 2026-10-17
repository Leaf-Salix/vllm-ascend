# SPDX-License-Identifier: Apache-2.0
"""d253完整动态CSA的测试专用Attention逐tick Scalar时钟。

不改高层源/ABI/原指令，只在已生成C++添加标记和独占padding写入。
每核最多60 tick，记录在signal[tick,physical_core,32:64]；真实通知在第0项。
没有新增设备fence。记录由后继原ATTN_CUBE/ATTN_VECTOR release统一写回。
"""

import contextlib
import json
import re
import statistics
from pathlib import Path

CAPACITY = 60
START = 32
WIDTH = 32
AIV = {
    'generation': 0, 'tick': 1, 'tick_begin': 2, 'gather_begin': 3,
    'ring_store_issue_end': 4, 'kv_ready_before': 5, 'kv_ready_after': 6,
    'score_wait_before': 7, 'score_wait_after': 8, 'softmax_body_begin': 9,
    'prob_ready_before': 10, 'prob_ready_after': 11,
    'merge_gate_begin': 12, 'pv_wait_before': 13, 'pv_wait_after': 14,
    'merge_end': 15, 'publish_begin': 16, 'publish_end': 17, 'tick_end': 18,
    'ring_join_before': 19, 'ring_join_after': 20,
    'tick_reuse_before': 21, 'tick_reuse_after': 22,
}
AIC = {
    'generation': 0, 'tick': 1, 'tick_begin': 2,
    'kv_wait_before': 3, 'kv_wait_after': 4, 'l1_load_issue_end': 5,
    'score_ready_before': 6, 'score_ready_after': 7,
    'prob_wait_before': 8, 'prob_wait_after': 9,
    'pv_ready_before': 10, 'pv_ready_after': 11, 'tick_end': 18,
}
TAG = re.compile(r'/\* CSA_ATTN_PROBE_BEGIN \*/.*?/\* CSA_ATTN_PROBE_END \*/', re.S)


def block(code):
    return '/* CSA_ATTN_PROBE_BEGIN */' + code + '/* CSA_ATTN_PROBE_END */'


def close_brace(text, pos):
    depth = 1
    for i in range(pos + 1, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if depth == 0:
            return i
    raise AssertionError('C++ brace is not closed')


def first_marker(text, name, start=0):
    m = re.search(r'(?m)^ *// pto: %' + name + r'_inline\d+__[^\n]+', text[start:])
    assert m, name
    return start + m.start()


def label_before(text, pos):
    return list(re.finditer(r'(?m)^label\d+:\n', text[:pos]))[-1]


def stamp(index):
    # 空asm仅约束C++编译重排，不生成设备pipe fence。
    return block(f'\n  __asm__ __volatile__("" ::: "memory");\n  csa_at[{index}] = (uint32_t)get_sys_cnt();\n  __asm__ __volatile__("" ::: "memory");\n')


def instrument_function(text, role):
    fields = AIV if role == 'vector_schedule' else AIC
    first_notify = re.search(r'pto::comm::TNOTIFY\((\w+), (\w+), (\w+)\);', text)
    assert first_notify
    own_phase0, generation = first_notify[1], first_notify[2]
    insertions, anchors = [], []

    def add(pos, code, name):
        insertions.append((pos, code))
        anchors.append({'name': name, 'function_line': text[:pos].count('\n') + 1,
                        'context': text[max(0, pos - 80):pos + 140]})

    def point(pos, name):
        add(pos, stamp(fields[name]), name)

    if role == 'vector_schedule':
        init = first_marker(text, 'm_idx')
        tail = first_marker(text, 'm_after', init)
        mov = re.search(r'\bTMOV\([^\n]+\);', text[tail:])
        assert mov
        after_mov = tail + mov.end()
        back = re.search(r'\n  (v\d+) = \(int64_t\).*?\n  (v\d+) = \1;\n  goto (label\d+);', text[after_mov:])
        assert back
        tick, header = back[2], back[3]
        header_pos = text.index('\n' + header + ':\n', init) + 1
        branch = re.search(r'if \([^\n]+\) \{\n    goto (label\d+);\n  \} else \{\n    goto (label\d+);', text[header_pos:])
        assert branch
        begin = text.index('\n' + branch[1] + ':\n', header_pos) + len(branch[1]) + 3
        end = after_mov
        # 三处动态FFTS wait：大小档互斥SCORE两处和PV一处。
        waits = list(re.finditer(r'(?m)^ *wait_flag_dev\([^\n]+\);', text[begin:end]))
        sends = list(re.finditer(r'(?m)^ *ffts_cross_core_sync\([^\n]+\);', text[begin:end]))
        assert len(waits) == 3 and len(sends) == 3
        for m, kind in zip(waits, ('score', 'score', 'pv')):
            point(begin + m.start(), kind + '_wait_before')
            point(begin + m.end(), kind + '_wait_after')
        for m, kind in zip(sends, ('kv', 'prob', 'kv')):
            point(begin + m.start(), kind + '_ready_before')
            point(begin + m.end(), kind + '_ready_after')
        gather = first_marker(text, 'qk_kv_half', begin)
        point(gather, 'gather_begin')
        ring = re.search(r'(?m)^ *TSTORE\([^\n]+\);', text[gather:begin + waits[0].start()])
        assert ring
        ring_pos = gather + ring.start()
        point(gather + ring.end(), 'ring_store_issue_end')
        # 只分离原块末MTE2/MTE3汇合；不对每个内部数学barrier都加时钟。
        join_label = label_before(text, ring_pos)
        join_waits = list(re.finditer(r'(?m)^ *wait_flag\([^\n]+\);', text[join_label.end():ring_pos]))
        assert join_waits
        point(join_label.end() + join_waits[0].start(), 'ring_join_before')
        point(join_label.end() + join_waits[-1].end(), 'ring_join_after')
        reuse_waits = list(re.finditer(r'(?m)^ *wait_flag\([^\n]+\);', text[begin:gather]))
        assert reuse_waits
        point(begin + reuse_waits[0].start(), 'tick_reuse_before')
        point(begin + reuse_waits[-1].end(), 'tick_reuse_after')
        point(first_marker(text, 'qk_reduce_tmp', gather), 'softmax_body_begin')
        pv_wait = begin + waits[-1].start()
        pub = first_marker(text, 'm_gather_tmp', pv_wait)
        pub_label = label_before(text, pub)
        preceding = label_before(text, pub_label.start())
        point(preceding.end(), 'merge_end')
        point(pub_label.end(), 'publish_begin')
        # 发布真分支末尾在reset_right的最后TEXPANDS之后、goto汇合之前。
        reset = first_marker(text, 'reset_right', pub)
        exp = re.search(r'\bTEXPANDS\([^\n]+\);', text[reset:])
        assert exp
        point(reset + exp.end(), 'publish_end')
        # PV门控前保留原PIPE_ALL/local等待，区间可能含它们。
        gate_start = begin + sends[-1].end()
        barriers = list(re.finditer(r'(?m)^ *pipe_barrier\(PIPE_ALL\);', text[gate_start:pv_wait]))
        assert len(barriers) == 1
        point(gate_start + barriers[0].start(), 'merge_gate_begin')
    else:
        choices = []
        for m in re.finditer(r'for \(int64_t (v\d+) = [^\n]+\) \{', text):
            last = close_brace(text, m.end() - 1)
            body = text[m.end():last]
            if body.count('wait_flag_dev(') == 2 and body.count('ffts_cross_core_sync(') == 2:
                choices.append((m, last))
        assert len(choices) == 1
        loop, end = choices[0]
        begin, tick = loop.end(), loop[1]
        waits = list(re.finditer(r'(?m)^ *wait_flag_dev\([^\n]+\);', text[begin:end]))
        sends = list(re.finditer(r'(?m)^ *ffts_cross_core_sync\([^\n]+\);', text[begin:end]))
        for m, kind in zip(waits, ('kv', 'prob')):
            point(begin + m.start(), kind + '_wait_before')
            point(begin + m.end(), kind + '_wait_after')
        for m, kind in zip(sends, ('score', 'pv')):
            point(begin + m.start(), kind + '_ready_before')
            point(begin + m.end(), kind + '_ready_after')
        offset = begin + waits[0].end()
        load = re.search(r'\bTLOAD\([^\n]+\);', text[offset:])
        assert load
        point(offset + load.end(), 'l1_load_issue_end')

    guard = '#if defined(__DAV_VEC__)\n' if role == 'vector_schedule' else '#if defined(__DAV_CUBE__)\n'
    assert text.count(guard) == 1
    add(text.index(guard) + len(guard), block('\n  uint32_t csa_at[32];\n'), 'declaration')
    add(begin, block('\n  for (int csa_i = 0; csa_i < 32; ++csa_i) csa_at[csa_i] = 0;\n'
                     f'  csa_at[0] = (uint32_t){generation}; csa_at[1] = (uint32_t){tick};\n')
        + stamp(fields['tick_begin']), 'tick_begin')
    add(end, stamp(fields['tick_end']) + block(
        f'\n  if ({tick} < 60) {{\n'
        f'    volatile __gm__ uint32_t* csa_dst = reinterpret_cast<volatile __gm__ uint32_t*>({own_phase0}.data()) + {tick} * 72 * 128 + 32;\n'
        '    for (int csa_i = 0; csa_i < 32; ++csa_i) csa_dst[csa_i] = csa_at[csa_i];\n'
        '  } else {\n'
        f'    reinterpret_cast<volatile __gm__ uint32_t*>({own_phase0}.data())[64] = (uint32_t){generation};\n'
        '  }\n'), 'tick_save')
    transformed = text
    for pos, code in sorted(insertions, key=lambda item: item[0], reverse=True):
        transformed = transformed[:pos] + code + transformed[pos:]
    assert TAG.sub('', transformed) == text, '删去插桩后原C++必须逐字恢复'
    return transformed, {'role': role, 'tick_variable': tick, 'phase0_tensor': own_phase0,
                         'generation_variable': generation, 'points': anchors,
                         'original_statements_direct_text_recovered': True}


def instrument_cpp(source):
    original = source
    audits = []
    # 倒序处理函数，保持前一个函数的偏移不受后一个修改影响。
    for role in ('vector_schedule', 'cube_schedule'):
        start = source.index('AICORE void ' + role + '(')
        opening = source.index('{', start)
        end = close_brace(source, opening) + 1
        changed, audit = instrument_function(source[start:end], role)
        source = source[:start] + changed + source[end:]
        audits.append(audit)
    assert TAG.sub('', source) == original
    sync = r'\b(?:pto::comm::TNOTIFY|pto::comm::TWAIT|wait_flag_dev|ffts_cross_core_sync|wait_flag|set_flag|pipe_barrier|dsb|dcci)\([^\n]+\);'
    assert re.findall(sync, source) == re.findall(sync, original)
    return source, {'functions': audits, 'original_source_recovered': True,
                    'original_sync_sequence_equal': True, 'extra_hardware_fences': 0,
                    'capacity': CAPACITY, 'signal_columns': [START, START + WIDTH - 1],
                    'overflow_column': 64, 'fields': {'aic': AIC, 'aiv': AIV}}


@contextlib.contextmanager
def compilation_probe(directory):
    from pypto.backend import pto_backend
    from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.wait_codegen import rewrite_wait_source
    from compile_diagnostics import instrument_clock

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    original = pto_backend._compile_pto_module

    def hooked(pto_code, unit_name, output_dir, *args, **kwargs):
        raw = original(pto_code, unit_name, output_dir, *args, **kwargs)
        if unit_name != 'dynamic_csa_full':
            return raw
        marked, audit = instrument_cpp(raw)
        marked, audit['wait_backoff'] = rewrite_wait_source(marked)
        marked, audit['phase_clock'] = instrument_clock(marked, wait_backoff=True)
        audit.update(original_cpp_bytes=len(raw.encode()), instrumented_cpp_bytes=len(marked.encode()),
                     raw_cpp_path=str(Path(output_dir) / (unit_name + '.cpp')),
                     note='只保存小型审计；原物与返回给编译器的C++由正常编译目录保留，不另复制大dump。')
        (directory / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
        return marked

    pto_backend._compile_pto_module = hooked
    try:
        yield
    finally:
        pto_backend._compile_pto_module = original


def export_clock(signal, destination, phases):
    from compile_diagnostics import export_clock as phase_export
    destination = Path(destination)
    report = phase_export(signal, destination, phases)
    flat = signal.tolist()
    rows = []
    for core in range(72):
        generation = flat[0][core][0]
        assert flat[0][core][64] == 0, '逐tick记录容量溢出；本轮诊断无效'
        fields = AIC if core < 24 else AIV
        core_ticks = []
        for tick in range(CAPACITY):
            values = [x & 0xffffffff for x in flat[tick][core][32:64]]
            if values[0] == generation:
                assert values[1] == tick and values[2] and values[18]
                core_ticks.append(tick)
                rows.append({'core': core, 'kind': 'aic' if core < 24 else 'aiv',
                             'tick': tick, 'generation': generation,
                             'ticks': {name: values[index] for name, index in fields.items()},
                             'raw_words': values})
        # 本设备入口只允许T=96；每组4 query，AIC多2拍、AIV多3拍排空。
        assert core_ticks == list(range(22 if core < 24 else 23)), (core, core_ticks)
    assert rows
    pairs = {
        'aiv': {
            'tick_span': ('tick_begin', 'tick_end'),
            'gather_ring_issue': ('gather_begin', 'ring_store_issue_end'),
            'ring_local_join': ('ring_join_before', 'ring_join_after'),
            'tick_reuse_waits': ('tick_reuse_before', 'tick_reuse_after'),
            'kv_notify_issue': ('kv_ready_before', 'kv_ready_after'),
            'score_ffts_wait': ('score_wait_before', 'score_wait_after'),
            'softmax_body_to_prob_issue': ('softmax_body_begin', 'prob_ready_after'),
            'prob_notify_issue': ('prob_ready_before', 'prob_ready_after'),
            'pv_gate_merge': ('merge_gate_begin', 'merge_end'),
            'pv_ffts_wait': ('pv_wait_before', 'pv_wait_after'),
            'final_publish_reset_issue': ('publish_begin', 'publish_end'),
        },
        'aic': {
            'tick_span': ('tick_begin', 'tick_end'),
            'kv_ffts_wait': ('kv_wait_before', 'kv_wait_after'),
            'kv_l1_load_issue': ('kv_wait_after', 'l1_load_issue_end'),
            'qk_after_wait_to_score_issue': ('kv_wait_after', 'score_ready_after'),
            'prob_ffts_wait': ('prob_wait_before', 'prob_wait_after'),
            'pv_after_wait_to_ready_issue': ('prob_wait_after', 'pv_ready_after'),
        },
    }
    for row in rows:
        times = row['ticks']
        row['interval_us'] = {}
        for name, (left, right) in pairs[row['kind']].items():
            if times[left] and times[right]:
                delta = (times[right] - times[left]) & 0xffffffff
                assert delta < 2**31, (row['core'], row['tick'], name, '时钟顺序错误')
                row['interval_us'][name] = delta / 50
    summary = {}
    for kind, interval_pairs in pairs.items():
        cores = range(24) if kind == 'aic' else range(24, 72)
        summary[kind] = {}
        for name in interval_pairs:
            totals = []
            for core in cores:
                values = [row['interval_us'][name] for row in rows if row['core'] == core and name in row['interval_us']]
                totals.append({'core': core, 'count': len(values), 'total_us': sum(values)})
            summary[kind][name] = {'per_core': totals, 'median_core_total_us': statistics.median(x['total_us'] for x in totals)}
    (destination / 'attention_ticks.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    (destination / 'attention_summary.json').write_text(json.dumps({
        'eligible_for_performance': False, 'clock_freq_hz': 50000000,
        'interval_pairs': pairs, 'summary': summary,
        'warning': '区间有嵌套，禁止相加冒充纯计算；tick_end之后的记录store/下一tick清零不含于tick_span，但仍扰动整条流水。所有区间保留原局部等待，issue端点不是DMA完成。',
    }, ensure_ascii=False, indent=2) + '\n')
    report.update(attention_ticks=str(destination / 'attention_ticks.json'), attention_tick_records=len(rows),
                  attention_summary=str(destination / 'attention_summary.json'),
                  capacity_per_physical_core=CAPACITY, eligible_for_performance=False,
                  timing_meaning='Scalar代码区间/原位等待；含现有依赖与插桩开销，不等于DMA完成或纯数学时间')
    return report
