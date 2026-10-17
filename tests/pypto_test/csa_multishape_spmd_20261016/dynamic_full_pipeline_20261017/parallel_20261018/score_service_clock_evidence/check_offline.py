# SPDX-License-Identifier: Apache-2.0
"""只读已有668 C++的hook审计和合成信号槽检查；不导入Torch/PyPTO或调用编译器。"""
import ast
import json
import re
import sys
from pathlib import Path

import score_service_clock as hook

ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
CANONICAL = ROOT / 'vllm-ascend-dsv4-pto-0251rc1'
DRIVER = CANONICAL / 'tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017'
BUILD = DRIVER / 'parallel_20261018/oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r'
HERE = Path(__file__).parent


def valid_slots(generation=3):
    slots = [[[0] * 128 for _ in range(72)] for _ in range(60)]
    for core in range(72):
        slots[0][core][0] = generation
        slots[0][core][16:20] = [1000 + core, 1005 + core, 0, 0]
    for core in range(24):
        steps = [0, 1, 2, 3, 7, 33][core % 6]
        row = slots[34][core]
        row[0] = generation
        row[16:20] = [100000 + core, 100005 + core, 0, 0]
        row[32:36] = [hook.SCHEMA, generation, (steps + 2) // 3, steps]
        ticks = [steps * 1000 + 999, steps * 1000, steps * 20,
                 steps * 50, steps * 10, steps * 20, steps * 15]
        for i, value in enumerate(ticks):
            row[36 + 2*i] = value & 0xffffffff
            row[37 + 2*i] = value >> 32
        row[50:54] = [steps, steps, steps * hook.LOCAL_WAIT_SITES, hook.LOCAL_WAIT_SITES]
    return slots


class ListSignal:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def main():
    ast.parse((HERE / 'score_service_clock.py').read_text())
    compile((HERE / 'score_service_clock.py').read_text(), 'score_service_clock.py', 'exec')
    compiled = []
    for kind, path in [('raw', BUILD / 'ptoas/dynamic_csa_full.cpp'),
                       ('wrapper', BUILD / 'kernels/aic/cube_schedule.cpp')]:
        raw = path.read_text()
        marked, audit = hook.instrument_cpp(raw)
        assert hook.TAG.sub('', marked) == raw
        # 独立核对：每个开始变量仅声明一次、减一次，原函数括号数不被插入改变。
        starts = re.findall(r'const uint64_t (csa_score_start_\d+) = get_sys_cnt\(\);', marked)
        ends = re.findall(r'get_sys_cnt\(\) - (csa_score_start_\d+);', marked)
        assert len(starts) == len(set(starts)) and sorted(starts) == sorted(ends)
        assert marked.count('{') == raw.count('{') and marked.count('}') == raw.count('}')
        assert hook._function(marked, 'vector_schedule')[2] == hook._function(raw, 'vector_schedule')[2]
        cube = hook._function(marked, 'cube_schedule')[2]
        active_start = cube.index('#if defined(__DAV_CUBE__)')
        active_end = cube.index('#endif', active_start)
        assert cube[:active_start].count('CSA_SCORE_PROBE_BEGIN') == 0
        assert cube[active_end:].count('CSA_SCORE_PROBE_BEGIN') == 0
        # 所有新增store必须位于一次dump，原release先于原TNOTIFY，不额外加维护。
        assert marked.count('volatile __gm__ uint32_t* csa_score_dst') == 1
        assert len(re.findall(r'csa_score_dst\[\d+\] = ', marked)) == 32
        dump = cube.index('volatile __gm__ uint32_t* csa_score_dst')
        original_release = cube.index('dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);', dump)
        notify = cube.index('pto::comm::TNOTIFY(', dump)
        assert dump < original_release < notify
        output = HERE / f'{kind}_hook_audit.json'
        output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
        compiled.append({'kind': kind, 'path': str(path), 'bytes': path.stat().st_size,
                         'starts': len(starts), 'local_wait_sites': audit['local_wait_sites_per_sample'],
                         'marked_bytes': len(marked.encode()), 'marked_cpp_written': False})
    assert compiled[0]['starts'] == compiled[1]['starts'] == 20

    # 只按地址集合核对768个写项，0项信号与16..19阶段时钟完全不相交。
    writes = {(34 * 72 + c) * 128 + i for c in range(24) for i in range(32, 64)}
    notify_words = {(p * 72 + c) * 128 for p in range(60) for c in range(72)}
    phase_words = {(p * 72 + c) * 128 + i for p in range(60) for c in range(72) for i in range(16, 20)}
    assert len(writes) == 768 and not writes & notify_words and not writes & phase_words
    slots = valid_slots()
    assert hook.check_probe_padding(ListSignal(slots), 3)
    failures = []
    # 各禁止区与schema/epoch/计数各取一个有界反例，不做设备或性能模型。
    for phase, core, column, value, reason in [
        (34, 0, 1, 1, 'signal guard'), (34, 0, 15, 1, 'signal guard end'),
        (34, 0, 20, 1, 'FFTS hook forbidden'), (34, 0, 31, 1, 'pre probe guard'),
        (34, 0, 64, 1, 'post probe guard'), (34, 0, 127, 1, 'slot end'),
        (34, 24, 32, 1, 'AIV cannot write'), (33, 0, 32, 1, 'other phase cannot write'),
        (34, 0, 32, 0, 'schema missing'), (34, 0, 33, 2, 'stale epoch'),
        (0, 4, 0, 2, 'BEGIN stale epoch'), (34, 4, 0, 2, 'phase stale epoch'),
        (34, 1, 50, 2, 'ACK count'), (34, 1, 51, 0, 'sample count'),
        (34, 1, 52, 9, 'local wait count'), (34, 1, 53, 11, 'wrong compiled sites'),
        (34, 1, 54, 1, 'reserved probe word'),
    ]:
        before = slots[phase][core][column]
        slots[phase][core][column] = value
        assert not hook.check_probe_padding(slots, 3), reason
        slots[phase][core][column] = before
        failures.append(reason)
    # ACK包含step之外的leaf drain，正确实现不能要求ACK <= step累计。
    row = slots[34][1]
    old = (row[36], row[38], row[40])
    row[36], row[38], row[40] = 10000, 100, 9000
    assert hook.check_probe_padding(slots, 3)
    row[36], row[38], row[40] = old
    sys.path.insert(0, str(DRIVER))
    exported = hook.export_clock(ListSignal(slots), HERE / 'synthetic_export', {0:'BEGIN', 34:'SCORE_CUBE'})
    assert exported['score_padding_passed'] and exported['eligible_for_performance'] is False
    summary = {'passed': True, 'new_pypto_compilation': False, 'device_execution': False,
               'python_ast_and_syntax': True, 'cpp_lexical_checks_only': True,
               'final_ccec_instruction_order_or_spill_proven': False,
               'inputs': compiled, 'unique_padding_words': 768,
               'synthetic_step_counts': [0,1,2,3,7,33], 'negative_cases': failures,
               'tail_ack_greater_than_step_case_passed': True,
               'synthetic_export_only': exported,
               'limitations': ['未运行C++编译器', '保留现有机器码不可读限制，不把它记作编译失败',
                               '合成信号不是设备计时或精度证据', 'runner本次A/B/A仅hidden变化']}
    (HERE / 'offline_checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('inputs','synthetic_export_only')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
