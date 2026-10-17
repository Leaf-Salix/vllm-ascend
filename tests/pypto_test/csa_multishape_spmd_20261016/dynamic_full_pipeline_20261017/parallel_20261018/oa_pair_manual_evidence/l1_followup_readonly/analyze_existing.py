#!/usr/bin/env python3
"""只读复算既有 OA pair 同进程 L1；不编译、不加载设备、不修改输入证据。

输入是四个已完成作业的 result 与原始 incore 时钟。候选和参考每核仅有
整函数的一条记录，故只报告整核端点，不猜测内部 OA/quant/OB 阶段用时。
原分 task 的阶段包络单独列出，不能当成候选内部阶段的替代测量。
"""
import argparse
import json
import re
from pathlib import Path
from statistics import median


DEFAULT_ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018')
CASES = ('oa_pair_manual_h131072_b4_v1', 'oa_pair_manual_h131072_b8_v1',
         'oa_pair_manual_h131072_b16_v1', 'oa_pair_manual_h8192_b16_v1')


def number(x):
    return round(x, 6)


def read_json(path, inventory):
    inventory[str(path)] = path.stat().st_size
    return json.loads(path.read_text())


def stage_of(name):
    if name.startswith('proj_a_mm'):
        return 'OA'
    if name == 'quant' or name.startswith('quant_'):
        return 'O_quant'
    if name.startswith('_proj_b_mm_nz_kernel'):
        return 'OB'
    if name.startswith('proj_b_act_hc_post'):
        return 'O_dequant_HCpost'
    if name.startswith('qk_pv_aic'):
        return 'attention_AIC'
    if name.startswith('qk_pv_aiv'):
        return 'attention_AIV'
    if name.startswith('indexer_score_topk_native_pair'):
        return 'indexer_score_AIC' if '_aic_' in name else 'indexer_score_AIV'
    return None


def parse_window(window, inventory):
    raw_path, trace_path = Path(window['raw']), Path(window['trace'])
    raw, trace = read_json(raw_path, inventory), read_json(trace_path, inventory)
    records = raw['aicore_tasks']
    frequency = raw['metadata']['clock_freq_hz']
    unit = 1e6 / frequency
    origin = min(r[3] for r in records)
    latest = max(r[4] for r in records)
    events = [x for x in trace['traceEvents'] if x.get('cat') == 'event' and x.get('ph') == 'X']
    # MIX 的 AIC/AIV 共用 taskId，必须连同物理 core 取名，不能误把72条都归到AIV。
    task_names = {(x['args']['taskId'], int(re.search(r'CoreId:(\d+)', x['args']['event-hint']).group(1))):
                  x['name'].split('(')[0] for x in events}
    core_ids = sorted({r[0] for r in records})
    core_first = {c: min(r[3] for r in records if r[0] == c) for c in core_ids}
    summary = {
        'raw': str(raw_path), 'trace': str(trace_path), 'raw_field_semantics':
        ['physical_core', 'task_id', 'sequence', 'incore_start_ticks', 'incore_end_ticks', 'local_setup_ticks'],
        'clock_freq_hz': frequency, 'records': len(records),
        'task_count': len({r[1] for r in records}),
        'function_names': sorted(set(task_names.values())),
        'span_us': number((latest-origin)*unit),
        'start_spread_us': number((max(core_first.values())-origin)*unit),
        'last_core': max(records, key=lambda r: r[4])[0],
    }
    assert abs(summary['span_us'] - window['incore_span_us']) < 1e-6
    for role, selected in [('AIC', [r for r in records if r[0] < 24]),
                           ('AIV', [r for r in records if r[0] >= 24])]:
        endpoints = [(max(r[4] for r in selected if r[0] == c)-origin)*unit
                     for c in sorted({r[0] for r in selected})]
        summary[role] = {
            'last_end_us': number(max(endpoints)), 'first_end_us': number(min(endpoints)),
            'end_spread_us': number(max(endpoints)-min(endpoints)),
            'last_core': max(selected, key=lambda r: r[4])[0],
        }
    summary['AIV_after_last_AIC_us'] = number(summary['AIV']['last_end_us']-summary['AIC']['last_end_us'])
    if window['side'] != 'dynamic_pto':
        assert len(records) == 72 and summary['task_count'] == 1
        assert len(set(r[0] for r in records)) == 72
        summary['per_core'] = [dict(core=r[0], start_us=number((r[3]-origin)*unit),
                                    end_us=number((r[4]-origin)*unit), duration_us=number((r[4]-r[3])*unit))
                               for r in records]
    else:
        stages = {}
        for r in records:
            stage = stage_of(task_names[(r[1], r[0])])
            if stage:
                stages.setdefault(stage, []).append(r)
        summary['original_task_stage_envelopes'] = {
            s: dict(records=len(rs), task_count=len({r[1] for r in rs}),
                    first_start_us=number((min(r[3] for r in rs)-origin)*unit),
                    last_end_us=number((max(r[4] for r in rs)-origin)*unit),
                    envelope_us=number((max(r[4] for r in rs)-min(r[3] for r in rs))*unit),
                    median_record_us=number(median((r[4]-r[3])*unit for r in rs)),
                    maximum_record_us=number(max((r[4]-r[3])*unit for r in rs)))
            for s, rs in stages.items()
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('analysis.json'))
    args = parser.parse_args()
    inventory, cases = {}, {}
    for case in CASES:
        base = args.root / case
        result = read_json(base/'result.json', inventory)
        matrix = read_json(base/'source_matrix/result.json', inventory)
        assert result['phase_clock'] is False and matrix['passed'] is True
        windows = {side: [] for side in ('dynamic_pto', 'spmd_reference', 'spmd_candidate')}
        for w in matrix['windows']:
            windows[w['side']].append(parse_window(w, inventory))
        assert all(len(ws) == 5 for ws in windows.values())
        endpoint_medians = {}
        for side, ws in windows.items():
            endpoint_medians[side] = {
                'span_us': number(median(w['span_us'] for w in ws)),
                'AIC_last_end_us': number(median(w['AIC']['last_end_us'] for w in ws)),
                'AIV_after_last_AIC_us': number(median(w['AIV_after_last_AIC_us'] for w in ws)),
                'AIV_end_spread_us': number(median(w['AIV']['end_spread_us'] for w in ws)),
                'start_spread_us': number(median(w['start_spread_us'] for w in ws)),
            }
            assert endpoint_medians[side]['span_us'] == matrix['statistics_us'][side]['median_us']
        paired = [dict(window=i, span_candidate_minus_reference_us=number(c['span_us']-r['span_us']),
                       AIC_end_candidate_minus_reference_us=number(c['AIC']['last_end_us']-r['AIC']['last_end_us']),
                       AIV_tail_candidate_minus_reference_us=number(c['AIV_after_last_AIC_us']-r['AIV_after_last_AIC_us']))
                  for i, (r, c) in enumerate(zip(windows['spmd_reference'], windows['spmd_candidate']))]
        original = {}
        for stage in windows['dynamic_pto'][0]['original_task_stage_envelopes']:
            original[stage] = {key: number(median(w['original_task_stage_envelopes'][stage][key] for w in windows['dynamic_pto']))
                               for key in windows['dynamic_pto'][0]['original_task_stage_envelopes'][stage]}
        cases[case] = dict(history=result['history'], batch=result['batch'], query_tokens=result['query_tokens'],
                          tokens=result['batch']*result['query_tokens'], phase_clock=False,
                          policy=result['policy'], source_policies=matrix['policies'],
                          statistics_us=matrix['statistics_us'], endpoint_medians=endpoint_medians,
                          paired_windows=paired, original_task_stage_envelope_medians=original, windows=windows)
    out = dict(scope='只读既有四作业，每作业三方五窗；无新编译/设备。',
               limitations=['SPMD 每核仅整函数起止，不能恢复 OA/quant/OB/HCpost 内部边界。',
                            'AIV-after-AIC 是两角色最后结束之差，不是纯 HCpost 耗时。',
                            '原分 task 阶段包络可能相互重叠，不能求和，也不能替代 SPMD 阶段计时。',
                            '各侧先归一化自己的 incore 最早起点；排除 orchestrator 和 task local_setup。',
                            '不同 history 的两个作业并非同一次跨场景配对，仅各自 candidate/reference 配对有效。'],
               cases=cases, input_inventory=[dict(path=p, bytes=n) for p, n in sorted(inventory.items())])
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v['endpoint_medians'] for k, v in cases.items()}, indent=2))


if __name__ == '__main__':
    main()
