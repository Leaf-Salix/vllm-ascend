"""沿用主机/builder及既有标记间隙，只读本轮正式样本。"""

import json
import statistics

from collect_model import REVISION, ROOT, load_module


def export_host_entry_spread():
    """Host monotonic clocks retain persistent skew removed by device centering."""
    host = json.loads((ROOT / 'host.json').read_text())
    phases = json.loads((ROOT / 'phases.json').read_text())
    result = {'operator_revision': REVISION,
              'scope': '本机跨进程monotonic_ns；每步相对16rank中位数，不减各rank十步中位数。',
              'limits': '主机入口不是设备开始或通信到达；不从正式forward扣除差额，不单独证明CPU抢占。',
              'cases': []}
    lines = ['# 持续的主机入场偏移', '', result['scope'], '', result['limits'], '',
             '设备起始事件分析会消去稳定的跨设备时钟偏移，也会消去持续的rank迟到；本表补充后者。', '',
             '| 档位 | 侧 | 十步平均execute入口跨度ms | forward入口跨度ms | forward入口最大跨度ms |',
             '| --- | --- | ---: | ---: | ---: |']
    rank_lines = ['', '以下列出十步中位偏移≥1ms的rank，仅为诊断筛选；全部rank保留在JSON。', '',
                  '| 档位 | 侧/rank | 主机forward中位偏移ms | 设备forward中位ms | metadata墙钟/线程CPU中位ms |',
                  '| --- | --- | ---: | ---: | ---: |']
    for case in host['cases']:
        phase_case = next(c for c in phases['cases'] if (c['history'], c['batch']) ==
                          (case['history'], case['batch']))
        record = {'history': case['history'], 'batch': case['batch'], 'sides': {}}
        label = f"{case['history']//1024}K/B{case['batch']}"
        for side in ('native', 'pto'):
            ranks = case[side]
            step_ids = [s['step'] for s in ranks[0]['steps']]
            by_step = {step: [next(s for s in rank['steps'] if s['step'] == step) for rank in ranks]
                       for step in step_ids}
            spreads = {field: [(max(s[field] for s in values) - min(s[field] for s in values)) / 1e6
                              for values in by_step.values()]
                       for field in ('execute_entry_ns', 'forward_entry_ns', 'forward_submitted_ns')}
            side_data = {'steps': step_ids, 'spread_ms': spreads, 'ranks': []}
            for rank in ranks:
                offsets = [(s['forward_entry_ns'] - statistics.median(
                    v['forward_entry_ns'] for v in by_step[s['step']])) / 1e6 for s in rank['steps']]
                phase_rank = next(r for r in phase_case['sides'][side]['ranks'] if r['rank'] == rank['rank'])
                median_phases = {name: {key: statistics.median(s['phases'][name][key] for s in phase_rank['steps'])
                                       for key in ('wall_ms', 'thread_cpu_ms')}
                                 for name in phase_rank['steps'][0]['phases']}
                row = {'rank': rank['rank'], 'forward_entry_offset_ms': offsets,
                       'median_offset_ms': statistics.median(offsets),
                       'forward_us': [s['forward_us'] for s in rank['steps']], 'median_phases_ms': median_phases}
                side_data['ranks'].append(row)
                if row['median_offset_ms'] >= 1:
                    metadata = median_phases['attention_metadata']
                    rank_lines.append(f"| {label} | {side}/{rank['rank']} | {row['median_offset_ms']:.3f} | "
                                      f"{statistics.median(row['forward_us'])/1000:.3f} | "
                                      f"{metadata['wall_ms']:.3f}/{metadata['thread_cpu_ms']:.3f} |")
            record['sides'][side] = side_data
            lines.append(f"| {label} | {side} | {statistics.mean(spreads['execute_entry_ns']):.3f} | "
                         f"{statistics.mean(spreads['forward_entry_ns']):.3f} | "
                         f"{max(spreads['forward_entry_ns']):.3f} |")
        result['cases'].append(record)
    lines += rank_lines + ['', '父子builder区间嵌套，不相加。墙钟与线程CPU差距包含等待及调度，尚未区分具体阻塞调用。',
                           '[逐rank十步偏移、设备forward及分项](host_spread.json)、'
                           '[瞬时偏移](ARRIVAL.md)、[正式forward](model/RESULTS.md)。']
    (ROOT / 'HOST_SPREAD.md').write_text('\n'.join(lines) + '\n')
    formatter = load_module('spread_formatter', ROOT.parent / 'csa_ascendc_topk_hc_ep16_20260928/collect_model.py')
    (ROOT / 'host_spread.json').write_text(formatter.format_json(result) + '\n')


def main():
    phases = load_module('targeted_phases', ROOT.parent / 'csa_kv_k512_ep16_20260928/analyze.py')
    phases.ROOT = ROOT
    phases.PHASES.update({
        'execute_to_input_sync_gap': ('execute_entry', 'input_sync_begin'),
        'inputs_to_coordination_gap': ('inputs_end', 'batch_coordination_begin'),
        'coordination_to_metadata_gap': ('batch_coordination_end', 'attention_metadata_begin'),
        'metadata_to_preprocess_gap': ('attention_metadata_end', 'preprocess_begin'),
    })
    phases.main(attribution_note=f'本轮{REVISION}仅验收长B4/B8新预取及短B16控制；'
                '保留原始尾部，不扣除EP等待，不将跨轮差额归因于单项。')
    compact = load_module('targeted_tail', ROOT.parent / 'csa_ub_combined_20260928/analyze.py')
    compact.ROOT = ROOT
    compact.export_tail_evidence()
    export_host_entry_spread()


if __name__ == '__main__':
    main()
