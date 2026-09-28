"""复用入场/GC诊断，不新增设备执行。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def export_tail_evidence():
    phases = json.loads((ROOT / 'phases.json').read_text())
    host_cases = {
        (case['history'], case['batch']): case
        for case in json.loads((ROOT / 'host.json').read_text())['cases']
    }
    cases = []
    for case in phases['cases']:
        host = host_cases[case['history'], case['batch']]
        for side, phase_side in case['sides'].items():
            tail_ranks = {tail['rank'] for tail in phase_side['entry_tails']}
            cases.append({
                'history': case['history'], 'batch': case['batch'], 'side': side,
                'measured_window_gc_counts': {
                    str(rank['rank']): len(rank['measured_window_gc']) for rank in host[side]
                },
                'entry_tails': phase_side['entry_tails'],
                'tail_rank_ten_step_phases': [
                    rank for rank in phase_side['ranks'] if rank['rank'] in tail_ranks
                ],
                'tail_rank_host': [rank for rank in host[side] if rank['rank'] in tail_ranks],
            })
    evidence = {
        'scope': '精简保存入场异常rank的全部十步标记和基线，不改变正式性能样本。',
        'limits': phases['limits'],
        'selection': '沿用相对入场>2ms诊断过滤；不作为性能验收阈值。',
        'sources': ['phases.json', 'host.json', 'arrival.json'],
        'cases': cases,
    }
    (ROOT / 'metadata_tail.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')


def main():
    source = ROOT.parent / 'csa_kv_k512_ep16_20260928/analyze.py'
    spec = importlib.util.spec_from_file_location('forward_host_analysis', source)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    analyzer.ROOT = ROOT
    analyzer.main(attribution_note=(
        '本轮冻结2d2f9ca0，验证Top-K UB与QR尾修正版组合；两侧预热计时事件，'
        '本会话候选编译在模型任务仍pending时完成，正式窗口未并行编译。'
        '不能把旧版本跨轮差额归因于任一单项。'
    ))
    export_tail_evidence()


if __name__ == '__main__':
    main()
