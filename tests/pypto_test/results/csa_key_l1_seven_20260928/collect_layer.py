"""复用已通过B16，补齐同一CSA实现七档单卡证据，不重复设备测试。"""

import json
import statistics
import subprocess
from pathlib import Path

from collect_model import CASES, REVISION, ROOT, load_module, weighted_changes

REMAINING_TASK = 'task_20260928_181207_305949327631'
REUSED_TASKS = ('task_20260928_175014_22857933418', 'task_20260928_180158_266927127827')
STATE_NAMES = {'x_out', 'idx_topk', 'swa.0', 'compressed.0', 'state.0',
               'indexer.0', 'indexer.1', 'indexer_state.0'}


def read(path):
    return json.loads(path.read_text())


def paths(history, batch):
    if batch == 16:
        base = ROOT.parent / 'csa_score_key_l1_20260928' / f'h{history}_b16'
        return base / 'candidate/report.json', base / 'swimlane/candidate/report.json'
    base = ROOT / 'layer' / f'h{history}_b{batch}'
    return base / 'report.json', base / 'swimlane/report.json'


def require_pass(values, label):
    if not values or any(value['status'] != 'PASS' for value in values.values()):
        raise ValueError(f'{label}: 自重放或metadata/保护区失败')


def check_config(report, history, batch):
    expected = {'history': history, 'batch': batch, 'seed': 1024, 'layer_index': 4,
                'variant': 'performance', 'weight_nz_mode': 2, 'effective_weight_nz_mode': 2,
                'deterministic_level': 0, 'hccl_deterministic': False,
                'checkpoint': '/data/model/DeepSeek-V4-Flash-0731-w8a8',
                'pto_reduction': {'atomic_add': 0, 'qr_split_k': 1, 'kv_split_k': 1}}
    for field, value in expected.items():
        if report[field] != value:
            raise ValueError(f'H{history}/B{batch}: 配置{field}不符')
    if report['status'] != 'MEASURED' or report['accuracy_fixture']['kind'] != 'per_physical_row':
        raise ValueError('报告未完成或未使用逐物理行scale')
    if set(report['pto_self']) != STATE_NAMES:
        raise ValueError('PTO八类自重放状态不完整')
    require_pass(report['pto_self'], 'PTO eager自重放')
    for group in (*report['native_guards'], *report['pto_guards']):
        require_pass(group, 'eager保护区')
    if report['topk_selection']['structural_errors']:
        raise ValueError('Top-K结构错误')
    if any(v.get('nonfinite', 0) for v in report['pto_native'].values()):
        raise ValueError('Native/PTO输出或状态出现非有限值')
    return {**expected, 'accuracy_fixture': report['accuracy_fixture']}


def collect(history, batch, timing_helper, worker):
    timing_path, swimlane_path = paths(history, batch)
    report, swimlane = read(timing_path), read(swimlane_path)
    config = check_config(report, history, batch)
    if check_config(swimlane, history, batch) != config:
        raise ValueError('独立DFX配置与正式计时不一致')
    timing = report['timing']
    if (timing['status'], timing['iters'], timing['warmup'], timing['compact_metadata_policy']) != (
        'MEASURED', 20, 5, 'reuse'
    ):
        raise ValueError('单层计时合同不符')
    for side in ('native', 'pto'):
        values = timing[side]
        if len(values['samples_us']) != 20 or len(set(values['start_timestamps_raw'])) != 20:
            raise ValueError('计时样本或事件更新时间不足20次')
        if any(v <= 0 for v in values['samples_us']):
            raise ValueError('计时区间非正')
        require_pass(values['guards'], side + '计时图保护区')
        if values['topk_selection']['structural_errors']:
            raise ValueError(side + '计时图Top-K结构错误')
    require_pass(timing['pto']['eager_comparison'], 'PTO计时图状态')
    graph = report['graph']
    if graph['status'] != 'PASS' or [r['input'] for r in graph['replays']] != ['A', 'B', 'A']:
        raise ValueError('A→B→A图重放未通过')
    windows = swimlane['swimlane_windows']
    if len(windows) != (4 if batch == 16 else 2):
        raise ValueError('DFX窗口数不完整')
    for window in windows:
        if not window['exported'] or (window['layer_index'], window['compact_metadata_policy'],
                                     window['input_source'], window['execution']) != (
            4, 'reuse', 'formal_layer_weights_synthetic_history', 'graph_replay'
        ):
            raise ValueError('DFX窗口采集合同不符')
    if batch == 16:
        reused = read(timing_path.parents[1] / 'evidence.json')
        if reused['status'] != 'PASS' or reused['errors']:
            raise ValueError('复用B16跨版本状态证据未通过')
    stats = {side: timing_helper.stats(timing[side]['samples_us']) for side in ('native', 'pto')}
    diagnostic_fields = ('status', 'dtype', 'atol', 'rtol', 'elements', 'nonfinite',
                         'max_abs', 'rmse', 'mismatches', 'reason')
    return {'history': history, 'batch': batch, 'config': config, 'status': 'PASS',
            'sources': {'timing': str(timing_path), 'swimlane': str(swimlane_path),
                        'tasks': list(REUSED_TASKS) if batch == 16 else [REMAINING_TASK],
                        'reuse': '已通过B16，同一生产算子，原任务退出1及缺失DFX补采保留' if batch == 16 else None},
            'device': timing['device'], 'timing': stats,
            'change_pct': (stats['pto']['mean_us'] / stats['native']['mean_us'] - 1) * 100,
            'checks': {'pto_self': 'PASS', 'timing_pto_eager': 'PASS', 'graph_A_B_A': 'PASS',
                       'metadata_guards': 'PASS', 'topk_structure': 'PASS'},
            'native_arithmetic_diagnostics': {
                name: {k: v for k, v in comparison.items() if k in diagnostic_fields}
                for name, comparison in report['pto_native'].items()},
            'topk_selection': report['topk_selection'],
            'worker_windows': [worker.summarize(Path(w['merged_swimlane'])) for w in windows],
            'scheduler_windows': [timing_helper.scheduling(Path(w['merged_swimlane']), worker.canonical)
                                  for w in windows]}


def main():
    status = subprocess.check_output(['task-submit', '--status', REMAINING_TASK], text=True).strip()
    if status != 'completed (exit=0)':
        raise SystemExit(f'{REMAINING_TASK}: {status}; 等待设备任务结束再解析')
    helper = load_module('layer_evidence', ROOT.parent / 'csa_short_score_sync_20260928/summarize.py')
    worker = load_module('worker_summary', ROOT.parent / 'csa_scheduling_20260927/upstream_725/compare.py')
    cases = [collect(history, batch, helper, worker) for history, batch in CASES]
    weighted = weighted_changes(cases)
    result = {'operator': REVISION, 'all_cases_pass': True, 'cases': cases,
              'performance_observations': weighted,
              'scope': '单卡layer4真实权重/合成历史，20次无profiler图重放；第二CSA层metadata复用。'
                       'PASS仅指自重放、图状态/保护区与结构，不表示Native逐bit或最终token/DSpark通过。'
                       'DFX独立采集，核内含等待；setup不是纯调度开销。'}
    formatter = load_module('compact_evidence', ROOT.parent / 'csa_ascendc_topk_hc_ep16_20260928/collect_model.py')
    (ROOT / 'layer.json').write_text(formatter.format_json(result) + '\n')
    lines = ['# 554b3bca七档单卡CSA', '', result['scope'], '',
             '长短B16复用同一实现的已通过样本；其他五档为阶段补测。所有20次原始样本及DFX窗口保留。',
             '以下比较各档同轮Native，不是Key L1单因素收益；不同档位可能由队列分配到不同卡。', '',
             '| 档位 | Native均值 μs | PTO均值 μs | 变化 | Native/PTO P95 μs | Native/PTO最大 μs |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in cases:
        native, pto = (row['timing'][s] for s in ('native', 'pto'))
        lines.append(f"| {row['history']//1024}K/B{row['batch']} | {native['mean_us']:.2f} | "
                     f"{pto['mean_us']:.2f} | {row['change_pct']:+.2f}% | "
                     f"{native['p95_us']:.2f}/{pto['p95_us']:.2f} | "
                     f"{native['max_us']:.2f}/{pto['max_us']:.2f} |")
    lines += ['', f"各history内等权后，128K/8K七三变化为{weighted['weighted_change_pct']:+.3f}%。",
              'P95/最大值逐档检查，不由综合均值抵消；新七档EP16尚须独立验收。', '',
              '## Score与Sparse的独立DFX', '',
              '核内均值为全部窗口等权；窗口范围全部保留，不与正式计时逐样本对应。', '',
              '| 档位 | 任务 | 窗口数 | 核内均值 μs | 各窗口核内均值 μs | 启动散布 μs |',
              '| --- | --- | ---: | ---: | --- | --- |']
    for row in cases:
        for task in ('indexer_score_topk_native_pair_aic', 'indexer_score_topk_native_pair_aiv',
                     'indexer_topk_query_merge', 'qk_pv_aic', 'qk_pv_aiv', 'merge_norm'):
            records = [w['tasks'][task] for w in row['worker_windows']]
            values = [r['kernel_mean_us'] for r in records]
            spreads = [r['start_spread_us'] for r in records]
            lines.append(f"| {row['history']//1024}K/B{row['batch']} | {task} | {len(values)} | "
                         f"{statistics.mean(values):.2f} | " + '/'.join(f'{v:.2f}' for v in values) +
                         ' | ' + '/'.join(f'{v:.2f}' for v in spreads) + ' |')
    lines += ['', '## 同核复用观测', '',
              '下面只列Score/Sparse中同一核执行多份的窗口；原始全部18窗口都在JSON中。',
              '这是独立DFX中的分派现象，不能与无profiler的P95或模型某一步建立一对一因果关系。', '',
              '| 档位 | 窗口 | 任务 | AIC任务数/使用核数 | 单核最多任务 |',
              '| --- | ---: | --- | --- | ---: |']
    for row in cases:
        for index, window in enumerate(row['scheduler_windows']):
            for label in ('score_assignment', 'sparse_assignment'):
                aic = window[label]['aic']
                if aic['max_blocks_per_core'] > 1:
                    lines.append(f"| {row['history']//1024}K/B{row['batch']} | {index} | {label} | "
                                 f"{aic['blocks']}/{aic['used_cores']} | {aic['max_blocks_per_core']} |")
    lines += ['', 'Native零容差算术/Top-K差异单独保留，不与PTO自重放PASS混用。',
              '[全部原始计时、状态诊断、核内与调度记录](layer.json)、[阶段合同](README.md)。']
    (ROOT / 'LAYER.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'all_cases_pass': True, **weighted}, ensure_ascii=False))


if __name__ == '__main__':
    main()
