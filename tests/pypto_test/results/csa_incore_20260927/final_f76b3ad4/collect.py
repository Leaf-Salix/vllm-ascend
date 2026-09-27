"""汇总固定f76b3ad4七档，复用同源码两档计时/泳道，图文件集中下载。"""
import argparse
import importlib.util
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]
CASES = ((131072, 4), (131072, 8), (131072, 16), (8192, 16), (8192, 24), (8192, 32), (8192, 40))
REUSED = {(131072, 16), (8192, 16)}
REVISION = 'f76b3ad4'
CHECKOUT = '/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-cache-f76b3ad4'
JOBS = ['task_20260927_220827_47245426905', 'task_20260927_220827_4724979000', 'task_20260927_214806_30823321511']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--available', action='store_true', help='只读已齐的档位，不生成下载目录')
    args = parser.parse_args()
    common = load('timing', RESULTS / 'csa_native_cube_matrix_20260927/summarize.py')
    compare = load('worker', RESULTS / 'csa_scheduling_20260927/upstream_725/compare.py')
    baseline = json.loads((RESULTS / 'csa_incore_20260927/final_3d1f0f65/cases.json').read_text())
    previous = {(c['history'], c['batch']): c for c in baseline['cases']}
    direct = json.loads((RESULTS / 'csa_incore_20260927/final_05cb2758/cases.json').read_text())
    direct_previous = {(c['history'], c['batch']): c for c in direct['cases']}
    rows, artifacts = [], []
    for history, batch in CASES:
        key = f'h{history}_b{batch}'
        reuse = (history, batch) in REUSED
        source = RESULTS / 'csa_split_optimization_20260927/cache_tail_guard' / key if reuse else ROOT / key
        profile_root = ROOT / key / ('profile' if reuse else 'timing') / 'profile'
        if args.available and (not (source / 'swimlane/report.json').exists() or
                               not all(list((profile_root / side).glob('**/trace_view.json')) for side in ('native', 'pto'))):
            continue
        timing = common.summarize_timing(source / 'timing/report.json')
        raw_timing = json.loads((source / 'timing/report.json').read_text())['timing']
        tail = {side: {'p95_over_p50_pct': (raw_timing[side]['us_p95'] / raw_timing[side]['us_p50'] - 1) * 100,
                       'max_us': max(raw_timing[side]['samples_us']),
                       'samples_us': raw_timing[side]['samples_us']} for side in ('native', 'pto')}
        raw = json.loads((source / 'swimlane/report.json').read_text())
        windows = [compare.summarize(Path(w['merged_swimlane'])) for w in raw['swimlane_windows']]
        if len(windows) != 4:
            raise ValueError(f'Expected four independent windows: {key}')
        profiles, native = {}, defaultdict(list)
        for side in ('native', 'pto'):
            paths = list((profile_root / side).glob('**/trace_view.json'))
            if len(paths) != 1:
                raise ValueError(f'Expected one profile: {key}/{side}: {paths}')
            profiles[side] = str(paths[0])
            artifacts.append((paths[0], f'{key}_{side}_pytorch.json'))
            if side == 'native':
                data = json.loads(paths[0].read_text())
                events = data if isinstance(data, list) else data['traceEvents']
                pids = {e['pid'] for e in events if e.get('ph') == 'M' and e.get('name') == 'process_name' and e.get('args', {}).get('name') == 'Ascend Hardware'}
                for event in sorted(events, key=lambda e: float(e.get('ts', 0))):
                    if event.get('ph') == 'X' and event.get('pid') in pids:
                        native[event['name']].append(float(event['dur']))
        tasks = {}
        for name in sorted(set().union(*(w['tasks'] for w in windows))):
            measured = [w['tasks'][name] for w in windows if name in w['tasks']]
            values = [t['kernel_mean_us'] for t in measured]
            tasks[name] = {'blocks': [t['blocks'] for t in measured], 'window_block_means_us': values,
                           'mean_us': statistics.mean(values), 'min_window_mean_us': min(values), 'max_window_mean_us': max(values),
                           'worker_envelopes_us': [t['worker_envelope_us'] for t in measured],
                           'start_spreads_us': [t['start_spread_us'] for t in measured]}
        for i, w in enumerate(windows):
            artifacts.append((Path(w['path']), f'{key}_pto_swimlane_w{i}.json'))
        before = previous[(history, batch)]
        comparison = {'revision': '3d1f0f65', 'body_mean_us': before['timing']['body']['mean_us'],
                      'full_mean_us': before['timing']['pto_full']['mean_us'],
                      'body_p50_us': before['timing']['body']['p50_us'],
                      'full_change_pct': (timing['pto_full']['mean_us'] / before['timing']['pto_full']['mean_us'] - 1) * 100,
                      'body_change_us': timing['body']['mean_us'] - before['timing']['body']['mean_us'],
                      'body_change_pct': (timing['body']['mean_us'] / before['timing']['body']['mean_us'] - 1) * 100,
                      'hotspots': {}}
        for name in ('indexer_score_topk_native_pair_aic', 'indexer_score_topk_native_pair_aiv', 'indexer_topk_query_merge', 'qk_pv_aic', 'qk_pv_aiv', 'merge_norm'):
            old = before['pto_tasks'][name]['window_block_means_us']
            comparison['hotspots'][name] = {'old_window_means_us': old, 'old_mean_us': statistics.mean(old),
                                          'current_mean_us': tasks[name]['mean_us'],
                                          'change_pct': (tasks[name]['mean_us'] / statistics.mean(old) - 1) * 100}
        direct_t = direct_previous[(history, batch)]['timing']['pto_full']
        direct_comparison = {'revision': '05cb2758', 'full': direct_t,
                             'mean_change_pct': (timing['pto_full']['mean_us'] / direct_t['mean_us'] - 1) * 100,
                             'p95_change_pct': (timing['pto_full']['p95_us'] / direct_t['p95_us'] - 1) * 100}
        rows.append({'versus_05cb2758': direct_comparison, 'history': history, 'batch': batch, 'operator_revision': REVISION, 'reused_timing_swimlane': reuse,
                     'timing': timing, 'tail': tail, 'profiles': profiles, 'native_kernel_us': dict(native), 'pto_tasks': tasks,
                     'pto_windows': [{k: w[k] for k in ('path', 'worker_span_us', 'phases_us')} for w in windows],
                     'versus_3d1f0f65': comparison,
                     'swimlane_checks': {'guards': common.guard_failures(raw['pto_guards']),
                                         'nonfinite': {n: v['nonfinite'] for n, v in raw['pto_native'].items() if v.get('nonfinite')},
                                         'topk_structural_errors': raw['topk_selection']['structural_errors']}})
    if not args.available and len(rows) != len(CASES):
        raise ValueError('Incomplete seven-case matrix')
    manifest = []
    if not args.available:
        download = ROOT / 'download'
        download.mkdir(exist_ok=True)
        for source, name in artifacts:
            target = download / name
            shutil.copyfile(source, target)
            manifest.append({'file': name, 'source': str(source), 'bytes': target.stat().st_size})
        if len(manifest) != 42:
            raise ValueError(f'Expected 42 JSON artifacts, got {len(manifest)}')
        (download / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result = {'operator_revision': REVISION, 'source_checkout': CHECKOUT, 'task_ids': JOBS,
              'complete': len(rows) == len(CASES), 'cases': rows,
              'scope': '单卡layer4正式权重＋合成输入/历史；第二CSA metadata复用；TP1/S6/mode2/atomic1/deterministic0，无EPLB。每档5预热20无profiler样本，4个独立DFX窗口，两侧独立PyTorch profile。复用同源码两档计时/泳道，profile-only的一次计时不替代正式20次数据。',
              'limits': 'Native完整kernel与PTO逐block核内口径不同，不相加重建CSA，不直接计算等范围加速比。非当前16卡token/DSpark验收。'}
    (ROOT / ('partial.json' if args.available else 'cases.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    lines = [f'# 当前核内保留实现七档对照（{REVISION}）', '', result['scope'], '',
             '保留PTO内部直接读写Native物理页及N128流水；本轮仅给长档Score设置整组准入且禁止提前释放，短档标志保留。Native布局/流程不改。',
             '单位μs，每次HC_pre→norm→CSA→HC_post设备区间，20次无profiler均值。当前完整PTO＝本体，入口历史复制及外部写回已删除。',
             '旧七档为3d1f0f65，旧完整路径含入口拆分/外部写回，旧本体不含它们；本体增加与完整获益分别披露。两阶段数据不可直接相加。', '',
             '| 上下文/B | Native | 新完整PTO＝本体 | p50 / p95 | PTO对Native | 旧纯CSA | 本体增量 | 旧完整PTO | 完整变化 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        t = row['timing']; n = t['native']['mean_us']; b = t['body']
        old = row['versus_3d1f0f65']
        lines.append(f"| {row['history']//1024}K/{row['batch']} | {n:.2f} | {b['mean_us']:.2f} | {b['p50_us']:.2f} / {b['p95_us']:.2f} | {(b['mean_us']/n-1)*100:+.2f}% | {old['body_mean_us']:.2f} | {old['body_change_us']:+.2f} ({old['body_change_pct']:+.2f}%) | {old['full_mean_us']:.2f} | {old['full_change_pct']:+.2f}% |")
    lines += ['', '旧3d1f0f65的128K/B16本体均值含长尾，当前相对旧均值仅增加24.20μs不能解释成分页读取代价也只有24.20μs。',
              '同档本体p50为1177.61→1257.91μs（+80.30，+6.82%）；8K/B16为793.74→797.77（+4.03，+0.51%）。',
              '当前均值还受新增调度尾部保护影响，核内代价仍看下方任务数据；不把压低旧长尾冒认为读取成本消失。']
    lines += ['', '## 尾延迟', '', 'p95和最大值来自同一组20次无profiler计时；用于暴露尾部，不代表EP16尾延迟已经验收。', '',
              '| 档位 | PTO p95/p50增幅 | PTO最大值 | Native p95/p50增幅 | Native最大值 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        tails = row['tail']
        lines.append(f"| {row['history']//1024}K/{row['batch']} | {tails['pto']['p95_over_p50_pct']:.2f}% | {tails['pto']['max_us']:.2f} | {tails['native']['p95_over_p50_pct']:.2f}% | {tails['native']['max_us']:.2f} |")
    lines += ['', '128K/B16另有原版/新版本各100次的尾部确认，见[定向保护报告](../../csa_cache_tail_guard_20260927/README.md)。',
              '该确认与此表20次正式计时独立，不拼接样本；新策略是否仍有异常以本表全部p95/max和原始样本为准。',
              '', '## 相邻05cb2758直接读取基底', '',
              '| 档位 | 原完整PTO | 新完整PTO | 均值变化 | 原p95 | 新p95 | p95变化 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        old = row['versus_05cb2758']; cur = row['timing']['pto_full']
        lines.append(f"| {row['history']//1024}K/{row['batch']} | {old['full']['mean_us']:.2f} | {cur['mean_us']:.2f} | {old['mean_change_pct']:+.2f}% | {old['full']['p95_us']:.2f} | {cur['p95_us']:.2f} | {old['p95_change_pct']:+.2f}% |")
    lines += ['', '## Native融合kernel与PTO核内热点', '', 'PTO列为四窗口block均值范围；不能与Native完整kernel按等范围直接相减。', '',
              '| 档位 | Native QLI | PTO Score AIC | PTO Top-K merge | Native Sparse | PTO qk_pv AIC | PTO merge_norm |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def interval(task):
        return f"{task['min_window_mean_us']:.2f}–{task['max_window_mean_us']:.2f}"
    for row in rows:
        native = row['native_kernel_us']; task = row['pto_tasks']
        for name in ('VllmQuantLightningIndexer', 'SparseAttnSharedkv'):
            if len(native.get(name, [])) != 1:
                raise ValueError(f'Ambiguous Native hotspot: {name}')
        lines.append(f"| {row['history']//1024}K/{row['batch']} | {native['VllmQuantLightningIndexer'][0]:.2f} | {interval(task['indexer_score_topk_native_pair_aic'])} | {interval(task['indexer_topk_query_merge'])} | {native['SparseAttnSharedkv'][0]:.2f} | {interval(task['qk_pv_aic'])} | {interval(task['merge_norm'])} |")
    lines += ['', result['limits'], '保护区、Top-K结构、非有限值和Native零容差结果均保存在cases.json，不能把MEASURED当作精度验收通过。',
              '两档B16计时/泳道复用尾部保护先导的同一算子源码；提交前只补代码注释及文档，不改变核函数。两档profile在固定提交另采。',
              '[相邻bad985a9两档本体/完整路径对照](../../csa_cache_panel_read_20260927/README.md)单列；不要把它与旧七档3d1f0f65混为同一基底。',
              '[全部核内任务、原始路径、p95与检查](cases.json)；[42张JSON图集中下载目录](download/manifest.json)。',
              '[固定源码命令](run_case.sh)、[补五档](run_missing.sh)、[补两档profile](run_profiles.sh)、[只读汇总](collect.py)。']
    if not args.available:
        (ROOT / 'README.md').write_text('\n'.join(lines) + '\n')
        (ROOT / 'partial.json').unlink(missing_ok=True)
    for row in rows:
        print(row['history'], row['batch'], row['timing']['body'], row['timing']['guard_failures'], row['timing']['topk'])
    print(f'{len(rows)}/{len(CASES)} cases available; copied {len(manifest)} artifacts')

if __name__ == '__main__':
    main()
