"""Collect one fixed-source seven-case baseline, reusing matching measurements."""
import importlib.util
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]
CASES = ((131072, 4), (131072, 8), (131072, 16), (8192, 16), (8192, 24), (8192, 32), (8192, 40))
REUSED = {(131072, 16), (8192, 16), (8192, 40)}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    common = load('timing_summary', RESULTS / 'csa_native_cube_matrix_20260927/summarize.py')
    compare = load('worker_summary', ROOT.parent / 'upstream_725/compare.py')
    rows, artifacts = [], []
    download = ROOT / 'download'
    download.mkdir(exist_ok=True)
    for history, batch in CASES:
        key = f'h{history}_b{batch}'
        reuse = (history, batch) in REUSED
        source = RESULTS / 'csa_split_optimization_20260927/round05_indexer_comp_overlap' / key if reuse else ROOT / key
        timing = common.summarize_timing(source / 'timing/report.json')
        report = json.loads((source / 'swimlane/report.json').read_text())
        windows = [compare.summarize(Path(w['merged_swimlane'])) for w in report['swimlane_windows']]
        if len(windows) != 4:
            raise ValueError(f'Expected four windows: {key}')
        profiles, native = {}, defaultdict(list)
        profile_root = ROOT / key / ('profile' if reuse else 'timing') / 'profile'
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
                for e in events:
                    if e.get('ph') == 'X' and e.get('pid') in pids:
                        native[e['name']].append(e['dur'])
        for i, w in enumerate(windows):
            artifacts.append((Path(w['path']), f'{key}_pto_swimlane_w{i}.json'))
        rows.append(dict(history=history, batch=batch, operator_revision='07365e52', reused_timing_swimlane=reuse,
                         timing=timing, profiles=profiles, native_kernel_us=dict(native), pto_windows=windows))
    manifest = []
    for source, name in artifacts:
        target = download / name
        shutil.copyfile(source, target)
        manifest.append(dict(file=name, source=str(source), bytes=target.stat().st_size))
    (download / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    result = dict(operator_revision='07365e52', source_checkout='/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-scheduling-07365e52',
                  task_ids=['task_20260927_190319_338416115614', 'task_20260927_190319_338387729413'],
                  scope='Seven single-card cases; 20 unprofiled samples, 4 separate DFX windows. Three matching-source measurements reused; their one-sample profile-only runs do not replace timing.',
                  cases=rows)
    (ROOT / 'cases.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    lines = ['# 追加五轮调度的固定七档基线（07365e52）', '',
             '单卡正式第4层权重+合成输入历史，第二CSA层metadata复用；TP1/S6/mode2/atomic1/deterministic0，无EPLB。',
             '本体含HC_pre→norm→CSA→HC_post；完整PTO另含入口拆分和更新后写回。单位μs。',
             '5预热/20无profiler样本，DFX独立采4窗口。复用同源码8K/B16/B40、128K/B16；其他四档新采。',
             '三档profile-only的单次计时不用于替换20次正式计时。所有七档均有本次Native/PTO PyTorch profile。', '',
             '| 上下文/B | Native | PTO本体 | 本体p50 / p95 | 本体对Native | 完整PTO |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        t = row['timing']; n=t['native']['mean_us']; b=t['body']
        lines.append(f"| {row['history']//1024}K/{row['batch']} | {n:.2f} | {b['mean_us']:.2f} | {b['p50_us']:.2f} / {b['p95_us']:.2f} | {(b['mean_us']/n-1)*100:+.2f}% | {t['pto_full']['mean_us']:.2f} |")
    lines += ['', '这是一套源码的基线，不拼历史各版本最优值；不是16卡整模型验收。Native零容差结果在cases.json原样记录。',
              '保护区、索引结构和非有限值检查在cases.json逐项列出；浮点差异不改名为逐bit通过。',
              '[全部数值、Native分项与PTO四窗口](cases.json)，[可一起下载的42张JSON图](download/manifest.json)。',
              'Native为整个融合kernel，PTO为逐block核内及调度窗口；不能把两者直接相减或将交叠任务求和。',
              '第11–15轮每轮按8K/B16与128K/B16共同判断，之后再恢复incore优化。']
    (ROOT / 'README.md').write_text('\n'.join(lines)+'\n')
    for row in rows:
        print(row['history'], row['batch'], row['timing']['body'], row['timing']['guard_failures'], row['timing']['topk'])


if __name__ == '__main__':
    main()
