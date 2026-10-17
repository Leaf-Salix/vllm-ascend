"""仅解释真实编排 AST；通过后生成四处发布 guard 的隔离源码，不编译/上卡。

复用主 check_schedule 的数学别名、实际 HC 消费集合与输出组覆盖检查，
额外检查所有 wait 目标、保留事件逐核顺序、旧 signal 槽的跨调用存活。
这是有限组合/偏序模型，不模拟缓存、流水指令或数值精度。
"""

import ast
import builtins
import difflib
import json
import random
from pathlib import Path

E = Path(__file__).resolve().parent
C = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
D = C / 'tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017'
REFERENCE = D / 'parallel_20261018/adopted_source_d56c9eb6'
SOURCE = E / 'source_candidate'
TARGETS = {'PHASE_RAW': 16, 'PHASE_GATES': 8, 'PHASE_NORM': 8, 'PHASE_COMB': 8}
PHASE_IDS = {'PHASE_RAW': 3, 'PHASE_GATES': 4, 'PHASE_NORM': 6, 'PHASE_COMB': 5}
ID_TILES = {PHASE_IDS[name]: tile for name, tile in TARGETS.items()}


def proposal(text):
    """只包围四条既有 publish；math/wait 与 INPUT/代际保持原文。"""
    for phase, tile in TARGETS.items():
        indent = '        ' if phase == 'PHASE_RAW' else '    '
        line = f'{indent}signal = publish_phase(signal, {phase}, core, CORES, generation)'
        assert text.count(line) == 1
        note = (
            'RAW消费者只覆盖真实T的M16 owner；较大P16的原数学/尾写仍保留。'
            if phase == 'PHASE_RAW' else
            '只省略没有本阶段有效输出的M8角色通知；活跃角色发布与全部wait/math不变。'
        )
        text = text.replace(line, f'{indent}# {note}\n{indent}if worker < (tokens + {tile-1}) // {tile}:\n    {line}')
    text = text.replace(
        '    # 所有HC角色先发布COMB再进入Score的48核完成集合，所以post/comb此时已',
        '    # 所有有有效comb输出的HC角色先发布COMB，再进入未改动的Score全48核完成集合；post/comb此时已',
    )
    return text


class StripFourGuards(ast.NodeTransformer):
    def __init__(self):
        self.removed = []

    def visit_If(self, node):
        if len(node.body) == 1 and isinstance(node.body[0], ast.Assign):
            call = node.body[0].value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'publish_phase':
                phase = call.args[1].id
                if phase in TARGETS:
                    expected = f'worker < (tokens + {TARGETS[phase]-1}) // {TARGETS[phase]}'
                    assert ast.dump(node.test) == ast.dump(ast.parse(expected, mode='eval').body)
                    assert not node.orelse
                    self.removed.append(phase)
                    return node.body[0]
        return self.generic_visit(node)


class VirtualSource:
    """给既有模型提供内存中的源文本；通过前不落地候选包。"""
    def __init__(self, variant, texts, name=''):
        self.variant, self.texts, self.name = variant, texts, name

    def __truediv__(self, name):
        return VirtualSource(self.variant, self.texts, name)

    def read_text(self):
        return self.texts[self.name]

    def __str__(self):
        return f'{self.variant}/{self.name}'


def load_existing_model():
    """只增加模型事件输出及编译缓存，不更改其执行/依赖判断。"""
    path = D / 'check_schedule.py'
    text = path.read_text()
    old = '    waiting = {}\n'
    assert text.count(old) == 1
    text = text.replace(old, '    programs = {core: [] for core in generators}\n' + old)
    old = '                waiting[core] = next(generators[core])\n'
    assert text.count(old) == 1
    text = text.replace(old, old + '''                event_kind, event_value = waiting[core]
                if event_kind == "step" and event_value == "publish_phase":
                    programs[core].append(("publish", trace[-1][2]))
                elif event_kind == "step" and event_value == "begin_invocation":
                    programs[core].append(("begin", None))
                else:
                    programs[core].append((event_kind, event_value))
''')
    old = '        "T": tokens,\n'
    assert text.count(old) == 1
    text = text.replace(old, '        "programs": programs,\n        "completed": sorted(completed),\n' + old)
    ns = {'__file__': str(path), '__name__': 'readonly_schedule_model'}
    exec(compile(text, str(path), 'exec'), ns)
    cache = {}

    def cached_compile(tree, filename, mode):
        key = (filename, mode)
        if key not in cache:
            cache[key] = builtins.compile(tree, filename, mode)
        return cache[key]

    ns['compile'] = cached_compile
    return ns


def case_metadata(t, kind='regular'):
    if kind == 'dummy':
        return [0, 0, 0, 0], [0, 0, 0, 0]
    if kind == 'mixed':
        return [1, 4, 3, 2], [131073, 32767, 8191, 0]
    if kind == 'padding':
        return [2, 0, 7, 1], [8191, 0, 131071, 7]
    q = [6] * (t // 6) + ([t % 6] if t % 6 else [])
    return q, [131072] * len(q)


def check_pair(ns, texts, case):
    t, kind, seed, g, grouped, cols, row_deps, consumer = case
    q, h = case_metadata(t, kind)
    values = []
    for variant in ('baseline', 'candidate'):
        ns['PKG'] = VirtualSource(variant, texts[variant])
        values.append(ns['run_case'](t, q, h, seed, g, grouped, cols, row_deps, consumer))
    before, after = values
    removed = set(before['completed']) - set(after['completed'])
    expected = {
        (phase, 24 + w) for phase, tile in ID_TILES.items()
        if phase != 3 or not consumer
        for w in range(48) if w >= (t + tile - 1) // tile
    }
    assert removed == expected
    assert set(after['completed']) <= set(before['completed'])
    waited = set()
    for core in range(72):
        filtered = [event for event in before['programs'][core]
                    if not (event[0] == 'publish' and (event[1], core) in removed)]
        assert filtered == after['programs'][core], (case, core)
        for kind_event, value in after['programs'][core]:
            if kind_event == 'wait':
                waited.update(value)
    assert not removed & waited, (case, removed & waited)
    return dict(T=t, metadata=kind, query_group=g, seed=seed, grouped=grouped,
                oa_cols=cols, hc_row_dependencies=row_deps, consumer_reduce=consumer,
                removed_publications=len(removed), consumer_misses=0,
                math_and_wait_programs_identical=True), after['programs']


def replay_generations(ns, texts):
    """不清零信号槽；当前代必须等待当前活跃 producer，旧值不能误满足。"""
    scenarios = [(24, 'regular'), (384, 'regular'), (1, 'regular'), (769, 'regular'),
                 (16, 'padding'), (24, 'dummy'), (24, 'mixed'), (96, 'regular'),
                 (0, 'regular'), (192, 'regular'), (385, 'regular'), (24, 'regular')]
    slots = {(phase, core): 0 for phase in range(60) for core in range(72)}
    reports = []
    for generation, (t, metadata) in enumerate(scenarios, 1):
        _, programs = check_pair(ns, texts, (t, metadata, generation, 6, True, 128, True, False))
        old_slots = dict(slots)
        pc = {core: 0 for core in range(72)}
        pairs = {}
        rng = random.Random(generation)
        stale_wait_would_block = set()
        while any(pc[c] < len(programs[c]) for c in pc):
            progressed = False
            order = list(pc)
            rng.shuffle(order)
            for core in order:
                if pc[core] == len(programs[core]):
                    continue
                kind, value = programs[core][pc[core]]
                if kind == 'begin':
                    slots[0, core] += 1
                    assert slots[0, core] == generation
                elif kind == 'publish':
                    slots[value, core] = slots[0, core]
                elif kind == 'wait':
                    pending = [key for key in value if slots[key] < generation]
                    if pending:
                        stale_wait_would_block.update(key for key in pending if slots[key] > 0)
                        continue
                    assert all(slots[key] == generation for key in value)
                elif kind == 'pair':
                    pairs.setdefault(value, set()).add(core)
                    if len(pairs[value]) != 3:
                        continue
                pc[core] += 1
                progressed = True
            assert progressed, (generation, t, 'unexpected wait cycle')
        active, inactive, nonzero_stale = [], [], []
        for phase, tile in ID_TILES.items():
            for w in range(48):
                key = phase, 24 + w
                if w < (t + tile - 1) // tile:
                    assert slots[key] == generation
                    active.append(key)
                else:
                    assert slots[key] == old_slots[key]
                    inactive.append(key)
                    if slots[key] > 0:
                        nonzero_stale.append([*key, slots[key]])
        reports.append(dict(generation=generation, T=t, metadata=metadata,
                            phase0_all_72=generation, active_slots=len(active),
                            inactive_slots=len(inactive), nonzero_stale_slots=nonzero_stale,
                            old_values_rejected_before_new_publish=len(stale_wait_would_block),
                            slots_not_cleared=True))
    assert any(r['nonzero_stale_slots'] for r in reports)
    return reports


def raw_capacity_cases():
    """额外 P16 仍保留原数学；只证明有效 gates/comb 没有读到省略 RAW 槽。"""
    rows = []
    for t in (0, 1, 8, 9, 16, 17, 24, 96, 192, 383, 384, 385, 767, 768, 769):
        exact = (t + 15) // 16 * 16
        for p16 in sorted({exact, exact + 16, exact + 768, max(exact, 2048)}):
            used_raw_blocks = {row // 16 for row in range(t)}
            consumers = {block % 48 for block in used_raw_blocks}
            kept = {w for w in range(48) if w < (t + 15) // 16}
            assert consumers == kept
            math_owners = {block % 48 for block in range(p16 // 16)}
            assert kept <= math_owners
            rows.append(dict(T=t, P16=p16, raw_math_owners=len(math_owners),
                             raw_consumed_owners=len(consumers), retained_publishes=len(kept),
                             omitted_but_math_active=sorted(math_owners-kept)))
    return rows


def main():
    files = sorted(REFERENCE.glob('*.py'))
    assert len(files) == 33
    baseline = {p.name: p.read_text() for p in files}
    proposed = dict(baseline)
    proposed['vector_schedule.py'] = proposal(baseline['vector_schedule.py'])
    strip = StripFourGuards()
    stripped = strip.visit(ast.parse(proposed['vector_schedule.py']))
    assert set(strip.removed) == set(TARGETS) and len(strip.removed) == 4
    assert ast.dump(stripped) == ast.dump(ast.parse(baseline['vector_schedule.py']))
    ns = load_existing_model()
    texts = {'baseline': baseline, 'candidate': proposed}
    # 原 check_schedule.main 的全部 576 个组合原样保留。
    old_cases = [(0, 'regular')]
    old_cases += [(t, 'regular') for t in (1,7,8,9,23,24,25,48,96,144,168,169,192,376,377,384,385,511,512,513)]
    old_cases += [(16,'padding'), (24,'dummy'), (24,'mixed')]
    cases = [(t,k,seed,g,grouped,cols,False,False) for t,k in old_cases
             for g in (2,6) for seed in range(3) for grouped in (False,True) for cols in (128,256)]
    assert len(cases) == 576
    # 正式 HC_ROW_DEPENDENCIES=True，并额外覆盖 M8/M16、48-owner 饱和与多wave边界。
    more_t = (0,1,7,8,9,15,16,17,23,24,25,31,32,33,47,48,49,95,96,97,127,128,129,143,144,145,167,168,169,191,192,193,375,376,377,383,384,385,767,768,769,1536,1537)
    cases += [(t,'regular',seed,g,True,128,True,False) for t in more_t for g in (2,6) for seed in (0,1)]
    cases += [(t,k,0,g,True,256,row_deps,consumer) for t,k in ((16,'padding'),(24,'dummy'),(24,'mixed'),(385,'regular'))
              for g in (2,6) for row_deps in (False,True) for consumer in (False,True)]
    rows = [check_pair(ns,texts,case)[0] for case in cases]
    generations = replay_generations(ns,texts)
    capacity = raw_capacity_cases()
    seven = []
    for history,batch in ((131072,4),(131072,8),(131072,16),(131072,24),(131072,28),(8192,16),(8192,32)):
        t = batch * 6
        by_phase = {name:48-min(48,(t+tile-1)//tile) for name,tile in TARGETS.items()}
        seven.append(dict(history=history,batch=batch,tokens=t,omitted_by_phase=by_phase,
                          omitted_total=sum(by_phase.values()),baseline_four_publishes=192,
                          candidate_four_publishes=192-sum(by_phase.values())))
    result = dict(passed=True, reference=str(REFERENCE), candidate=str(SOURCE),
                  changed_files=['vector_schedule.py'], added_guards=4,
                  normalized_AST_identical=True, original_schedule_cases=576,
                  total_paired_cases=len(rows), rows=rows, generations=generations,
                  raw_capacity_cases=capacity, seven_cases=seven,
                  hardware_cache_validated=False, numerical_accuracy_validated=False,
                  no_compile_or_device_execution=True)
    (E/'audit_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    SOURCE.mkdir(exist_ok=False)
    manifest=[]
    for name,text in proposed.items():
        (SOURCE/name).write_text(text)
        manifest.append(dict(name=name,bytes=(SOURCE/name).stat().st_size,
                             direct_text_same_as_reference=text==baseline[name]))
    (E/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (E/'candidate.patch').write_text(''.join(difflib.unified_diff(
        baseline['vector_schedule.py'].splitlines(keepends=True), proposed['vector_schedule.py'].splitlines(keepends=True),
        fromfile='a/vector_schedule.py',tofile='b/vector_schedule.py')))
    print(json.dumps({k:result[k] for k in ('passed','changed_files','added_guards','normalized_AST_identical','original_schedule_cases','total_paired_cases')},ensure_ascii=False))
    print(json.dumps(dict(cross_generation_calls=len(generations),raw_capacity_cases=len(capacity),seven_cases=seven),ensure_ascii=False))


if __name__ == '__main__':
    main()
