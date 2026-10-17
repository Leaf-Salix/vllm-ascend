"""读取 d56 编排，枚举 phase 消费，并隔离删除两条 AIC orphan 发布。

不更改通用同步 helper，不叠加 HC guard，不编译或执行设备；不计算 hash。
"""

import ast
import difflib
import json
from pathlib import Path

E = Path(__file__).resolve().parent
D = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017')
REF = D / 'parallel_20261018/adopted_source_d56c9eb6'
TARGETS = {'PHASE_SCORE_CUBE', 'PHASE_ATTN_CUBE'}


def key(expr):
    if isinstance(expr, ast.BinOp) and isinstance(expr.left, ast.Name) and expr.left.id.endswith('_GROUP_BASE'):
        return expr.left.id + ' + group'
    return ast.unparse(expr)


def inventory():
    out = {}
    for name in ('cube_schedule.py', 'vector_schedule.py'):
        for node in ast.walk(ast.parse((REF / name).read_text())):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in ('publish_phase', 'wait_producer', 'wait_row_owners'):
                continue
            phase = key(node.args[1])
            row = out.setdefault(phase, {'publish': [], 'wait': []})
            kind = 'publish' if node.func.id == 'publish_phase' else 'wait'
            row[kind].append({'file': name, 'line': node.lineno, 'call': ast.unparse(node)})
    orphans = sorted(phase for phase,row in out.items() if row['publish'] and not row['wait'])
    assert set(orphans) == TARGETS | {'PHASE_GATES','PHASE_COMB','PHASE_IDX_COMMIT','PHASE_MAIN_COMMIT','PHASE_OUT'}
    report = {'reference': str(REF), 'phases': out, 'no_phase_wait': orphans,
              'normalization': '输出组发布/等待使用不同循环变量，按 PHASE_*_GROUP_BASE+group 对齐，不误判为无人消费。',
              'scope': '只统计两个正式 schedule 的 phase 调用；FFTS 与本核 payload 依赖另见 README，不因无 wait 自动删除 fence。'}
    (E/'phase_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


class DeleteTargetCalls(ast.NodeTransformer):
    def __init__(self):
        self.removed = []

    def visit_Assign(self, node):
        value = node.value
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='publish_phase':
            if isinstance(value.args[1],ast.Name) and value.args[1].id in TARGETS:
                self.removed.append(value.args[1].id)
                return None
        return self.generic_visit(node)


def main():
    inventory()
    files = sorted(REF.glob('*.py'))
    assert len(files)==33
    before = (REF/'cube_schedule.py').read_text()
    after = before
    comments = {
        'PHASE_SCORE_CUBE': 'Score每leaf已等双AIV ACK排空；后继plan/query获取保留，无需无人等待的SCORE_CUBE槽。',
        'PHASE_ATTN_CUBE': '后继ATTN_VECTOR获取包含PV消费、归并及packed发布；无需无人等待的ATTN_CUBE槽。',
    }
    for phase in TARGETS:
        line = f'    signal = publish_phase(signal, {phase}, worker, CORES, generation)'
        assert after.count(line)==1
        after = after.replace(line,'    # '+comments[phase])
    transform = DeleteTargetCalls()
    expected = transform.visit(ast.parse(before))
    assert set(transform.removed)==TARGETS and len(transform.removed)==2
    assert ast.dump(expected)==ast.dump(ast.parse(after))
    target = E/'source_candidate'
    target.mkdir(exist_ok=False)
    manifest=[]
    for p in files:
        text = after if p.name=='cube_schedule.py' else p.read_text()
        (target/p.name).write_text(text)
        manifest.append({'name':p.name,'bytes':(target/p.name).stat().st_size,'direct_text_same_as_reference':text==p.read_text()})
    assert sum(not row['direct_text_same_as_reference'] for row in manifest)==1
    (E/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (E/'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='a/cube_schedule.py',tofile='b/cube_schedule.py')))
    result={'passed':True,'reference':str(REF),'candidate':str(target),'changed_files':['cube_schedule.py'],
            'removed_calls':sorted(TARGETS),'all_remaining_AST_identical':True,'vector_source_text_identical':True,
            'begin_invocation_unchanged':True,'all_waits_unchanged':True,'all_math_unchanged':True,
            'phase_sync_and_signals_unchanged':True,'hc_guard_not_composed':True}
    (E/'source_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
