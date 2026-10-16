# SPDX-License-Identifier: Apache-2.0
"""仅诊断已有 B16 模板的进度协议，不执行数学、不替代完整 CSA 验证。"""

import argparse
import ast
import copy
import json
import os
import traceback
from pathlib import Path


def call_name(node):
    return ast.unparse(node.value.func) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) else None


def make_probe(case, output, max_polls):
    """从现有完整源码抽取原控制段；计算段按明确的 fence 边界移除并记录。"""
    source = ast.parse((case / 'entry_jit.py').read_text())
    kernels = []
    audit = {}
    for role in ('aic', 'aiv'):
        fn = copy.deepcopy(next(n for n in source.body if isinstance(n, ast.FunctionDef)
                                and n.name == 'csa_static_' + role))
        progress = next(p for p in fn.args.args if p.arg == 'progress')
        result_annotation = copy.deepcopy(progress.annotation.slice)
        fn.args.args = [p for p in fn.args.args if p.arg in ('progress', 'records', 'waits')]
        fn.returns = result_annotation
        fn.body[-1] = ast.Return(value=ast.Name(id='progress', ctx=ast.Load()))
        fn.body.insert(1, ast.parse('failed = 0').body[0])
        removed = []
        for stmt in fn.body:
            if not isinstance(stmt, ast.If) or not ast.unparse(stmt.test).startswith('mapped_block_'):
                continue
            # 原 wrapper：五个标量绑定、依赖循环、fence/cacheinvalid、真实核体、
            # fence/cacheinvalid/fence、完成发布。只认这套已核对的明确结构。
            assert len(stmt.body) > 12
            assert isinstance(stmt.body[5], ast.For)
            assert [call_name(n) for n in stmt.body[6:8]] == ['pl.system.fence', 'pl.system.cacheinvalid']
            assert [call_name(n) for n in stmt.body[-4:-1]] == ['pl.system.fence', 'pl.system.cacheinvalid', 'pl.system.fence']
            assert isinstance(stmt.body[-1], ast.If)
            number = int(stmt.test.left.id.removeprefix('mapped_block_'))
            removed.append({'batch': number, 'removed_top_level_statements': len(stmt.body[8:-4])})
            stmt.body = stmt.body[:8] + stmt.body[-4:]
            stmt.test = ast.BoolOp(op=ast.And(), values=[stmt.test, ast.parse('failed == 0', mode='eval').body])
            dependency = stmt.body[5]
            loop = next(n for n in dependency.body if isinstance(n, ast.For))
            before = ast.unparse(loop)
            assert f'pl.yield_(pl.tensor.read(progress, [flag_{number}]))' in before
            # 有限轮询只用于故障诊断，避免整卡无限等待。失败后不伪造完成信号；
            # 记录原等待对象，跳过该核后续工作，使 Host 能取回诊断字段。
            replacement = ast.parse(f'''
for (ready_{number}, polls_{number}) in pl.while_(init_values=(initial_{number}, 0)):
    pl.cond(ready_{number} < target_{number} and polls_{number} < {max_polls})
    pl.system.cacheinvalid(progress, [1], [flag_{number}])
    after_{number}, spin_count_{number} = pl.yield_(pl.tensor.read(progress, [flag_{number}]), polls_{number} + 1)
if after_{number} < target_{number}:
    failed = 1
    pl.tensor.write(progress, [own_flag + 8], pl.cast(1, pl.INT64))
    pl.tensor.write(progress, [own_flag + 9], pl.cast({number}, pl.INT64))
    pl.tensor.write(progress, [own_flag + 10], pl.cast(other_{number}, pl.INT64))
    pl.tensor.write(progress, [own_flag + 11], target_{number})
    pl.tensor.write(progress, [own_flag + 12], after_{number})
    pl.tensor.write(progress, [own_flag + 13], pl.cast(spin_count_{number}, pl.INT64))
    pl.system.cacheinvalid(progress, [8], [own_flag + 8])
    pl.system.fence()
''').body
            pos = dependency.body.index(loop)
            dependency.body[pos:pos + 1] = replacement
            dependency.body = [ast.If(test=ast.parse('failed == 0', mode='eval').body,
                                      body=dependency.body, orelse=[])]
            publish = stmt.body[-1]
            publish.test = ast.BoolOp(op=ast.And(), values=[publish.test, ast.parse('failed == 0', mode='eval').body])
        audit[role] = removed
        kernels.append(fn)
    params = ', '.join(f'{p.arg}: {ast.unparse(p.annotation)}' for p in kernels[0].args.args)
    annotation = ast.unparse(kernels[0].returns)
    wrappers = ast.parse(f'''
@pl.jit.inline
def csa_static_group({params}) -> {annotation}:
    with pl.cluster(name_hint='csa_static_protocol'):
        progress = csa_static_aic(progress, records, waits)
        progress = csa_static_aiv(progress, records, waits)
    return progress

@pl.jit
def entry({params}) -> {annotation}:
    with pl.spmd(24, sync_start=True):
        progress = csa_static_group(progress, records, waits)
    return progress
''').body
    # 去掉数学后 InCore 自动推断会把纯标量函数归到 AIV，不能据此测原 AIC。
    # 诊断使用 program 的显式 AIC/AIV 类型，不给 JIT 增加私有属性或虚假计算。
    class BindSelf(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id.startswith('csa_static_'):
                node.func = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                          attr=node.func.id, ctx=ast.Load())
            return node
    for role, fn in zip(('AIC', 'AIV'), kernels):
        fn.args.args.insert(0, ast.arg(arg='self'))
        fn.decorator_list = [ast.parse(f'pl.function(type=pl.FunctionType.{role})', mode='eval').body]
    group, entry = wrappers
    # 显式 Group 的两个成员就是同一 MIX 的两半，不再使用 Inline+cluster。
    group.body = group.body[0].body + [group.body[-1]]
    for kind, fn in (('Group', group), ('Orchestration', entry)):
        fn.args.args.insert(0, ast.arg(arg='self'))
        fn.decorator_list = [ast.parse(f'pl.function(type=pl.FunctionType.{kind})', mode='eval').body]
        BindSelf().visit(fn)
    entry.body.insert(0, ast.parse('pl.dump_tag(progress)').body[0])
    cls = ast.ClassDef(name='CSAStaticPrepared', bases=[], keywords=[],
                       body=kernels + wrappers, decorator_list=[ast.parse('pl.program', mode='eval').body])
    module = ast.Module(body=[ast.parse('import pypto.language as pl').body[0], cls], type_ignores=[])
    ast.fix_missing_locations(module)
    path = output / 'protocol_program.py'
    path.write_text('# 诊断抽取：仅进度协议，无 CSA 数学，不作为完整算子通过证据。\n' + ast.unparse(module) + '\n')
    report = {'source': str(case / 'entry_jit.py'), 'layout': str(case / 'layout.json'),
              'probe': str(path), 'removed_compute': audit, 'max_polls': max_polls,
              'retained': '原分核、records/waits、代际公式、cacheinvalid/fence、单24组MIX',
              'changes': '移除计算段，加入有限轮询和失败记录；仅三个协议参数，program 显式 AIC/AIV',
              'full_csa_validated': False}
    (output / 'extraction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-polls', type=int, default=200000)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--compiled', type=Path)
    args = parser.parse_args()
    if args.case.name != 'h131072_b16':
        raise ValueError('本轮只诊断已有 128K/B16')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.compiled is None:
        path = make_probe(args.case, args.output, args.max_polls)
    else:
        path = args.output / 'protocol_program.py'
        # 保留准备阶段的候选源码，不能在上卡时重新生成并覆盖 fence 等诊断变化。
        assert path.read_text() == (args.compiled.parent / 'source.py').read_text()
    if args.prepare_only:
        print(path)
        return
    if 'TASK_DEVICE' not in os.environ:
        raise RuntimeError('必须通过 task-submit 分配设备')

    import dsv4_csa_env
    dsv4_csa_env.activate()
    import torch
    from pypto.ir import CompiledProgram
    from pypto.runtime import RunConfig
    from test_static_jit_device import verify_isolation, progress_expectation

    report = {'scope': '128K/B16 完整控制表诊断，无数学计算', 'passed': False,
              'full_csa_validated': False, 'device_completion_passed': False, 'launches': []}
    try:
        report['environment'] = verify_isolation()
        device_id = int(os.environ['TASK_DEVICE'])
        if args.compiled is None:
            raise ValueError('先 CPU 编译，传入 --compiled；不占卡重复编译')
        compiled = CompiledProgram.from_dir(args.compiled)
        layout = json.loads((args.case / 'layout.json').read_text())
        workspace = layout['workspace']
        width = layout['protocol']['progress_words_per_core']
        assert width >= 16
        progress = torch.zeros(workspace['sections']['progress']['shape'], dtype=torch.int64)
        records = torch.tensor(workspace['records'], dtype=torch.int64)
        waits = torch.tensor(workspace['waits'], dtype=torch.int64)
        config = RunConfig(platform='a2a3', device_id=device_id, enable_dump_args=1,
                           save_kernels=True, save_kernels_dir=str(args.output / 'run'))
        for generation in (1, 2, 3):
            compiled(progress, records, waits, config=config)
            report['device_completion_passed'] = True
            observed = progress.cpu().view(-1, width)
            failures = []
            for core, row in enumerate(observed):
                if row[8].item():
                    failures.append({'core': core, 'batch': row[9].item(), 'waiting_core': row[10].item(),
                                     'target': row[11].item(), 'observed': row[12].item(), 'polls': row[13].item()})
            expected = progress_expectation(layout, generation)
            equal = torch.equal(observed[:, 0], expected)
            report['launches'].append({'generation': generation, 'progress': observed[:, 0].tolist(),
                                       'expected': expected.tolist(), 'failures': failures, 'passed': equal and not failures})
            if failures or not equal:
                raise AssertionError('原控制表的有限轮询诊断未通过，见逐核等待记录')
        report['passed'] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'traceback'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
