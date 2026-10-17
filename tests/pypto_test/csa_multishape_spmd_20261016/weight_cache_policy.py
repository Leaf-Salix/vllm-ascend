# SPDX-License-Identifier: Apache-2.0
"""恢复实验融合生成器过滤掉的只读权重 cache_policy 函数属性。

原 Tensor 数学体仍会经过 ConvertTensorToTileOps，因此属性必须跟随融合后
InCore 函数的参数编号重绑定。原来已经带 cache=1 的 KV Tile load 保持原样。
只读取冻结参考中的显式声明，不把 KV/state/临时区推断成可 BYPASS。
"""

import ast
from pathlib import Path


def restore_policies(source, schedule, catalog):
    logical = ast.parse(Path(catalog['logical_source_ir']).read_text())
    policies = {}
    for fn in ast.walk(logical):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for stmt in fn.body:
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and ast.unparse(stmt.value.func) == 'pl.func_attr'):
                attrs = ast.literal_eval(stmt.value.args[0])
                if attrs.get('cache_policy'):
                    policies[fn.name] = attrs['cache_policy']

    roots, declarations = set(), []
    for call in schedule['call_bindings'].values():
        for index, policy in policies.get(call['logical_function'], []):
            arg = call['arguments'][index]
            ref = arg['value']
            if (policy != 1 or not isinstance(ref, dict)
                    or not ref.get('allocation', '').startswith('argument:')
                    or 'pl.Out[' in arg['annotation'] or 'pl.InOut[' in arg['annotation']):
                raise ValueError('只允许恢复原数学体声明的只读根参数 BYPASS')
            root = ref['allocation'].removeprefix('argument:')
            roots.add(root)
            declarations.append({'ordinal': call['ordinal'], 'function': call['logical_function'],
                                 'original_parameter_index': index, 'root': root, 'policy': policy})
    if roots != {'wq_a', 'wq_b', 'wkv', 'wo_a'}:
        raise ValueError(f'参考声明的权重集合改变，需重新审计：{sorted(roots)}')
    # 整个融合调用的所有同根视图都必须只读。不要只检查声明所在的单一调用。
    for call in schedule['call_bindings'].values():
        for arg in call['arguments']:
            ref = arg['value']
            if isinstance(ref, dict) and ref.get('allocation') in {'argument:' + r for r in roots}:
                if 'pl.Out[' in arg['annotation'] or 'pl.InOut[' in arg['annotation']:
                    raise ValueError('同一权重在另一数学调用中被写入，不满足 BYPASS 合同')

    module = ast.parse(source)
    before = ast.dump(module, include_attributes=False)
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef)
              and n.name == 'csa_static_aic')
    attr_call = next(n.value for n in fn.body if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'pl.func_attr')
    attrs = ast.literal_eval(attr_call.args[0])
    original = attr_call.args[0]
    if 'cache_policy' in attrs:
        raise ValueError('融合函数已有缓存声明，拒绝隐式覆盖')
    pairs = []
    for root in sorted(roots):
        index, param = next((i, a) for i, a in enumerate(fn.args.args) if a.arg == 'arg_' + root)
        if 'pl.Out[' in ast.unparse(param.annotation) or 'pl.InOut[' in ast.unparse(param.annotation):
            raise ValueError('融合入口将只读权重标成可写')
        pairs.append((index, 1))
    attrs['cache_policy'] = pairs
    attr_call.args[0] = ast.parse(repr(attrs), mode='eval').body
    ast.fix_missing_locations(module)
    result = ast.unparse(module) + '\n'
    # 独立检查此次变换只增加函数属性；数学、信号发布和等待均逐 AST 相同。
    attr_call.args[0] = original
    if ast.dump(module, include_attributes=False) != before:
        raise AssertionError('缓存策略变换意外改变了函数属性以外的源码')
    audit = {'original_declarations': declarations, 'immutable_roots': sorted(roots),
             'fused_function': fn.name, 'fused_parameter_policies': pairs,
             'only_function_attribute_changed': True, 'schedule_and_math_unchanged': True,
             'reference_source': catalog['logical_source_ir'],
             'reason': '实验生成器过滤func_attr；Tensor数学体需要重绑定，现有Tile load显式cache不变',
             'compiler_and_runtime_modified': False, 'device_validated': False}
    return result, audit
