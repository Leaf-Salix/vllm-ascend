# SPDX-License-Identifier: Apache-2.0
"""让未选中的G6资源分支也静态合法，不向专用helper传G2资源参数。"""

import ast
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    base=HERE/'source_score_split_store';target=HERE/'source_score_split_store_v2'
    target.mkdir(exist_ok=False)
    for path in base.glob('*.py'):
        (target/path.name).write_bytes(path.read_bytes())
    path=target/'indexer_score_stages.py';text=path.read_text()
    start=text.index('        result = _indexer_score_cube_split_store(')
    end=text.index('\n        )',start)
    call=text[start:end]
    for name,value in [('query_group_size','6'),('qk_panel_cols','64'),('score_tile','1024'),('balance_leaves','True')]:
        old=f'            {name},\n';assert call.count(old)==1
        call=call.replace(old,f'            {value},  # 此helper固定的数学资源合同；不是T/B/历史实值。\n')
    text=text[:start]+call+text[end:]
    ast.parse(text);path.write_text(text)
    # 仅修改公共wrapper；原/候选完整算法函数和AIV均保持相同AST。
    def funcs(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    old,new=funcs((base/path.name).read_text()),funcs(text)
    assert old.keys()==new.keys()
    changed=[name for name in old if old[name]!=new[name]]
    assert changed==['indexer_score_cube_stage'],changed
    info=dict(changed=changed,resource_constants=dict(query_group_size=6,qk_panel_cols=64,score_tile=1024,balance_leaves=True),
              actual_t_b_history_unchanged=True,formal_changed=False,requires_enabled_cpp_comparison=True,
              note='未选分支必须类型/shape合法；新增固定的是原G6数学资源，不是输入动态维度')
    (HERE/'SCORE_SPLIT_LEGAL_PREPARE.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(info)


if __name__=='__main__':
    main()
