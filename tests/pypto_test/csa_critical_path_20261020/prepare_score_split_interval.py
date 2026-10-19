# SPDX-License-Identifier: Apache-2.0
"""按既有M16前段资源边界保留小T原Score，避免按B4等验收点加白名单。"""

import ast
import importlib.util
import json
from pathlib import Path
import sys

from prepare_qb_stream_reuse import wire

HERE=Path(__file__).resolve().parent
FLAG='SPLIT_SCORE_ROWS'


def main():
    # 复用已审阅的签名接线算法，但显式传本次flag，避免修改另一个准备脚本。
    import prepare_qb_stream_reuse as wiring
    previous_flag=wiring.FLAG
    wiring.FLAG=FLAG
    base=HERE/'source_score_split_store_v2';target=HERE/'source_score_split_interval'
    target.mkdir(exist_ok=False)
    for p in base.glob('*.py'):
        (target/p.name).write_bytes(p.read_bytes())
    try:
        for filename,defs,calls in [
            ('entry.py',('_csa_core_group','decode_csa_spmd'),('_csa_core_group','cube_schedule')),
            ('cube_schedule.py',('cube_schedule',),('indexer_score_cube_stage',)),
            ('indexer_score_stages.py',('indexer_score_cube_stage',),()),
        ]:
            path=target/filename;path.write_text(wire(path.read_text(),defs,calls))
    finally:
        wiring.FLAG=previous_flag
    path=target/'indexer_score_stages.py';text=path.read_text()
    old='    if balance_leaves and query_group_size == 6 and qk_panel_cols == 64 and score_tile == 1024:\n'
    assert text.count(old)==1
    text=text.replace(old,'    if SPLIT_SCORE_ROWS and balance_leaves and query_group_size == 6 and qk_panel_cols == 64 and score_tile == 1024:\n')
    path.write_text(text)
    path=target/'policy.py';text=path.read_text()
    for old,new in [
        ('SHORT_PREFETCH_HALF_SORT_LIMIT = 2048\n','SHORT_PREFETCH_HALF_SORT_LIMIT = 2048\n# 沿已有QA的小M16并行区间边界，两个完整M16行块起再启用Score新流水。\nSCORE_SPLIT_MIN_TOKENS = 32\n'),
        ('    plan_physical_rows: bool = False\n','    plan_physical_rows: bool = False\n    split_score_rows: bool = False\n'),
        ('            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,\n','            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,\n            "SPLIT_SCORE_ROWS": self.split_score_rows,\n'),
        ('        plan_physical_rows=tokens >= EARLY_GEOMETRY_MIN_TOKENS,\n','''        plan_physical_rows=tokens >= EARLY_GEOMETRY_MIN_TOKENS,
        # 小T保持原Score，中大T使用早FIX与单次WS写回。分界沿已有M16
        # 前段资源区间；依据完整实测取舍，不把回退唯一归因于QA或假定最优。
        # 实际有效query/长度/页表仍在device读取，不按B4/B8等测试点枚举。
        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,
'''),
    ]:
        assert text.count(old)==1,old[:100];text=text.replace(old,new)
    pos=text.index('"""',text.index('"""')+3)
    text=text[:pos]+'''SPLIT_SCORE_ROWS在既有长G6域、T>=32时选择双QK早FIX和每panel单次WS
写回。32沿QA小M16区间的已有资源边界，不是B4/B8白名单或理论最优声明。
首轮长B8/B16/B24/B28改善而B4回退，因此小域保留原Score；该最终区间
仍须检查关闭/开启生成代码、T30/T36边界及S1/S5，尚不预记全部采用。
'''+text[pos:]
    path.write_text(text)
    for p in target.glob('*.py'):ast.parse(p.read_text())
    spec=importlib.util.spec_from_file_location('split_interval_policy',target/'policy.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    choices={}
    for history,batch,query in [(131072,b,6) for b in [4,5,6,8,16,24,28]]+[(8192,b,6) for b in [16,32]]+[(131072,1,1),(131072,16,5)]:
        ctrl=mod.select_policy(tokens=batch*query,requests=batch,max_query_tokens=query,max_seq_len=history+query).constexprs()
        tag=f'h{history}_b{batch}_s{query}';choices[tag]=ctrl
        (HERE/f'score_interval_controls_{tag}.json').write_text(json.dumps(ctrl,indent=2)+'\n')
    (HERE/'SCORE_SPLIT_INTERVAL_PREPARE.json').write_text(json.dumps(dict(choices=choices,threshold=32,basis='既有M16前段资源边界',formal_changed=False,device_validated=False),ensure_ascii=False,indent=2)+'\n')
    print({k:v[FLAG] for k,v in choices.items()})


if __name__=='__main__':
    main()
