#!/usr/bin/env python3
"""冻结 d253 正式 33 源，仅删除 O_A 量化同一输入的第二次 GM 读取。

原值仍由第一次 pl.load 提供。延长其到 row_expand_mul 的生命周期，要求
编译器为 abs 结果安排不同存储，不能强制原地覆盖。该脚本只操作私有副本；
不导入设备、不编译、不改正式源码，已存在的输出目录拒绝覆盖。
"""

import ast
import copy
import difflib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
C = ROOT / 'vllm-ascend-dsv4-pto-0251rc1'
FORMAL = C / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'


def main():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C, text=True).strip()
    assert commit.startswith('d253d627'), commit
    files = sorted(FORMAL.glob('*.py'))
    assert len(files) == 33
    before, after = OUT/'source_before', OUT/'source_candidate'
    before.mkdir(exist_ok=False)
    after.mkdir(exist_ok=False)
    for path in files:
        shutil.copy2(path, before/path.name)
        shutil.copy2(path, after/path.name)
    name = 'output_projection_a_stages.py'
    old = (before/name).read_text()
    target = '            oc_q = pl.load(o_r_pad, [qt, col_g], [QUANT_TOKEN_TILE, O_LORA], valid_shape=[rows, O_LORA])'
    assert old.count(target) == 1
    new = old.replace(target, '''            # 第一次加载的带符号 FP32 原值在本 qt 内继续存活，供量化乘法复用。
            # abs/row_max 的结果不得覆盖该原值；依赖正常 Tile 生命周期分配，
            # 不指定地址、不添加复制或额外常驻 Tile。尾行有效形状沿用原 load。
            oc_q = oc_amax''')
    old_ast, new_ast = ast.parse(old), ast.parse(new)
    old_assignment = next(n for n in ast.walk(old_ast) if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id=='oc_q' for t in n.targets))
    new_assignment = next(n for n in ast.walk(new_ast) if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id=='oc_q' for t in n.targets))
    assert ast.dump(new_assignment.value) == ast.dump(ast.Name(id='oc_amax',ctx=ast.Load()))
    # 将唯一替换归一化后整个模块 AST 必须一致，包括全部有效行、零写和组边界。
    restored = copy.deepcopy(new_ast)
    restored_assignment = next(n for n in ast.walk(restored) if isinstance(n,ast.Assign)
                               and any(isinstance(t,ast.Name) and t.id=='oc_q' for t in n.targets))
    restored_assignment.value = copy.deepcopy(old_assignment.value)
    assert ast.dump(restored,include_attributes=False) == ast.dump(old_ast,include_attributes=False)
    (after/name).write_text(new)
    unchanged=[p.name for p in files if p.name!=name and p.read_bytes()==(after/p.name).read_bytes()]
    assert len(unchanged)==32
    (OUT/'candidate.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
        fromfile='d253/'+name,tofile='candidate/'+name)))
    (OUT/'controls.json').write_text(json.dumps(dict(OA_COLS=128,OB_ROWS=96),indent=2)+'\n')
    result=dict(baseline_commit=commit,frozen_utc=datetime.now(timezone.utc).isoformat(),
                baseline=str(FORMAL),source=str(after),changed_files=[name],unchanged_files=unchanged,
                only_oc_q_load_replaced=True,restored_module_ast_equal=True,
                mathematical_ops_unchanged=True,valid_shapes_unchanged=True,
                root_abi_unchanged=True,device_executed=False)
    (OUT/'source_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
