# SPDX-License-Identifier: Apache-2.0
"""准备O_A按组轮换M行块的动态候选；不改变任何矩阵tile或累加链。"""
import ast
import itertools
import json
import shutil
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=here,text=True).strip())
    formal = repo / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
    base = here.parent / 'attention_explicit_row_dma_20261019'
    saved = {p.name:p.read_bytes() for p in formal.glob('*.py')}
    assert all(data == (base/'source_reference'/name).read_bytes() for name,data in saved.items())
    text = saved['output_projection_a_stages.py'].decode()
    marker = '分工、数学与尾行\n'
    assert marker in text
    text = text.replace(marker, marker+
        '    本候选在同一输出组内将物理M行块循环平移g个位置，group/列块/\n'
        '    logical%24的执行槽不变。对于两个M行块，奇数组交换满块与短尾块，\n'
        '    避免24与每组8项整除关系让同一批AIC反复承接满块。更多行块也\n'
        '    使用同一循环置换，不依赖某个B/S。单M行块跳过置换。\n'
        '    当前量化按组等待该组全部OA owner，因此组内行块置换不改变\n'
        '    等待集合。每个输出仍只写一次，K累加链、有效行与尾零责任保持。\n',1)
    before = '        pa_rb = pa_unit // column_blocks\n'
    assert text.count(before)==2
    text = text.replace(before,before+
        '        # g内只是行块的双射；执行槽owner及整组完成通知保持不变。\n'
        '        # row_blocks由实际T得到，T≤128时不增加一次无用的模运算。\n'
        '        if row_blocks > 1:\n'
        '            pa_rb = (pa_rb + g) % row_blocks\n')
    ast.parse(text)
    for name,data in saved.items():
        p=here/'source_reference'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
    try:
        (formal/'output_projection_a_stages.py').write_text(text)
        shutil.copytree(formal,here/'source_candidate',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    finally:
        for name,data in saved.items():(formal/name).write_bytes(data)
    for name in ['run_cpu.sh','run_device.sh','policy_candidate.json']:
        shutil.copyfile(base/name,here/name)
    controls=json.loads((base/'controls_on.json').read_text())
    controls.update(OA_COLS=256,OB_ROWS=128,LOCAL_ATTENTION_STATS=False,SCALE_PAGE_COALESCING=True)
    (here/'controls_on.json').write_text(json.dumps(controls,ensure_ascii=False,indent=2)+'\n')
    checks=[]
    for tokens in [1,24,48,64,96,127,128,129,144,168,192,255,256,257,384,512,1024,2048]:
        for n in [128,256]:
            rb=(tokens+127)//128;cb=1024//n
            old=[];new=[];old_cost=[0]*24;new_cost=[0]*24
            for logical in range(8*rb*cb):
                group,unit=divmod(logical,rb*cb);row,col=divmod(unit,cb);rot=(row+group)%rb
                old.append((group,row,col,logical%24));new.append((group,rot,col,logical%24))
                old_cost[logical%24]+=min(128,tokens-row*128)*n
                new_cost[logical%24]+=min(128,tokens-rot*128)*n
            expected=set(itertools.product(range(8),range(rb),range(cb)))
            assert {x[:3] for x in new}==expected and len(new)==len(expected)
            for group in range(8):
                assert {x[3] for x in old if x[0]==group}=={x[3] for x in new if x[0]==group}
            assert sum(old_cost)==sum(new_cost)
            checks.append(dict(tokens=tokens,n=n,old_max_row_columns=max(old_cost),
                               new_max_row_columns=max(new_cost),old_per_core=old_cost,new_per_core=new_cost))
    (here/'OWNER_COVERAGE.json').write_text(json.dumps(dict(
        passed=True,scope='整数分工覆盖/唯一写/整组owner集合；行列工作量不是性能模型',
        cases=checks),ensure_ascii=False,indent=2)+'\n')
    (here/'SOURCE_SCOPE.json').write_text(json.dumps(dict(
        authored_in_ops=True,formal_source_restored=True,changed_files=['output_projection_a_stages.py'],
        reference_commit='289cc072',dependencies_changed=False,math_tiles_changed=False,
        other_candidates_combined=False,cpu_validated=False,device_validated=False,
    ),ensure_ascii=False,indent=2)+'\n')
    print('prepared rotation and',len(checks),'integer coverage cases')


if __name__=='__main__':main()
