"""隔离唯一 DQ 索引候选并审核数学/owner；不导入 PyPTO，不编译或访问设备。

以9d提交的33源为准，拒绝覆盖冻结副本。只将完整8行块的索引链替换为
INT32 arange XOR 1；归一化后其余全部AST须相同，包括owner、尾行、KV和同步。
"""
from pathlib import Path
import ast
import difflib
import json
import shutil
import subprocess

ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
REPO = ROOT / 'vllm-ascend-dsv4-pto-0251rc1'
OUT = Path(__file__).resolve().parent
REL = 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
BASE = ROOT / '.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_indexer_20261018/query_dequant_group_20261018'
OLD = '''            q_swap_idx = pl.load(rope_swap_idx, [tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
            q_flat_i = pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32), pl.FP32)
            q_row_i = pl.cast(pl.mul(q_flat_i, 1.0 / ROPE_DIM), pl.INT32, mode="trunc")
            q_row_starts = pl.mul(pl.cast(q_row_i, pl.FP32), ROPE_DIM_SCALE)
            q_abs_swap = pl.cast(
                pl.add(pl.cast(pl.reshape(q_swap_idx, [1, Q_ROPE_T_TILE * ROPE_DIM]), pl.FP32), q_row_starts), pl.INT32
            )'''
NEW = '''            # 完整八行的局部索引j=64*r+c满足j XOR 1=64*r+(c XOR 1)。
            # prepare仍按行内0..63写共享rope_swap，KV照常消费；这里只去掉
            # DQ自己的索引GM读取和行偏移转换，不改变sin/cos等待或尾行数学。
            # arange/src、输出、tmp须由allocator分为三个不重叠UB区，A3 TXORS
            # 内部有多条Vector指令与barrier，不能把本表达式当作单条硬件XOR。
            # 生命周期仅限当前dq_work，禁止提升到动态工作循环外长期驻留。
            q_flat_i = pl.tile.arange(0, [1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32)
            q_swap_xor_tmp = pl.tile.create([1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32)
            q_abs_swap = pl.tile.xors(q_flat_i, 1, q_swap_xor_tmp)'''
NOTE = '''\n局部索引候选的边界\n    仅DQ完整八行块在本核生成j XOR 1，读取的cos/sin、head数学及动态owner\n    完全不变；q_rope_prepare仍为KV写行内swap，GM参数和跨核同步合同不变。\n    本地512项INT32临时只在当前工作块活跃；实际TXORS别名/UB须核对生成物。\n    此为独立CPU候选，尚无设备精度和完整L1性能结论。\n'''

class Normalize(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value,str):
            return None
        return self.generic_visit(node)
    def visit_Assign(self, node):
        if isinstance(node.targets[0],ast.Name) and node.targets[0].id in {'q_swap_idx','q_flat_i','q_row_i','q_row_starts','q_abs_swap','q_swap_xor_tmp'}:
            return None
        return self.generic_visit(node)

def dump(source):
    return ast.dump(Normalize().visit(ast.parse(source)), include_attributes=False)

def main():
    for name in ('source_before','source_candidate'):
        assert not (OUT/name).exists(), '拒绝覆盖已冻结源'
        (OUT/name).mkdir()
    paths=subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','--name-only','9d5327e5',REL],text=True).splitlines()
    paths=[p for p in paths if p.endswith('.py')]
    assert len(paths)==33
    checks=[]
    for rel in paths:
        src=subprocess.check_output(['git','-C',str(REPO),'show','9d5327e5:'+rel])
        name=Path(rel).name
        (OUT/'source_before'/name).write_bytes(src)
        (OUT/'source_candidate'/name).write_bytes(src)
        checks.append({'file':name,'formal_current_text_equal':(REPO/rel).read_bytes()==src,'existing_cpu_source_text_equal':(BASE/'source_candidate'/name).read_bytes()==src})
    assert all(x['formal_current_text_equal'] and x['existing_cpu_source_text_equal'] for x in checks)
    p=OUT/'source_candidate/query_projection_stages.py'
    original=p.read_text()
    assert original.count(OLD)==1
    result=original.replace(OLD,NEW)
    result=result.replace('\n阅读顺序\n',NOTE+'\n阅读顺序\n',1)
    p.write_text(result)
    assert dump(original)==dump(result), '除索引构造外的执行AST不得变化'
    assert all((64*r+(c^1))==((64*r+c)^1) for r in range(8) for c in range(64))
    # owner保持整段AST相同；枚举全部支持T验证每个真实token/head唯一覆盖。
    cases=[]
    for t in range(385):
        tiles=(t+7)//8
        group=4
        if t%8==0:
            for g in (64,32,16,8):
                denominator=48*g//64
                if tiles>=denominator and tiles%denominator==0:
                    group=g
                    break
        owners={}
        for worker in range(48):
            for work in range(worker,tiles*(64//group),48):
                tg=work//(64//group)*8
                hg=work%(64//group)*group
                for row in range(tg,min(tg+8,t)):
                    for h in range(hg,hg+group):
                        assert (row,h) not in owners
                        owners[row,h]=worker
        assert len(owners)==t*64
        cases.append({'tokens':t,'group':group,'unique_row_heads':len(owners),'full_chunk_index_max':511 if t>=8 else None})
    (OUT/'candidate.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),result.splitlines(True),fromfile='source_before/query_projection_stages.py',tofile='source_candidate/query_projection_stages.py')))
    shutil.copyfile(BASE/'controls.json',OUT/'controls.json')
    report={'passed':True,'baseline_commit':'9d5327e5','files':checks,'only_changed_file':'query_projection_stages.py','normalized_execution_ast_equal':True,'index_identity_cases':512,'owner_cases':cases,'owner_model_scope':'T0..384、48 AIV；源码owner AST未变化','dynamic_or_runtime_abi_changes':False,'shared_kv_swap_and_tail_math_unchanged':True,'cpu_or_device_executed_by_this_script':False}
    (OUT/'source_owner_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'files':len(paths),'index_cases':512,'owner_T_cases':len(cases),'source':str(p)}))

if __name__=='__main__':
    main()
