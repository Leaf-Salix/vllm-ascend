"""解释本worktree真实plan源码，核对物理行表示、回退及页表A→B→A。

复用已验证的NumPy基础DSL解释器，补INT32位移/UB gather；期望值来自独立
逐候选逻辑plan oracle再作64位物理地址计算，不使用候选Vector公式推期望。
这只是CPU语义/访存范围审计，不能替代NPU精度、Scalar缓存可见性或性能。
"""
import ast
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').exists() and (p / 'vllm_ascend').exists())
PACKAGE = Path(__file__).resolve().parent / 'source_candidate'
CANONICAL = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
N = CANONICAL / 'tests/pypto_test/csa_multishape_spmd_20261016'


def main():
    spec = importlib.util.spec_from_file_location('physical_plan_base_oracle', N/'dynamic_attention_plan_20261017/check_oracle.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    accesses = []
    ns, names, allowed, writes = base.source_namespace(PACKAGE, accesses)
    pl, Tile = ns['pl'], base.Tile
    ns.update(MAX_CANDIDATES=262144, CORES=72, GROUPS=24, PHASE_TOPK=36)
    # 此oracle只验证地址与数学，不模拟异步发布或缓存；真实协议由完整NPU门槛验证。
    ns['wait_row_owners'] = lambda signal, *unused: signal
    original_load = pl.load
    pl.load = lambda tensor, offsets, shape, valid_shape=None, **unused: original_load(tensor, offsets, shape, valid_shape)
    pl.MemorySpace = SimpleNamespace(Vec='Vec')
    def binary(fn, a, b):
        aa, bb = (x.data if isinstance(x, Tile) else x for x in (a, b))
        value = np.asarray(fn(aa, bb), np.int32)
        return Tile(value, a.valid if isinstance(a, Tile) else b.valid)
    pl.tile.maximums = lambda a,b: binary(np.maximum,a,b)
    pl.tile.minimums = lambda a,b: binary(np.minimum,a,b)
    pl.tile.shls = lambda a,b: binary(np.left_shift,a,b)
    pl.tile.shrs = lambda a,b: binary(np.right_shift,a,b)
    pl.tile.add = lambda a,b: binary(np.add,a,b)
    pl.tile.sub = lambda a,b: binary(np.subtract,a,b)
    def gather(source, indices, temporary):
        idx = indices.data
        assert idx.dtype == np.int32 and np.all(idx >= 0) and np.all(idx < source.valid[1])
        return Tile(source.data[0, idx], indices.valid)
    pl.tile.gather = gather
    cases = []
    for c in (0, 1, 32, 66, 255, 1025, 1026, 2048, 2049):
        cases.append(dict(name=f'columns_{c}', q=[0,5,9,13], lengths=[131077,139,0], tokens=13, columns=c, pages=19))
    for pages in (0,1,19,16777216,16777217):
        cases.append(dict(name=f'page_limit_{pages}',q=[0,1,6],lengths=[131073,137],tokens=9,columns=66,pages=pages))
    for name,q,l,t in (
        ('same_T_222',[0,2,4,6],[132,137,138],6),
        ('same_T_42',[0,4,6],[134,139],6),
        ('overlap',[0,6,2,8],[132,133,134],9),
        ('all_dummy',[99,-3,27],[0,-1],9),
        ('stale',[0,3,9],[134,139],11),
        ('fp32_position',[0,2],[16777219],2),
        ('int32_position',[0,2],[2147483647],2),
        ('empty',[0],[],0),
    ):
        cases.append(dict(name=name,q=q,lengths=l,tokens=t,columns=1026,pages=19))
    records=[]
    for case_num, c in enumerate(cases):
        q=np.asarray(c['q'],np.int32);lengths=np.asarray(c['lengths'],np.int32);t=c['tokens'];columns=c['columns'];pages=c['pages']
        owners=base.geometry(q,lengths,t)
        pos=np.full(t,-77,np.int64)
        for token,owner in enumerate(owners):
            if owner is not None:
                req,begin,end=owner;pos[token]=int(lengths[req])-(end-begin)+token-begin
        if c['name']=='stale':pos[::2]+=3
        topk=np.tile(np.arange(512,dtype=np.int32),(t,1))
        if t:topk[:,:14]=np.array([-2147483648,-1,0,31,32,63,64,columns*32-1,columns*32,32767,32768,16777216,16777217,2147483647],np.int32)
        original=np.tile(np.arange(4098,dtype=np.int32)%23,(len(lengths),1))
        table=np.tile(np.arange(columns,dtype=np.int32)%max(pages,1),(len(lengths),1))
        if columns:
            table[:,0]=pages-1 if pages else -1
        inputs=(topk,pos,q,lengths,original)
        inputs_before=[x.copy() for x in inputs]
        a_output=None
        for variant in ('A','B','A_again'):
            active_table=table.copy()
            if variant=='B' and columns:
                active_table=active_table[:,::-1].copy()
                active_table[:,::5]=-1
                if columns>1:active_table[:,1]=pages
                if columns>2:active_table[:,2]=2147483647
                if columns>3:active_table[:,3]=-2147483648
            expected_i, expected_b, expected_m, expected_owners=base.expected(topk,pos,q,lengths,original,active_table,23,pages)
            for enabled in (False,True):
                physical=0<columns<=2048 and 0<pages<=16777216
                indices=np.full((t+7,512),-777,np.int32);bias=np.full((t+7,640),777,np.float32);mask=np.full((t+7,16),-777,np.int32)
                expected=expected_i.copy()
                if physical:
                    for token,owner in enumerate(expected_owners):
                        if owner is not None:
                            req,_,_=owner
                            for k,raw in enumerate(expected_i[token]):
                                if raw>=0:expected[token,k]=int(active_table[req,int(raw)//32])*32+int(raw)%32
                names.clear();names.update({id(indices):'indices',id(bias):'bias',id(mask):'mask',id(active_table):'compressed_table'})
                accesses.clear();writes.clear();allowed.clear()
                worker_count=(1,3,48)[case_num%3]
                if enabled:
                    for worker in range(worker_count):
                        ns['attention_geometry_prepare_stage'](pos,q,lengths,active_table,indices,pages,worker,worker_count)
                    # 前缀写入是同代临时metadata；这里只统计最终indices的单写覆盖。
                    writes.clear()
                for worker in range(worker_count):
                    ns['attention_slot_plan_stage'](topk,pos,q,lengths,original,active_table,indices,bias,mask,None,np.int32(1),23,pages,worker,worker_count,False,enabled)
                np.testing.assert_array_equal(indices[:t],expected)
                np.testing.assert_array_equal(bias[:t],expected_b)
                np.testing.assert_array_equal(mask[:t,:5],expected_m)
                assert np.all(indices[t:]==-777) and np.all(bias[t:]==777) and np.all(mask[t:]==-777) and np.all(mask[:t,5:]==-777)
                counts=np.zeros_like(indices)
                for name,offset,shape in writes:
                    if name=='indices':counts[tuple(slice(o,o+n) for o,n in zip(offset,shape))]+=1
                assert np.all(counts[:t]==1) and np.all(counts[t:]==0)
                # 复用同一表示谓词：容量回退必须双方一致，不能把逻辑索引误当物理。
                assert bool(ns['physical_plan_supported'](columns,pages,True))==physical
                if enabled and variant=='A':a_output=indices.copy()
                if enabled and variant=='A_again':np.testing.assert_array_equal(indices,a_output)
                for left,right in zip(inputs,inputs_before):np.testing.assert_array_equal(left,right)
                records.append(dict(case=c['name'],variant=variant,early_geometry=enabled,physical=physical,tokens=t,columns=columns,pages=pages,workers=worker_count,passed=True))
    # 对照基线AST：所有原bias/clamp/causal赋值和全套同步调用须保持。
    audits={}
    for filename in ('attention_plan_stages.py','attention_stream_stages.py'):
        rel='vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/'+filename
        old=ast.parse(subprocess.check_output(['git','show','ef608731:'+rel],cwd=ROOT,text=True));new=ast.parse((PACKAGE/filename).read_text())
        sync=lambda tree:[ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func).startswith('pl.system.')]
        assert sync(old)==sync(new)
        checks=[]
        for name in ('c_pos_q','c_upper_b','c_ge_tile','c_lt_tile','c_mask_tile','c_out_tile','v_valid_tile','qk_mi','qk_li','qk_probability','next_l','next_left','next_right'):
            vals=lambda tree:[ast.dump(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(ast.unparse(x)==name for x in n.targets)]
            a,b=vals(old),vals(new)
            assert a==b,name
            if a:checks.append(name)
        audits[filename]=dict(sync_identical=True,math_assignments_identical=checks)
    out=dict(passed=True,device_execution=False,cases=records,source_audit=audits,case_runs=len(records),early_geometry_both_paths=True,synchronization_simulated=False,physical_address_dtype='INT32 shift/add; independent expected uses Python integer',boundary_guards_preserved=True,page_exchange='A -> B(reverse/negative/oob) -> A',limitations=['CPU解释不覆盖真实A3指令舍入或Scalar缓存可见性','未测性能；完整设备精度/graph/保护/协议仍待验证'])
    target=Path(__file__).parent/'oracle.json';target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,case_runs=len(records),source_audit=audits),ensure_ascii=False))

if __name__=='__main__':
    with np.errstate(invalid='ignore',over='ignore'):main()
