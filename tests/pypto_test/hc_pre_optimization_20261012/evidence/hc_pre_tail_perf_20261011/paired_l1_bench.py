"""同一进程、同一输入，A/B交替采三轮完整HC_pre的L1；只运行PTO。"""
import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

TESTS = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test')
sys.path.insert(0, str(TESTS))
from dsv4_csa_env import activate, write_json
from dsv4_hc_pre_bench import load_weights, capture, compare, require_pass
from dsv4_hc_pre_timing import summarize_swimlane


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--measurement',choices=['interleave','hot'],default='interleave')
    p.add_argument('--batch-size', nargs='+', type=int, default=[2,4,8,16,20,24,28])
    p.add_argument('--seqlen', nargs='+', type=int, default=[5,6])
    a=p.parse_args()
    if not os.environ.get('TASK_DEVICE'): raise RuntimeError('须通过task-submit --device auto执行')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'report.json').exists(): raise RuntimeError('不得覆盖历史结果')
    activate()
    fs=[f for f in sys.meta_path if 'pypto.pypto_core' in getattr(f,'known_wheel_files',{})]
    assert len(fs)==1
    fs[0].known_wheel_files['pypto.pypto_core']='/data/pyptouser/qinchuanyu/pto-eager/pypto/build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so'
    import pypto.torch
    import torch
    import torch_npu
    torch.npu.set_device(0)
    weights,wr=load_weights(Path('/data/model/DeepSeek-V4-Flash-0731-w8a8'),2,'attn',torch.device('npu:0'))
    pypto.torch.init(device=0,platform='a2a3',runtime='tensormap_and_ringbuffer',enable_chip_swimlane=1,enable_dep_gen=True,output_dir=str(a.output/'dfx'))
    ops={}
    for tag,path in [('baseline',a.baseline),('candidate',a.candidate)]:
        name='hc_pre_pair_'+tag
        spec=importlib.util.spec_from_file_location(name,path.resolve());mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
        ops[tag]=pypto.torch.register(mod.hc_pre,'hc_pair::'+tag)
    report={'status':'RUNNING','scope':'同进程双实现；完整HC_pre L1，每侧固定三次；无Native，无探针；采样顺序见measurement及execution_order', 'measurement':a.measurement,'physical_device':os.environ['TASK_DEVICE'],'sources':{'baseline':str(a.baseline.resolve()),'candidate':str(a.candidate.resolve())},'weights':wr,'cases':[]}
    dfx_window = 0
    try:
        for window,(B,S) in enumerate(itertools.product(a.batch_size,a.seqlen)):
            T=B*S;g=torch.Generator().manual_seed(20261009+B*100+S)
            x=torch.randn(T,4,4096,dtype=torch.bfloat16,generator=g).npu()
            allocations={tag:(torch.full((T+2,4096),37,dtype=torch.bfloat16,device='npu:0'),torch.full((T+2,4),37,dtype=torch.float32,device='npu:0'),torch.full((T+2,4,4),37,dtype=torch.float32,device='npu:0')) for tag in ops}
            outputs={tag:tuple(v[1:-1] for v in vs) for tag,vs in allocations.items()}
            graphs={};eager={};runs={}
            row={'name':f'B{B:02d}_S{S}_T{T:03d}','tokens':T}
            for tag in ops:
                def run(tag=tag):
                    ops[tag](x,*weights,*outputs[tag]);return outputs[tag]
                runs[tag]=run;eager[tag]=tuple(v.cpu() for v in run())
                graphs[tag],actual=capture(run)
                row[tag+'_graph_eager']=compare(actual,eager[tag],exact=True);require_pass(row[tag+'_graph_eager'])
            row['initial']=compare(eager['candidate'],eager['baseline']);require_pass(row['initial'])
            for _ in range(5):
                for tag in graphs:graphs[tag].replay();torch.npu.synchronize()
            # 相邻场景交换首发顺序，仍保证每侧只有三个固定样本。
            order=(['baseline','candidate'] if window%2==0 else ['candidate','baseline'])*3
            # 每窗口不超过三步；hot模式让各实现分别预热并连续采样。
            groups=[order[2*i:2*i+2] for i in range(3)]
            if a.measurement=='hot':groups=[[order[0]]*3,[order[1]]*3]
            order=[tag for group in groups for tag in group]
            windows=[]
            for group in groups:
                if a.measurement=='hot':
                    for _ in range(5):graphs[group[0]].replay();torch.npu.synchronize()
                pypto.torch.begin_dfx()
                try:
                    for tag in group:graphs[tag].replay();torch.npu.synchronize()
                finally:pypto.torch.end_dfx()
                d=a.output/'dfx'
                if dfx_window:d/=f'window_{dfx_window}'
                dfx_window+=1;windows.append(d)
            samples=[];sources=[]
            for pair,d in enumerate(windows):
                trace=d/'paired_swimlane.json'
                result=subprocess.run([sys.executable,'-m','simpler_setup.tools.swimlane_converter',str(d/'chip_swimlane_records.json'),'-o',str(trace)],capture_output=True,text=True)
                (d/'converter_output.txt').write_text(result.stdout+result.stderr)
                if result.returncode:raise RuntimeError(result.stderr)
                metrics=summarize_swimlane(trace,len(groups[pair]))
                if any(v['blocks']!=72 for v in metrics['samples']):raise RuntimeError('每次必须完整覆盖24AIC和48AIV')
                for v in metrics['samples']:v['dfx_source']=str(trace)
                samples.extend(metrics['samples']);sources.append(str(trace))
                raw=json.loads(trace.read_text());epoch_role={v['launch_epoch']:role for role,v in zip(groups[pair],metrics['samples'])}
                for event in raw['traceEvents']:
                    args=event.get('args',{});epoch=args.get('launch_epoch')
                    if epoch in epoch_role and event.get('ph')=='X':event['name']=epoch_role[epoch]+'/'+event['name']
                trace.write_text(json.dumps(raw))
            row['execution_order']=order;row['l1']={'sources':sources,'samples':samples,'steps_per_variant':3}
            row['samples_us']={tag:[v['span_us'] for role,v in zip(order,samples) if role==tag] for tag in graphs}
            row['p50_us']={tag:statistics.median(v) for tag,v in row['samples_us'].items()}
            row['candidate_minus_baseline_us']=row['p50_us']['candidate']-row['p50_us']['baseline']
            x.copy_(torch.randn(T,4,4096,dtype=torch.bfloat16,generator=g).npu())
            for tag in graphs:
                for v in outputs[tag]:v.fill_(float('nan'))
                graphs[tag].replay();torch.npu.synchronize()
            row['updated_input']=compare(outputs['candidate'],outputs['baseline']);require_pass(row['updated_input'])
            for vs in allocations.values():
                if not all(bool((v[0].cpu()==37).all() and (v[-1].cpu()==37).all()) for v in vs):raise AssertionError('保护区发生改写')
            row['status']='PASS';report['cases'].append(row);write_json(a.output/'report.json',report)
            print(json.dumps({'case':row['name'],'p50_us':row['p50_us'],'delta_us':row['candidate_minus_baseline_us']},ensure_ascii=False),flush=True)
        report['status']='PASS'
    except Exception as e:report.update(status='FAIL',error=repr(e));raise
    finally:write_json(a.output/'report.json',report)

if __name__=='__main__':main()
