"""8K真实权重、Native/TND完整CSA FULL图验收；首差停止。"""
import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import types

import torch
import torch_npu
import pypto.torch
from vllm.config import CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
import vllm_ascend.ops.dsv4_csa
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import make_layer, restore, collect_state, guard_checks
from ragged_fixture import make_ragged_fixture
from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph

ROOT = Path(__file__).resolve().parent
BASE = Path('/data/pyptouser/yejia/vllm-nalinaly-tnd-20261008-codex')
META_SHA = '863a1e5daffc54aec0e9e8afb9f809f1bc6f4b3a2487e440615f3c942f36c602'
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--lengths', default='1,1,1,1')
parser.add_argument('--layer', type=int, default=4)
parser.add_argument('--history', type=int, default=8191)
args = parser.parse_args()
lengths = list(map(int,args.lengths.split(',')))
tokens, batch = sum(lengths), len(lengths)
assert all(1<=q<=6 for q in lengths) and batch<=40 and tokens<=256
assert not (args.out/'result.json').exists()
args.out.mkdir(parents=True,exist_ok=True)
expected_helpers=json.loads((ROOT/'manifest.json').read_text())
helper_sha={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in expected_helpers}
assert helper_sha==expected_helpers,'helper source identity mismatch'
manifest_sha=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
report = {'helper_sha':helper_sha,'manifest_sha256':manifest_sha,'complete':False,'pass':False,'scope':'real layer weights, synthetic hidden/history, Native/TND complete CSA half-layer FULL graph; not whole model',
          'lengths':lengths,'true_tokens':tokens,'physical_graph_tokens':tokens,'per_request_padding':0,
          'hypothetical_s6_tokens':batch*6,'hypothetical_s6_extra_rows':batch*6-tokens,
          'baseline_experiment':'Native direct HC-pre through HC-post', 'candidate_experiment':'precision-spmd-prepack-full-root-v2',
          'candidate_variant':'one SPMD prefix, original PTO suffix; full HC-pre through HC-post',
          'layer':args.layer,'history':args.history,'graphs':{},'comparisons':{},'static':{}}
def save():
    (args.out/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
def exact(x,y):
    return x.shape==y.shape and x.dtype==y.dtype and torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
def metric(x,y):
    if x.shape!=y.shape or x.dtype!=y.dtype:
        return {'bit_equal':False,'shapes':[list(x.shape),list(y.shape)],'dtypes':[str(x.dtype),str(y.dtype)]}
    if x.is_floating_point():
        assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()), 'nonfinite external float boundary'
    equal=exact(x,y)
    delta=x.float()-y.float()
    return {'bit_equal':equal,'different_elements':int((x!=y).sum()),'max_abs':float(delta.abs().max()),
            'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(y.float()).clamp_min(1e-30)),
            'first_coordinates':(x!=y).nonzero()[:6].tolist()}

import vllm_ascend.ops.pypto as spmd_namespace
experiment=Path(os.environ['SPMD_EXPERIMENT'])
spmd_namespace.__path__.insert(0,str(experiment/'src/vllm_ascend/ops/pypto'))
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd_prepack.native_adapter import CSAOperators
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd_prepack.service import CSAServiceRuntime
from source_identity import source_identity
source_record=source_identity()
report.update(source_record)
import vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd_prepack.decode_csa as formal_root
assert Path(formal_root.__file__).resolve().is_relative_to(experiment/'src')
package=Path(formal_root.__file__).resolve().parent
candidate_sources={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in package.glob('*.py')}
receipt=json.loads((experiment/'compile-full-csa-prepack.json').read_text())
assert receipt['compiled'] and receipt['kernel_mode_abi_pass'] and receipt['torch_dispatcher_registration_pass']
assert receipt['formal_prepare_program_compiled'] and receipt['formal_prepare_kernel_mode_abi_pass'] and receipt['formal_operators_registered']
assert receipt['sources']==candidate_sources
assert receipt['baseline_sources']==source_record['tested_sources']
parent=package.with_name('deepseek_v4_flash_csa_tnd_precision_spmd')
assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob('*.py')}==receipt['parent_spmd_sources']
report['parent_spmd_sources']=receipt['parent_spmd_sources']
report['candidate_sources']=candidate_sources
report['candidate_package']=str(package)
report['compile_receipt_sha256']=hashlib.sha256((experiment/'compile-full-csa-prepack.json').read_bytes()).hexdigest()
report['formal_root_file']=formal_root.__file__

class NativeHalf(torch.nn.Module):
    def __init__(self,layer):
        super().__init__();self.layer=layer
    def forward(self,hidden,positions,output):
        l=self.layer
        mixed,post,comb=l.hc_pre(hidden,l.hc_attn_fn,l.hc_attn_scale,l.hc_attn_base)
        normed=l.input_layernorm(mixed)
        attended=l.self_attn(positions=positions,hidden_states=normed,llama_4_scaling=None)
        output.copy_(l.hc_post(attended,hidden,post,comb))
        return output

class PTOHalf(torch.nn.Module):
    def __init__(self,prefix):
        super().__init__();self.prefix=prefix
    def forward(self,hidden,positions,output):
        torch.ops.vllm.dsv4_csa_forward(hidden,positions,output,self.prefix)
        return output

class RequirePTO:
    def __init__(self,delegate):
        self.delegate,self.calls=delegate,0
    def eligible(self,*values):
        if not self.delegate.eligible(*values):
            raise RuntimeError('Unexpected PTO fallback')
        return True
    def __call__(self,*values,**kwargs):
        self.calls+=1
        return self.delegate(*values,**kwargs)

current_platform.pre_register_and_update()
torch.npu.set_device(0)
torch.npu.config.allow_internal_format=True
torch.use_deterministic_algorithms(True)
torch_npu.npu.set_deterministic_level(1)
torch.set_num_threads(4)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
checkpoint=Path('/data/model/DeepSeek-V4-Flash-0731-w8a8')
config=EngineArgs(model=str(checkpoint),tokenizer_mode='deepseek_v4',trust_remote_code=True,
 tensor_parallel_size=1,dtype='bfloat16',quantization='ascend',hf_overrides={'sliding_window':128},
 max_model_len=16384,max_num_seqs=40,max_num_batched_tokens=256,enable_prefix_caching=False,
 enforce_eager=False,block_size=32,speculative_config={'method':'dspark','num_speculative_tokens':5,'enforce_eager':True},
 compilation_config={'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[tokens],'cache_dir':str(args.out/'vllm-cache')},
 additional_config={'ascend_compilation_config':{'enable_npugraph_ex':True,'enable_static_kernel':True,'fuse_norm_quant':True,'align_decode_capture_sizes':False},
 'enable_cpu_binding':False,'multistream_overlap_shared_expert':True,'weight_nz_mode':2,'enable_kv_nz':False,'enable_dsa_cp':False}).create_engine_config()

states, fixtures, rows, keepalive, b_states = {}, {}, {}, [], {}
save()
try:
 with native_session(config,0),torch.inference_mode():
    layer,details=make_layer(config,checkpoint,torch.device('npu:0'),args.layer)
    report['weight_details']=details
    pypto.torch.init()
    impl=layer.self_attn.dsa_attn.dsa_attn.impl
    observed={}
    original_qli=torch.ops._C_ascend.npu_vllm_quant_lightning_indexer
    def observe_qli(*values,**kwargs):
        result=original_qli(*values,**kwargs);observed['topk']=result[0];return result
    torch.ops._C_ascend.npu_vllm_quant_lightning_indexer=observe_qli
    # 保留原13包校验桥；只在首次真实reselect后复用选择，禁止捕获之后变更选择。
    selection=[]
    original_reselect=torch_npu.npu._aclnn_reselect_static_kernel
    def qualified_reselect(*values,**kwargs):
        assert hashlib.sha256((Path(os.environ['LOW_ACCEPTANCE_OFFICIAL_CACHE'])/'metadata.json').read_bytes()).hexdigest()==META_SHA
        assert not values and not kwargs
        if not selection:
            selection.append(original_reselect())
            report['static']['actual_reselect_count']=1
        else:
            report['static']['validated_reuses']=report['static'].get('validated_reuses',0)+1
        report['static']['metadata_sha256']=META_SHA
        return selection[0]
    # 框架可能已from-import compile_static_kernel；在实际API边界观测，避免漏掉别名调用。
    torch_npu.npu._aclnn_reselect_static_kernel=qualified_reselect

    for arm in ('native','candidate'):
        print('ARM_START',arm,flush=True)
        torch._dynamo.reset()
        f=make_ragged_fixture(config,layer.self_attn,batch,args.history,1024,torch.device('npu:0'),lengths)
        fixtures[arm]=f
        if arm!='native':
            for field in ('hidden','positions'):
                assert exact(f[field],fixtures['native'][field]),(arm,field)
            for name,g in f['groups'].items():
                assert exact(g['initial'],fixtures['native']['groups'][name]['initial']),(arm,name)
        runtime=None
        if arm!='native':
            operators=CSAOperators.register()
            runtime=CSAServiceRuntime(layer.self_attn,operators,40,layer)
            from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd.workspace import NORM_PARAMETERS
            norm_offset=0
            norm_packet_checks={}
            for name,size in NORM_PARAMETERS:
                norm_packet_checks[name]=exact(runtime.norm_packet[norm_offset:norm_offset+size].cpu(),runtime.weights[name].cpu())
                norm_offset+=size
            assert norm_offset==6272 and all(norm_packet_checks.values())
            order_checks={}
            for (w,i,o),packet in operators.workspace.order_packets.items():
                cpu=packet.cpu()
                order_checks[str((w,i,o))]=all((exact(cpu[:512],runtime.wkv_orders[w].cpu().reshape(-1)),
                    exact(cpu[512:1536],runtime.indexer_orders[i].cpu().reshape(-1)),
                    exact(cpu[1536:],runtime.oproj_starts[o].cpu().reshape(-1))))
            assert len(order_checks)==33 and all(order_checks.values())
            report['packet_checks']={'norm_byte_equal':norm_packet_checks,'order_byte_equal':order_checks,
                'norm_bytes_per_layer':runtime.norm_packet.numel()*runtime.norm_packet.element_size(),
                'order_bytes_per_model':sum(t.numel()*t.element_size() for t in operators.workspace.order_packets.values())}
            readonly_packets={'norm':runtime.norm_packet,'hc_packed':runtime.weights['hc_packed_weight'],**{str(k):v for k,v in operators.workspace.order_packets.items()}}
            packet_snapshots={k:v.cpu() for k,v in readonly_packets.items()}
            report['hc_weight_qualification']=dict(runtime.hc_weight_qualification)
            assert report['hc_weight_qualification']['version']==1
            assert selection,'Native graph did not install and select static packages'
            runtime.qualify_static_wkv(metadata_sha256=META_SHA,reselected=True)
        @contextlib.contextmanager
        def context():
            with set_ascend_forward_context(f['metadata'],config,num_tokens=tokens,num_actual_tokens=tokens,
                                            aclgraph_runtime_mode=CUDAGraphMode.FULL):
                assert get_forward_context().cudagraph_runtime_mode is CUDAGraphMode.FULL
                yield
        if runtime is not None:
            with context():
                eligible=runtime.eligible(get_forward_context(),f['hidden'],f['positions'])
            report.setdefault('entry',{})[arm]={'eligible':eligible,'route':'PTO' if eligible else 'Native fallback','metadata':{
                name:{'bounds':f['metadata'][g['prefix']].decode.query_start_loc.cpu().tolist(),
                      'max_q':f['metadata'][g['prefix']].decode.max_seqlen_q,
                      'num_decodes':f['metadata'][g['prefix']].num_decodes,
                      'actual_tokens':f['metadata'][g['prefix']].num_actual_tokens} for name,g in f['groups'].items()}}
            save()
            assert eligible, 'TND precision fallback is forbidden'
            guarded=RequirePTO(runtime)
            layer.self_attn.dsa_attn._pto_csa_runtime=guarded
        else:
            layer.self_attn.dsa_attn._pto_csa_runtime=None
        output=torch.empty_like(f['hidden'])
        module=NativeHalf(layer) if arm=='native' else PTOHalf(layer.self_attn.dsa_attn.prefix)
        restore(f)
        with context():module(f['hidden'],f['positions'],output)
        torch.npu.synchronize()
        instances={}
        old_run=AclGraph.run
        def record(owner,key,*values,**kwargs):
            instances[(id(owner),key)]=owner
            return old_run(owner,key,*values,**kwargs)
        AclGraph.run=record
        options={'force_eager':False,'inplace_pass':True,'clone_input':False,'clone_output':False,
                 'static_kernel_compile':True,'super_kernel_optimize':False}
        torch.npu.set_compile_mode(jit_compile=False)
        try:
            compiled=torch.compile(module,backend='npugraph_ex',fullgraph=True,dynamic=False,options=options)
            restore(f)
            with context():compiled(f['hidden'],f['positions'],output)
            torch.npu.synchronize()
            assert len(instances)==1,(arm,'not exactly one actual graph',len(instances))
            (_,key),owner=next(iter(instances.items()))
            assert not owner._updated_node_infos,'Host updated nodes invalidate raw replay test'
            graph=owner.graph[key]
        finally:
            AclGraph.run=old_run
        topk=runtime.topk[:tokens] if runtime is not None and report['entry'][arm]['eligible'] else observed['topk']
        compiled_state=collect_state(f,output,topk)
        restore(f);output.fill_(float('nan'));topk.fill_(-12345);graph.replay();torch.npu.synchronize()
        actual=collect_state(f,output,topk)
        assert all(exact(value,compiled_state[name]) for name,value in actual.items()),(arm,'compiled/raw mismatch')
        assert torch.isfinite(output).all()
        guards=guard_checks(f)
        assert all(value['status']=='PASS' for value in guards.values()),(arm,'guard',guards)
        states[arm]=actual
        rows[arm]=(graph,f,output,topk)
        keepalive.append((compiled,module,owner,runtime))
        report['graphs'][arm]={'actual_acl_graph':True,'fullgraph':True,'options':options,
                                'true_tokens':tokens,'graph_tokens':tokens,'guards':guards,'compiled_raw_equal':True,
                                'topk_poison_replay_restored':True,'native_super_kernel_optimize':False}
        if arm!='native':
            checks={name:metric(value,states['native'][name]) for name,value in actual.items()}
            report['comparisons'][arm]=checks
            save()
            if not all(check['bit_equal'] for check in checks.values()):
                report['first_failed_arm']=arm
                report['failed_boundaries']=[name for name,m in checks.items() if not m['bit_equal']]
                raise RuntimeError(arm+': first bit difference; stop matrix and timing')
        before=f['hidden'].clone()
        restore(f);f['hidden'].add_(0.125);graph.replay();torch.npu.synchronize()
        assert not exact(output.cpu(),actual['x_out']),(arm,'stale input graph')
        b_states[arm]=collect_state(f,output,topk)
        if arm!='native':
            b_checks={name:metric(value,b_states['native'][name]) for name,value in b_states[arm].items()}
            report.setdefault('B_comparisons',{})[arm]=b_checks;save()
            if not all(check['bit_equal'] for check in b_checks.values()):
                torch.save(b_states,args.out/'first-B-difference.pt')
                raise RuntimeError('TND input B first bit difference; stop')
        assert all(value['status']=='PASS' for value in guard_checks(f).values()),(arm,'B guards')
        f['hidden'].copy_(before);restore(f);graph.replay();torch.npu.synchronize()
        restored=collect_state(f,output,topk)
        assert all(exact(value,actual[name]) for name,value in restored.items()),(arm,'A restore')
        report['graphs'][arm]['live_input_ABA']=True
        if runtime is not None:
            assert guarded.calls>0,'PTO runtime was never invoked'
            report['graphs'][arm]['pto_dispatch_calls']=guarded.calls
            schema=str(runtime.operators.attention._schema)
            expected_namespace='dsv4_csa_tnd_precision_spmd_prepack::attention'
            assert schema.startswith(expected_namespace+'('),(arm,schema)
            assert schema.split(' -> ')[0].count('Tensor')==64,(arm,schema)
            report['graphs'][arm]['root_schema']=schema
            report['workspace_bytes']=sum(v.numel()*v.element_size() for v in operators.workspace.buffers.values())
            report['hc_packed_weight_bytes']=runtime.weights['hc_packed_weight'].numel()*runtime.weights['hc_packed_weight'].element_size()
            report['workspace_capture_stream']=operators.workspace.capture_stream
        save()
        print('ARM_PASS',arm,flush=True)
    # Reproduce vLLM: first bucket captured on a side stream, NONE warmup
    # for a second token count on the compute stream, then capture that bucket.
    assert tokens==21 and lengths==[1,2,3,4,5,6]
    keepalive.append(dict(runtime.weights))
    compute=torch.npu.Stream()
    assert compute.npu_stream!=operators.workspace.capture_stream
    f6=make_ragged_fixture(config,layer.self_attn,1,args.history,1024,torch.device('npu:0'),[6])
    report['readonly_after_second_fixture']={n:metric(v.cpu(),expected) for n,(v,expected) in f['readonly'].items()}
    changed={n for n,m in report['readonly_after_second_fixture'].items() if not m['bit_equal']}
    report['shared_rope_views']={n:{'first_ptr':f['readonly'][n][0].data_ptr(),
        'second_ptr':f6['readonly'][n][0].data_ptr(),'first_shape':list(f['readonly'][n][0].shape),
        'second_shape':list(f6['readonly'][n][0].shape),
        'same_storage':f['readonly'][n][0].untyped_storage().data_ptr()==f6['readonly'][n][0].untyped_storage().data_ptr()}
        for n in ('rope.cos','rope.sin')}
    save()
    assert changed=={'rope.cos','rope.sin'}, ('unexpected readonly change',changed)
    assert all(row['same_storage'] for row in report['shared_rope_views'].values())
    def restore_readonly(fixture):
        for live,expected in fixture['readonly'].values():live.copy_(expected)
    @contextlib.contextmanager
    def context6():
        with set_ascend_forward_context(f6['metadata'],config,num_tokens=6,num_actual_tokens=6,
                                        aclgraph_runtime_mode=CUDAGraphMode.FULL):
            yield
    def capture_other(mod,fixture,token_count,ctx):
        torch._dynamo.reset()
        actual_graphs={}
        old=AclGraph.run
        def observe(owner,key,*values,**kwargs):
            actual_graphs[(id(owner),key)]=owner
            return old(owner,key,*values,**kwargs)
        AclGraph.run=observe
        try:
            wrapped=torch.compile(mod,backend='npugraph_ex',fullgraph=True,dynamic=False,options=options)
            out=torch.empty_like(fixture['hidden'])
            restore(fixture)
            with ctx():wrapped(fixture['hidden'],fixture['positions'],out)
            torch.npu.synchronize()
            assert len(actual_graphs)==1
            (_,key),owner=next(iter(actual_graphs.items()))
            assert not owner._updated_node_infos
            keepalive.append((wrapped,mod,owner,out))
            return owner.graph[key],out
        finally:
            AclGraph.run=old
    # Static Native qualification is already installed; a real second FULL graph
    # provides the six-row golden boundary instead of a pre-selection eager run.
    layer.self_attn.dsa_attn._pto_csa_runtime=None
    native6,out_n6=capture_other(NativeHalf(layer),f6,6,context6)
    restore(f6);native6.replay();torch.npu.synchronize()
    golden6=collect_state(f6,out_n6,observed['topk'])
    assert all(g['status']=='PASS' for g in guard_checks(f6).values())
    layer.self_attn.dsa_attn._pto_csa_runtime=guarded
    with torch.npu.stream(compute):
        restore(f6)
        out_w6=torch.empty_like(f6['hidden'])
        with context6():module(f6['hidden'],f6['positions'],out_w6)
        warm=collect_state(f6,out_w6,runtime.topk[:6])
    checks={n:metric(v,golden6[n]) for n,v in warm.items()}
    report['cross_bucket_warmup']={'first_tokens':21,'second_tokens':6,'different_stream':True,
                                   'checks':checks,'capture_stream':operators.workspace.capture_stream,
                                   'compute_stream':compute.npu_stream}
    save();assert all(v['bit_equal'] for v in checks.values()), 'first cross-stream warmup difference'
    with torch.npu.stream(compute):
        graph6,out6=capture_other(module,f6,6,context6)
    report['cross_bucket_warmup']['second_actual_FULL_graph']=True
    assert operators.workspace.capture_stream==report['cross_bucket_warmup']['capture_stream']
    for repeat in range(3):
        for label,g,fixture,out,count,golden in (
            ('21a',rows['candidate'][0],f,output,21,states['native']),
            ('6',graph6,f6,out6,6,golden6),
            ('21b',rows['candidate'][0],f,output,21,states['native'])):
            with torch.npu.stream(compute):
                restore(fixture);restore_readonly(fixture)
                assert all(c['status']=='PASS' for c in guard_checks(fixture).values()),'metadata input differs before replay'
                out.fill_(float('nan'));runtime.topk[:count].fill_(-12345)
                for v in operators.workspace.buffers.values():
                    if v.is_floating_point():v.fill_(float('nan'))
                g.replay()
                torch.npu.current_stream().synchronize()
                value=collect_state(fixture,out,runtime.topk[:count])
            comparisons={n:metric(v,golden[n]) for n,v in value.items()}
            report.setdefault('cross_bucket_replay_checks',{})[str(repeat)+'/'+label]=comparisons
            save();assert all(c['bit_equal'] for c in comparisons.values()), 'first cross-bucket replay difference'
            assert all(c['status']=='PASS' for c in guard_checks(fixture).values())
    report['cross_bucket_warmup']['checks_count']=8+9*8
    report['cross_bucket_warmup']['concurrent_replay_tested']=False
    print('CROSS_BUCKET_STREAM_WARMUP_PASS',flush=True)
    samples={arm:[] for arm in rows}
    arms=list(rows)
    for arm in arms:
        graph,f,output,topk=rows[arm]
        restore(f)
        for warm in range(3):graph.replay()
    torch.npu.synchronize()
    for round_id in range(8):
        for arm in arms if round_id%2==0 else list(reversed(arms)):
            graph,f,output,topk=rows[arm]
            restore(f);restore_readonly(f)
            assert all(c['status']=='PASS' for c in guard_checks(f).values())
            torch.npu.synchronize()
            start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            start.record()
            for iteration in range(10):graph.replay()
            end.record();end.synchronize()
            samples[arm].append(start.elapsed_time(end)*1000/10)
            actual=collect_state(f,output,topk)
            assert all(exact(value,states[arm][name]) for name,value in actual.items()),(arm,'timing state drift')
            assert all(g['status']=='PASS' for g in guard_checks(f).values()),(arm,'timing guards')
    report['timings']={'unit':'us/complete HC-pre through HC-post raw graph replay, SK0',
        'samples':samples,'summary':{arm:{'min':min(v),'mean':statistics.mean(v),'max':max(v)} for arm,v in samples.items()}}
    assert all(exact(v.cpu(),packet_snapshots[k]) for k,v in readonly_packets.items()), 'readonly packet changed'
    report['packet_checks']['readonly_after_replay']=True
    report['timing_executed']=True
    report['complete']=report['pass']=True
    save()
except Exception as exc:
    report['error']=repr(exc)
    save()
    torch.save({'states':states,'inputs':{arm:{'hidden':f['hidden'].cpu(),'positions':f['positions'].cpu(),
                    'initial':{name:g['initial'] for name,g in f['groups'].items()}} for arm,f in fixtures.items()}},args.out/'first-difference.pt')
    raise
print('COMPLETE',report['pass'],lengths,flush=True)
