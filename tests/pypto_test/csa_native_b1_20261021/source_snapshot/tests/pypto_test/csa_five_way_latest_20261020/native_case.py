# SPDX-License-Identifier: Apache-2.0
"""正式 W8A8 层 4、七档 S6 的 Native 整层性能复测。

执行 HC_pre→输入 RMSNorm→Native CSA attention→HCpost，输入实际 T=batch×6。
复用 10/12 整层 npugraph_ex 配置，权重由当前正式加载器加载；hidden 与历史
KV 为 seed=1024 的合成 fixture，不冒充真实服务激活。只做 Native，没有 PTO
fallback、算子替换或精度开关修改。图由编译后端捕获，固定地址直接重放。

分别保留 20 次未 profile 的图外 Event，及独立 30 次 PyTorch Profiler Level1
全设备 kernel 首尾跨度。后者按全部流计算，剔除默认流上无图 ID 的 fixture
reset/输出毒化，保留图内拷贝和旁流。它不是 PTO 内部泳道或纯 AICore 时间。
"""
import argparse
import contextlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS))
from dsv4_csa_env import activate

SOURCE = activate()
import torch
import torch_npu
from dsv4_csa_native_case import native_session
from dsv4_csa_single_layer import collect_state, guard_checks, make_fixture, make_layer, restore
from offline_pd.run import decode_additional_config
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

# 本目录的计量器是已验证版本的原样副本，仅增加结果中的旁流描述字段。
sys.path.insert(0, str(HERE))
from measure import layer_device_interval, measure_graph_interval


class CompiledNativeHalf(torch.nn.Module):
    """与历史已验证入口相同，保持 Native HC 与 attention 的原始数学调用。"""
    def __init__(self, *, vllm_config, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor):
        layer = self.layer
        residual = hidden_states.clone()
        mixed, post, comb = layer.hc_pre(
            hidden_states, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
        normed = layer.input_layernorm(mixed)
        attended = layer.self_attn(positions=positions, hidden_states=normed, llama_4_scaling=None)
        output.copy_(layer.hc_post(attended, residual, post, comb))
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=int(os.environ.get('TASK_DEVICE', '-1')))
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--history', type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'report.json'
    if report_path.exists():
        raise FileExistsError('禁止覆盖已有报告，请使用新的输出目录')
    if args.device < 0 or str(args.device) != os.environ.get('TASK_DEVICE'):
        raise RuntimeError('仅允许使用 task-submit 自动分配的设备')
    preexisting = list((Path(os.environ['ASCEND_OPP_PATH']) / 'static_kernel').glob('**/binary_info_config.json'))
    if preexisting:
        raise RuntimeError('本次必须使用全新的私有 OPP static_kernel 目录')
    options = dict(force_eager=False, inplace_pass=True, clone_input=False,
                   clone_output=False, static_kernel_compile=True, super_kernel_optimize=True)
    report = {
        'status': 'RUNNING', 'side': 'native', 'batch': args.batch, 'history': args.history,
        'query_tokens': 6, 'tokens': args.batch * 6, 'layer_index': 4, 'seed': 1024,
        'device': args.device, 'source': str(SOURCE), 'cwd': str(Path.cwd()),
        'checkpoint': '/data/model/DeepSeek-V4-Flash-0731-w8a8',
        'scope': 'Native HC_pre + input RMSNorm + CSA attention + HCpost；无 MoE/EP16/整模型验收',
        'fixture_origin': '正式第4层权重；固定seed合成hidden及历史KV，不是真实服务激活',
        'cann': os.environ['ASCEND_HOME_PATH'], 'opp': os.environ['ASCEND_OPP_PATH'],
        'custom_opp': os.environ['ASCEND_CUSTOM_OPP_PATH'],
        'backend_options': options, 'torch_compile_options': {'dynamic': False, 'fullgraph': True},
        'compile_entry': 'torch.compile(backend=npugraph_ex)',
        'compilation_scope': {'vllm_ascend_fx_passes': False,
                              'note': '直接使用 npugraph_ex passes；不假设 EngineArgs 自动执行 vLLM FX passes'},
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torch-npu', 'vllm')},
    }
    original_qli = original_graph_run = original_super = original_compile = None
    try:
        current_platform.pre_register_and_update()
        torch.npu.set_device(args.device)
        torch.npu.config.allow_internal_format = True
        torch_npu.npu.set_deterministic_level(0)
        from vllm_ascend.utils import enable_custom_op
        if not enable_custom_op():
            raise RuntimeError('Native 自定义算子不可用')
        additional = decode_additional_config(SimpleNamespace(
            graph_mode='full_decode_only', capture_sizes=[args.batch * 6], recompute_scheduler=False))
        additional.update(weight_nz_mode=2, enable_kv_nz=False, enable_dsa_cp=False,
                          pto_csa_ring_config={'heap_mb': [256, 128, 256, 32], 'task_window': 4096})
        config = EngineArgs(
            model=report['checkpoint'], tokenizer_mode='deepseek_v4', trust_remote_code=True,
            dtype='bfloat16', quantization='ascend', tensor_parallel_size=1,
            hf_overrides={'sliding_window': 128}, max_model_len=max(16384, args.history + 128),
            max_num_seqs=40, max_num_batched_tokens=256, block_size=32, enable_prefix_caching=False,
            speculative_config={'method': 'dspark', 'num_speculative_tokens': 5, 'enforce_eager': True},
            compilation_config={'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [args.batch * 6],
                                'cache_dir': str(args.output / 'compile_cache')},
            additional_config=additional).create_engine_config()
        report['requested'] = additional
        report['compile_config'] = str(config.compilation_config)
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.ops.dsv4_csa import _native_attention_half
        with native_session(config, args.device), torch.inference_mode():
            device = torch.device(f'npu:{args.device}')
            layer, report['weights'] = make_layer(config, Path(report['checkpoint']), device, 4)
            report['native_weight_formats'] = {
                name: int(torch_npu.get_npu_format(getattr(layer.self_attn, name).weight))
                for name in ('wq_a', 'wq_b', 'wo_a', 'wo_b')}
            if set(report['native_weight_formats'].values()) != {29}:
                raise RuntimeError('Native四张核心投影必须实际为NZ格式29')
            report['runtime_norm_dtypes'] = {
                'input_layernorm': str(layer.input_layernorm.weight.dtype),
                'q_norm': str(layer.self_attn.q_norm.weight.dtype)}
            overlap = layer.self_attn.dsa_attn.dsa_attn.impl.multistream_dsv4_dsa_overlap
            report['multistream'] = {'dsa_overlap': overlap,
                                    'expression': 'torch.npu.stream + record_event/wait_event/wait_stream',
                                    'graph_owner': 'npugraph_ex force_eager=False',
                                    'profiler_span': '全部设备流最早kernel开始至最晚kernel结束'}
            if not overlap:
                raise RuntimeError('必须保留已验证的 Native DSA 多流重叠')
            fixture = make_fixture(config, layer.self_attn, args.batch, args.history, 1024, device, query_tokens=6)
            group = fixture['groups']['indexer']
            scale = group['views'][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            group['initial'] = group['allocation'].cpu()
            report['fixture_layout'] = {name: item['layout'] for name, item in fixture['groups'].items()}
            report['input_shapes'] = {name: list(fixture[name].shape) for name in ('hidden', 'positions')}
            output = torch.empty_like(fixture['hidden'])
            topk = {}
            original_qli = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer
            def record_qli(*inputs, **kwargs):
                value = original_qli(*inputs, **kwargs)
                topk['value'] = value[0]
                return value
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = record_qli
            @contextlib.contextmanager
            def context():
                with set_ascend_forward_context(fixture['metadata'], config,
                                               num_tokens=args.batch * 6, num_actual_tokens=args.batch * 6):
                    yield
            def topk_value():
                return topk['value']
            restore(fixture)
            with context():
                _native_attention_half(layer.self_attn.dsa_attn, fixture['hidden'], fixture['positions'], output)
            torch.npu.synchronize()
            report['initial_guards'] = guard_checks(fixture)
            if any(item['status'] != 'PASS' for item in report['initial_guards'].values()):
                raise RuntimeError('Native eager 初始保护区检查失败')
            import npugraph_ex
            report['npugraph_ex_source'] = str(Path(npugraph_ex.__file__).resolve())
            report['npugraph_ex_version'] = getattr(npugraph_ex, '__version__', 'bundled_without_version')
            from npugraph_ex._acl_concrete_graph import static_kernel
            from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph
            graph_instances = {}
            original_graph_run = AclGraph.run
            def record_graph_run(owner, key, *inputs, **kwargs):
                graph_instances[(id(owner), key)] = owner
                return original_graph_run(owner, key, *inputs, **kwargs)
            AclGraph.run = record_graph_run
            static_results, static_super_flags = [], []
            report['super_kernel_graph_calls'] = []
            original_super = torch.npu.NPUGraph.super_kernel_optimize
            def observe_super(graph, *inputs, **kwargs):
                call = {'status': 'STARTED'}
                report['super_kernel_graph_calls'].append(call)
                value = original_super(graph, *inputs, **kwargs)
                call['status'] = 'RETURNED_SUCCESSFULLY'
                return value
            torch.npu.NPUGraph.super_kernel_optimize = observe_super
            original_compile = static_kernel.static_compile
            def observe_static(*inputs, **kwargs):
                bound = inspect.signature(original_compile).bind_partial(*inputs, **kwargs)
                enabled = bool(kwargs.get('super_kernel_optimize', bound.arguments.get('super_kernel_optimize', False)))
                static_super_flags.append(enabled)
                if not enabled:
                    raise RuntimeError('静态编译器未接收到 SuperKernel=True')
                success = original_compile(*inputs, **kwargs)
                static_results.append(success)
                return success
            static_kernel.static_compile = observe_static
            torch.npu.set_compile_mode(jit_compile=False)
            module = CompiledNativeHalf(vllm_config=config, layer=layer)
            compiled = torch.compile(module, backend='npugraph_ex', fullgraph=True,
                                     dynamic=False, options=options)
            def compiled_call():
                with context():
                    compiled(fixture['hidden'], fixture['positions'], output)
            restore(fixture)
            compiled_call()
            torch.npu.synchronize()
            report['compiler'] = {
                'wrapper_compiled': bool(static_results), 'static_super_flags': static_super_flags,
                'static_compile_results': static_results,
                'installed_static_packages': len(static_kernel._installed_run_pkgs)}
            if not static_results or not all(static_results) or not all(static_super_flags):
                raise RuntimeError(f'静态编译未实际成功：{report["compiler"]}')
            if not static_kernel._installed_run_pkgs:
                raise RuntimeError('没有实际安装的静态 kernel 包')
            if len(graph_instances) != 1:
                raise RuntimeError('要求唯一完整后端图')
            (_, key), owner = next(iter(graph_instances.items()))
            if owner._updated_node_infos:
                raise RuntimeError('Host 更新节点不能直接跳过后端 wrapper 重放')
            graph = owner.graph[key]
            # 编译首调含预热，可能改变cache；同初态再次通过compiled-call获取参考。
            restore(fixture)
            compiled_call()
            torch.npu.synchronize()
            reference = collect_state(fixture, output, topk_value())
            report['measurement'] = {'capture_owner': 'npugraph_ex', 'graph_count': len(graph_instances),
                                     'host_updated_nodes': 0, 'timed_call': 'backend NPUGraph.replay',
                                     'reference': '同一捕获图经 compiled callable；固定地址与相同初态'}
            report['timing'] = measure_graph_interval(
                fixture, graph.replay, output, topk_value, reference,
                iters=20, warmup=5, require_exact=False, profile_dir=args.output / 'profile')
            calls = report['super_kernel_graph_calls']
            if not calls or any(call['status'] != 'RETURNED_SUCCESSFULLY' for call in calls):
                raise RuntimeError('图 SuperKernel 优化没有真实成功调用')
            if any(value['status'] != 'PASS' for value in report['timing']['eager_comparison'].values()):
                raise RuntimeError('同初态直接重放与 compiled callable 不一致')
            report['device_profile'] = layer_device_interval(args.output / 'profile')
            (args.output / 'device_profile_layer_only.json').write_text(
                json.dumps(report['device_profile'], ensure_ascii=False, indent=2) + '\n')
            report['device_span_us'] = report['device_profile']['span_us']
            report['unprofiled_event_mean_us'] = statistics.mean(report['timing']['samples_us'])
            report['status'] = 'MEASURED'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        if original_qli is not None:
            torch.ops._C_ascend.npu_vllm_quant_lightning_indexer = original_qli
        if original_graph_run is not None:
            AclGraph.run = original_graph_run
        if original_super is not None:
            torch.npu.NPUGraph.super_kernel_optimize = original_super
        if original_compile is not None:
            static_kernel.static_compile = original_compile
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('status', 'device_span_us', 'unprofiled_event_mean_us',
                                               'compiler', 'error') if k in report}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
