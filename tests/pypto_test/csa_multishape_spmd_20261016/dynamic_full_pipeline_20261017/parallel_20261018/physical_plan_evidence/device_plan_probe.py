# SPDX-License-Identifier: Apache-2.0
"""在队列分配的同一张卡上验证私有候选plan的物理行含义。

本脚本只测metadata，不测性能、不创建KV大缓存。输入为实际device Q/L/position、
TopK和页表；13行含奇数query、mixed长度和dummy。A→B→A复用各Tensor地址，
B交换页表并加入负页/OOB页、陈旧position。C=2049核对完整旧逻辑回退。
输出indices以独立逐候选Python整数oracle核对；bias/mask原数学逐bit不变，
所有未写padding保留poison。此probe明确导入当前私有源码，绝不使用canonical
默认关闭开关的plan。它不等同完整CSA精度或Scalar跨核协议证明，后者由同任务
后续main matrix完成。只有整个probe成功才启动完整CSA测试。
"""
import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
CANONICAL = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
D = CANONICAL / 'tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017'
sys.path.insert(0, str(D))
sys.path.insert(0, str(CANONICAL / 'tests/pypto_test'))
import dsv4_csa_env
dsv4_csa_env.activate()
from source_loader import load_package

package = load_package(WORKTREE / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd', 'physical_plan_probe')
source = importlib.import_module(package.name + '.attention_plan_stages')
import pypto.language as pl

T_DYN = source.T_DYN
REQUESTS_DYN = source.REQUESTS_DYN
QUERY_BOUNDS_DYN = source.QUERY_BOUNDS_DYN
ORIGINAL_COLUMNS_DYN = source.ORIGINAL_COLUMNS_DYN
COMPRESSED_COLUMNS_DYN = source.COMPRESSED_COLUMNS_DYN
PLAN_CAP_DYN = source.PLAN_CAP_DYN
attention_slot_plan_stage = source.attention_slot_plan_stage


@pl.jit.incore
def plan_core(
    topk: pl.Tensor[[T_DYN, 512], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    original_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    compressed_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    indices: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 512], pl.INT32]],
    bias: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32]],
    mask: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32]],
):
    # Tile必须属于incore。所有worker按真实T分M8组，尾5行不能写进24行guard。
    worker = pl.tile.get_block_idx()
    indices, bias, mask = attention_slot_plan_stage(
        topk, positions, bounds, lengths, original_table, compressed_table,
        indices, bias, mask, 23, 19, worker, 48, False, True,
    )
    return indices, bias, mask


@pl.jit
def plan_probe(
    topk: pl.Tensor[[T_DYN, 512], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    original_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    compressed_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    indices: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 512], pl.INT32]],
    bias: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32]],
    mask: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32]],
):
    with pl.spmd(48, name_hint='private_physical_plan_probe'):
        indices, bias, mask = plan_core(
            topk, positions, bounds, lengths, original_table, compressed_table, indices, bias, mask,
        )
    return indices, bias, mask


def main():
    import numpy as np
    import torch
    import torch_npu
    import pypto.torch
    from pypto import passes
    from pypto.runtime import CompileOptions

    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, device_execution=True, performance_measured=False,
                  source_package=str(package.source), cases=[], probe_before_matrix=True)
    try:
        device_id = int(os.environ['TASK_DEVICE'])
        torch.npu.set_device(device_id)
        torch_npu.npu.set_deterministic_level(0)
        device = torch.device(f'npu:{device_id}')
        pypto.torch.init(device=device_id, platform='a2a3', runtime='tensormap_and_ringbuffer')
        config = CompileOptions(platform='a2a3', memory_planner=passes.MemoryPlanner.PYPTO)
        spec = importlib.util.spec_from_file_location('independent_logical_plan_expected',
            D.parent / 'dynamic_attention_plan_20261017/check_oracle.py')
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        for columns in (1, 66, 1026, 2048, 2049):
            q = np.array([0, 5, 9, 13], np.int32)
            # 最大页列也成为真正合法候选；不能只测试被causal屏蔽的clamped gather。
            long_length = max(131077, columns * 128 + 5)
            lengths = np.array([long_length, 139, 0], np.int32)
            positions = np.array(list(range(long_length-5, long_length)) + [135,136,137,138] + [-77]*4, np.int64)
            topk = np.tile(np.arange(512, dtype=np.int32), (13,1))
            topk[:,:12] = [-2147483648, -1, 0, 31, 32, 63, 64,
                          columns*32-1, columns*32, 32767, 32768, 2147483647]
            original = np.tile(np.arange(long_length//32+2, dtype=np.int32)%23, (3,1))
            table = np.tile(np.arange(columns, dtype=np.int32)%19, (3,1))
            host_inputs = [topk, positions, q, lengths, original, table]
            inputs = [torch.from_numpy(a).to(device) for a in host_inputs]
            indices = torch.full((24,512), -777, dtype=torch.int32, device=device)
            bias = torch.full((24,640), 777., dtype=torch.float32, device=device)
            mask = torch.full((24,16), -777, dtype=torch.int32, device=device)
            pointers = [t.data_ptr() for t in inputs + [indices,bias,mask]]
            a_first = None
            for variant in ('A','B','A_again'):
                active_table, active_positions = table.copy(), positions.copy()
                if variant == 'B':
                    active_table = active_table[:,::-1].copy()
                    active_table[:,::5] = -1
                    if columns > 1: active_table[:,1] = 19
                    if columns > 2: active_table[:,2] = 2147483647
                    if columns > 3: active_table[:,3] = -2147483648
                    active_positions[3] += 2
                    active_positions[7] = -1
                inputs[1].copy_(torch.from_numpy(active_positions))
                inputs[5].copy_(torch.from_numpy(active_table))
                exp_i, exp_b, exp_m, owners = oracle.expected(
                    topk, active_positions, q, lengths, original, active_table, 23, 19)
                if columns <= 2048:
                    for token, owner in enumerate(owners):
                        if owner is not None:
                            request = owner[0]
                            for k, logical in enumerate(exp_i[token]):
                                if logical >= 0:
                                    # Python宽整数期望独立于被测的INT32 Vector位移/有效mask。
                                    exp_i[token,k] = int(active_table[request,int(logical)//32])*32+int(logical)%32
                plan_probe(*inputs, indices, bias, mask, config=config)
                torch.npu.synchronize()
                actual = [a.cpu().numpy() for a in (indices,bias,mask)]
                np.testing.assert_array_equal(actual[0][:13], exp_i)
                np.testing.assert_array_equal(actual[1][:13], exp_b)
                np.testing.assert_array_equal(actual[2][:13,:5], exp_m)
                assert np.all(actual[0][13:]==-777) and np.all(actual[1][13:]==777)
                assert np.all(actual[2][13:]==-777) and np.all(actual[2][:13,5:]==-777)
                assert pointers == [t.data_ptr() for t in inputs + [indices,bias,mask]]
                for tensor, expected in zip(inputs, (topk,active_positions,q,lengths,original,active_table)):
                    np.testing.assert_array_equal(tensor.cpu().numpy(), expected)
                if variant == 'A': a_first=[a.copy() for a in actual]
                if variant == 'A_again':
                    for a,b in zip(actual,a_first): np.testing.assert_array_equal(a,b)
                item = dict(columns=columns,variant=variant,physical_mode=columns<=2048,
                            passed=True,guards_and_inputs=True,addresses_stable=True)
                report['cases'].append(item)
                (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                print(json.dumps(item), flush=True)
        report['passed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
