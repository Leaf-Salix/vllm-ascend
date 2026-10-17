#!/usr/bin/env python3
"""只读 O_B→反量化/HCpost 流式消费审计；不导入设备库，不编译/上卡。

根据现有 NZ 分核枚举七档输出组、量化 owner 和最终逐 token owner；对比
行外层与组外层的精确 (token,group) 左折叠序列，计算逻辑常驻字节及额外 GM
流量。读取既有旧 SPMD 三轮时钟与同轮原 task 五窗 raw L1，不预测新性能。
输出工作量和数据依赖，不以静态容量估算替代完整后端内存规划。
"""

import argparse
import ast
import json
import re
import statistics
from pathlib import Path


SEVEN = [(131072, b) for b in (4, 8, 16, 24, 28)] + [(8192, b) for b in (16, 32)]
ROW_BYTES = 4096 * 4


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def ownership(tokens):
    """模型固有 8 组/4096 列不变；T 保留为输入，枚举不创建算子特化。"""
    quant_per_group = ceildiv(tokens, 32)
    ob_owners = [[(g * 8 + col) % 24 for col in range(8)] for g in range(8)]
    quant_owners = [[(g * quant_per_group + tile) % 48 for tile in range(quant_per_group)] for g in range(8)]
    workers = []
    covered = []
    for worker in range(48):
        rows = list(range(worker, tokens, 48))
        row_order = [(t, g) for t in rows for g in range(8)]
        group_order = [(t, g) for g in range(8) for t in rows]
        for t in rows:
            assert [g for row, g in group_order if row == t] == list(range(8))
        assert sorted(row_order) == sorted(group_order)
        covered += rows
        workers.append(dict(worker=worker, rows=rows,
                            quant_groups=[g for g, owners in enumerate(quant_owners) if worker in owners],
                            resident_accumulator_bytes=len(rows) * ROW_BYTES,
                            row_outer_first_wave_fold_terms=7 if rows else 0,
                            group_outer_pre_final_group_fold_terms=7 * len(rows)))
    assert sorted(covered) == list(range(tokens)) and len(set(covered)) == tokens
    active = min(tokens, 48)
    return dict(tokens=tokens, ob_group_owners=ob_owners, quant_group_owners=quant_owners, workers=workers,
                existing_ob_notifications=24, additional_group_ob_notifications=64,
                existing_ob_waits=48 * 24, first_row_only_group_waits=active * 64,
                incorrectly_repeated_every_row_waits=tokens * 64,
                max_rows_per_worker=ceildiv(tokens, 48),
                max_resident_accumulator_bytes=ceildiv(tokens, 48) * ROW_BYTES,
                first_wave_pre_final_fold_fraction=(active * 7 / (tokens * 8)) if tokens else 0,
                gm_scratch_capacity_bytes=tokens * ROW_BYTES,
                gm_scratch_additional_payload_bytes=14 * tokens * ROW_BYTES,
                phase_count_before=60, phase_count_with_dedicated_ob_groups=68,
                additional_signal_bytes=8 * 72 * 128 * 4)


def phase_clocks(directory):
    result = []
    for number in range(3):
        path = directory / f'clock_round_{number}_spmd_reference/phase_clock.json'
        records = json.loads(path.read_text())
        ob = [r for r in records if r['name'] == 'OB']
        out = [r for r in records if r['name'] == 'OUT']
        quant = [r for r in records if r['name'].startswith('O_QUANT_GROUP')]
        last_ob = max(r['notify_end_us'] for r in ob)
        last_out = max(r['notify_end_us'] for r in out)
        core_ends = {r['core']: r['notify_end_us'] for r in ob}
        result.append(dict(round=number, source=str(path), latest_ob_us=last_ob, latest_out_us=last_out,
                           latest_out_minus_latest_ob_us=last_out-last_ob,
                           quant_latest_us=max(r['notify_end_us'] for r in quant),
                           cohort_final_ob_end_us=[max(core_ends[c] for c in range(start,start+8)) for start in (0,8,16)],
                           cohort_last_group=[6,7,5],
                           each_core_ob=[dict(core=r['core'], notify_end_us=r['notify_end_us']) for r in ob]))
    return result


def original_task_windows(directory):
    result = []
    for index in range(5):
        window = directory / f'l1_{index:02d}_dynamic_pto'
        raw_path = window / 'chip_swimlane_records.json'
        raw = json.loads(raw_path.read_text())
        trace = json.loads((window / 'swimlane.json').read_text())
        # 名字仅从本窗口配置生成的 trace 取得；时间只采用 raw 第3/4列 incore。
        names = {x['args']['taskId']: x['name'].split('_spmd(')[0] for x in trace['traceEvents']
                 if x.get('ph') == 'X' and x.get('pid') == 4}
        frequency = raw['metadata']['clock_freq_hz']
        start = min(row[3] for row in raw['aicore_tasks'])
        selected = {}
        ob_tasks = {}
        for row in raw['aicore_tasks']:
            name = names[row[1]]
            if not any(tag in name for tag in ('proj_b_act_hc_post', 'proj_b_mm', 'proj_a_mm')):
                continue
            record = dict(core=row[0], start_us=(row[3]-start)*1e6/frequency,
                          end_us=(row[4]-start)*1e6/frequency, duration_us=(row[4]-row[3])*1e6/frequency)
            selected.setdefault(name, []).append(record)
            if 'proj_b_mm' in name:
                ob_tasks.setdefault(str(row[1]), []).append(record)
        result.append(dict(window=index, source=str(raw_path), tasks={name:dict(
            count=len(rows), earliest_start_us=min(r['start_us'] for r in rows),
            latest_end_us=max(r['end_us'] for r in rows),
            median_incore_duration_us=statistics.median(r['duration_us'] for r in rows))
            for name, rows in selected.items()}, ob_task_endpoint_without_assumed_group={task:dict(
                earliest_start_us=min(r['start_us'] for r in rows), latest_end_us=max(r['end_us'] for r in rows),
                members=len(rows)) for task, rows in ob_tasks.items()}))
    return result


def generated_tail(cpp):
    """只记录现成真正 vector_schedule 的尾部地址，不混入同文件其它函数常量。"""
    source = cpp.read_text()
    # 已存在的产物包含辅助函数；正式大函数从独立定义开始，不能全文件合并v编号。
    match = re.search(r'\n(?:__aiv__\s+)?(?:AICORE\s+)?void vector_schedule\(', source)
    if match is None:
        match = re.search(r'\n[^\n]*\bvector_schedule\([^\n]*\)\s*\{', source)
    if match is None:
        raise ValueError('未定位现成 vector_schedule 函数定义')
    function = source[match.start():]
    constants = {m[1]:int(m[2]) for m in re.finditer(r'const \w+ (v\d+) = (-?\d+);', function)}
    # 原角色数学文本的稳定源码注释，取权重scale开始到最后OUT发布前。
    start = function.index('// pto: %wb_scale_inline')
    end = function.index('// pto: %', function.index('label1296:', start))
    tail = function[start:end]
    wanted = ['wb_scale', 'acc_inline', 'p_g_inline', 'out_t_inline', 'out_bf16', 'x_f32_inline', 'res_']
    records = []
    pattern = re.compile(r'// pto: %([^\n]+)\n\s*(v\d+) = Tile<TileType::Vec, ([^,]+), (\d+), (\d+),[^\n]+\n\s*// pto: %[^\n]+\n\s*(v\d+) = \(uint64_t\) (v\d+);\n\s*TASSIGN\(\2, \6\);')
    for m in pattern.finditer(tail):
        if any(tag in m[1] for tag in wanted):
            records.append(dict(ir_name=m[1], dtype=m[3], rows=int(m[4]), cols=int(m[5]),
                                ub_offset=constants[m[7]]))
    return dict(source=str(cpp), excerpt_start_line=source[:match.start()+start].count('\n')+1,
                selected_tiles=records, notes='这些是既有尾段地址，不是新方案峰值；不同生命周期允许地址复用'), tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    n = args.repo/'tests/pypto_test/csa_multishape_spmd_20261016'
    parallel = n/'dynamic_full_pipeline_20261017/parallel_20261018'
    clock = parallel/'current_clock_h131072_b16_v1'
    cpp = clock/'build_output/decode_csa_spmd_jqlk5eey/kernels/aiv/vector_schedule.cpp'
    compiled, excerpt = generated_tail(cpp)
    shapes = [dict(history=h,batch=b,**ownership(b*6)) for h,b in SEVEN]
    # 有限所有权检查只证明不同T的行覆盖与每行g顺序，不把它称成设备数值测试。
    for tokens in range(257):
        ownership(tokens)
    p = args.repo/'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
    locations = {}
    for name in ['output_projection_b_stages.py','cube_schedule.py','vector_schedule.py','phase_sync.py']:
        tree=ast.parse((p/name).read_text())
        locations[str(p/name)]={f.name:[f.lineno,f.end_lineno] for f in tree.body if isinstance(f,ast.FunctionDef)}
    result=dict(analysis_only=True, compiled=False,npu_executed=False, timing_prediction=False,
                finite_ownership_checks=257,source_locations=locations,seven=shapes,
                old_spmd_clocks=phase_clocks(clock),
                original_task_windows=original_task_windows(parallel/'compact_priority_h131072_b16_v1/source_matrix'),
                generated_tail=compiled)
    args.output.mkdir(exist_ok=True,parents=True)
    (args.output/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (args.output/'existing_tail.cpp.txt').write_text(excerpt)
    print(json.dumps(dict(ownership_checks=257,shapes=len(shapes),existing_tail_tiles=len(compiled['selected_tiles']),
                          compiled=False,npu_executed=False),ensure_ascii=False))


if __name__=='__main__':
    main()
