"""只读归并既有阶段时戳与完整 L1；结果只写入本私有审计目录。

不导入设备库、不编译、不执行算子，不做 hash。阶段端点和原分 task 的
kernel 首尾分别统计；禁止把不同作业之间的差值解释为候选收益。
"""

import ast
import collections
import json
import statistics
import subprocess
from pathlib import Path


CANONICAL = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
D = CANONICAL / 'tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017'
PACKAGE = Path('vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd')
MATRIX = D / 'parallel_20261018/sync_h131072_b16_v1/source_matrix'
DIAG = D / 'diag_h131072_b16_output_groups_v1'
OUTPUT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def spread(values):
    return {k: round(f(values), 4) for k, f in (
        ('min', min), ('median', statistics.median), ('max', max)
    )}


def function_text_comparison():
    result = {}
    for name in ('indexer_score_stages.py', 'indexer_group_stages.py',
                 'attention_stream_stages.py', 'attention_plan_stages.py',
                 'cube_schedule.py', 'vector_schedule.py', 'policy.py'):
        old_path = DIAG / 'source' / name
        current = subprocess.check_output(
            ['git', 'show', '593c5d25:' + str(PACKAGE / name)],
            cwd=CANONICAL, text=True,
        )
        old_tree, current_tree = ast.parse(old_path.read_text()), ast.parse(current)
        def functions(tree):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        old, new = functions(old_tree), functions(current_tree)
        result[name] = {
            'diag_path': str(old_path), 'diag_size_bytes': old_path.stat().st_size,
            'baseline_path': str(CANONICAL / PACKAGE / name),
            'baseline_size_bytes': len(current.encode()),
            'same_ast': [key for key in old if key in new and old[key] == new[key]],
            'changed_ast': [key for key in old if key in new and old[key] != new[key]],
            'added': sorted(set(new) - set(old)),
        }
    return result


def phase_summary():
    rows = read_json(DIAG / 'phase_clock/phase_clock.json')
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['name']].append(row)
    result = {}
    for name, members in groups.items():
        result[name] = {'cores': [x['core'] for x in members]}
        for col in ('notify_end_us', 'interval_us', 'wait_us', 'other_us'):
            values = [x[col] for x in members if col in x]
            if values:
                result[name][col] = spread(values)
    active_plan = [x for x in groups['ATTN_PLAN'] if 24 <= x['core'] < 36]
    result['ATTN_PLAN_ACTUAL_12_OWNERS'] = {
        'cores': [x['core'] for x in active_plan],
        'other_us': spread([x['other_us'] for x in active_plan]),
    }
    result['source'] = str(DIAG / 'phase_clock/phase_clock.json')
    result['eligible_for_performance'] = False
    result['instrumentation_scope'] = (
        'wait_us只覆盖TWAIT/CSA_WAIT_LOCAL_GM；other含数学、Scalar、DMA、FFTS等待、'
        'fence/cache维护和插桩。不是纯计算，也不是等待开销的完整分解。'
    )
    return result


def task_windows():
    result = []
    for window in range(5):
        directory = MATRIX / f'l1_{window:02d}_dynamic_pto'
        raw = read_json(directory / 'chip_swimlane_records.json')
        tasks = {int(x['task_id']): x for x in read_json(directory / 'deps.json')['tasks']}
        names = read_json(directory / 'name_map.json')['callable_id_to_name']
        records = raw['aicore_tasks']
        frequency = raw['metadata']['clock_freq_hz'] / 1_000_000
        anchor = min(x[3] for x in records)
        by_name = collections.defaultdict(list)
        for core, task, sequence, start, end, setup in records:
            kernels = tasks[task]['kernel_ids']
            kernel = kernels[0] if core < 24 else kernels[1] if kernels[1] >= 0 else kernels[2]
            by_name[names[str(kernel)]].append({
                'core': core, 'task': task, 'sequence': sequence,
                'start_us': (start - anchor) / frequency,
                'end_us': (end - anchor) / frequency,
                'kernel_us': (end - start) / frequency,
            })
        spans = {}
        for label, fragment in (
            ('score', 'indexer_score_topk_native_pair'),
            ('topk_merge', 'indexer_topk_query_merge'),
            ('plan', 'csa_slots_build_valid_qk_plan'),
            ('attention', 'qk_pv'), ('oa', 'proj_a_mm'),
            ('ob', '_proj_b_mm'), ('hcpost', 'proj_b_act'),
        ):
            selected = [x for name, members in by_name.items() if fragment in name for x in members]
            spans[label] = {
                'records': len(selected),
                'first_kernel_start_us': min(x['start_us'] for x in selected),
                'last_kernel_end_us': max(x['end_us'] for x in selected),
                'kernel_duration_us': spread([x['kernel_us'] for x in selected]),
            }
        score, attention = spans['score'], spans['attention']
        total = (max(x[4] for x in records) - anchor) / frequency
        partition = {
            'prefix_to_first_score_us': score['first_kernel_start_us'],
            'score_envelope_us': score['last_kernel_end_us'] - score['first_kernel_start_us'],
            'score_to_attention_handoff_us': attention['first_kernel_start_us'] - score['last_kernel_end_us'],
            'attention_envelope_us': attention['last_kernel_end_us'] - attention['first_kernel_start_us'],
            'after_attention_us': total - attention['last_kernel_end_us'],
        }
        assert all(value >= 0 for value in partition.values())
        assert abs(sum(partition.values()) - total) < 1e-6
        result.append({
            'window': window, 'raw': str(directory / 'chip_swimlane_records.json'),
            'incore_span_us': total, 'spans': spans,
            'ordered_partition_us': {k: round(v, 4) for k, v in partition.items()},
            'scope': '原分task自己的完整L1分段；不是SPMD预测或跨作业消融',
        })
    return result


def main():
    matrix = read_json(MATRIX / 'result.json')
    stats = matrix['statistics_us']
    baseline, task = stats['spmd_reference']['median_us'], stats['dynamic_pto']['median_us']
    result = {
        'source_baseline': '593c5d25',
        'new_device_execution': False, 'new_compilation': False,
        'same_process_full_l1': {
            'source': str(MATRIX / 'result.json'),
            'original_task': stats['dynamic_pto'], 'spmd_baseline': stats['spmd_reference'],
            'spmd_slowdown_percent': (baseline / task - 1) * 100,
            'target_at_30_percent_reduction_us': task * 0.7,
            'required_baseline_reduction_us': baseline - task * 0.7,
        },
        'task_windows': task_windows(), 'phase_clock': phase_summary(),
        'diag_to_baseline_function_ast': function_text_comparison(),
        'workload_counts_not_timings': {
            'scope': 'H131072/B16/S6完整有效输入；动态Q/L实际确定有效范围',
            'query_groups': 16, 'leaves_per_group': 6, 'score_tiles_per_leaf': [6, 6, 6, 5, 5, 5],
            'score_tiles_total': 528, 'score_tiles_per_mix_group': [22] * 24,
            'attention_tiles_per_mix_group': [20] * 24,
            'score_aic_page_gather_upper_count': 528 * 32,
            'score_aiv_page_scale_gather_upper_count': 528 * 32 * 2,
            'attention_compressed_row_gather_upper_count': 96 * 512,
            'note': '地址guard可跳过非法项；次数不等于关键路径时间。'
        },
        'selected_new_candidate': None,
        'single_missing_observation': {
            'scope': '当前593默认的G6/balanced Score最慢MIX组内生产/消费服务轨迹',
            'need': '同次调用按leaf/step对应AIC SCORE_CONSUMED与AIV SCORE_READY等待、页地址/gather准备及矩阵/排序段；保留双槽ACK和排空。',
            'reason': '阶段端点无法辨别230us来自AIC生产受限、AIV排序反压、Scalar页循环或三者叠加。',
            'wait_semantics_unverified': '本审计未验证wait_flag_dev阻塞语义，等待独立A3核对；前后Scalar时钟是否涵盖等待需以其证据为准。撤回仅issue耗时的判断；不得为计时擅自添加fence或改FFTS顺序。',
            'not_a_request_to_run_now': True,
        },
    }
    (OUTPUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'output': str(OUTPUT / 'result.json'),
        'task_median_us': task, 'spmd_median_us': baseline,
        'task_partitions': [x['ordered_partition_us'] for x in result['task_windows']],
        'new_candidate': None,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
