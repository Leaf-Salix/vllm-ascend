#!/usr/bin/env python3
"""只读文本复核；不导入算子、PyPTO 或设备运行时，不生成编译产物。"""
import difflib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE.parents[2]
C = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
PAR = C / 'tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018'
CURRENT = C / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
FROZEN = W / 'tests/pypto_test/ultra_idx_query_quant_owner_20261018/source_reference'
NAMES = ['indexer_score_stages.py', 'indexer_topk_stages.py',
         'indexer_group_stages.py', 'indexer_publish_stages.py', 'indexer_query_stages.py']

rows = []
for name in NAMES + ['policy.py']:
    old, frozen, current = PAR / 'baseline_source' / name, FROZEN / name, CURRENT / name
    rows.append(dict(file=name, current=str(current), frozen_668=str(frozen), old_593=str(old),
                     current_bytes=current.stat().st_size,
                     current_equals_frozen=current.read_text() == frozen.read_text(),
                     frozen_equals_old=frozen.read_text() == old.read_text()))
assert all(r['current_equals_frozen'] for r in rows)
assert all(r['frozen_equals_old'] for r in rows if r['file'] != 'policy.py')
old_policy = (PAR / 'baseline_source/policy.py').read_text().splitlines(keepends=True)
now_policy = (FROZEN / 'policy.py').read_text().splitlines(keepends=True)
policy_diff = ''.join(difflib.unified_diff(old_policy, now_policy,
                                        fromfile='593/policy.py', tofile='668/policy.py'))
(HERE / 'policy_context.diff').write_text(policy_diff)
result = dict(decision='暂无有充分新证据的大额省工候选；本轮收口', source_comparison=rows,
              policy_difference='仅QA_M16_PARALLEL字段、签名映射、选择及说明；Indexer各策略参数未改',
              model_reused='parallel_20261018/indexer_work_reduction_readonly/evidence.json',
              compiler_execution=False, device_execution=False)
(HERE / 'evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
