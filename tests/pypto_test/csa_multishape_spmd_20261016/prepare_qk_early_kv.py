# SPDX-License-Identifier: Apache-2.0
"""仅在CPU准备进程注入同步策略，不改变生产入口或默认catalog。"""

import argparse
import json
from pathlib import Path

import shape_specialization as shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base = json.loads((root/'b16_cached_row_deps_v1/specialization/bindings/h131072_b16_prepared.json').read_text())
    output = args.output.resolve()
    case, contract = shapes.build_variant(
        shapes.descriptors_from_roots(base['root_arguments']), max_seq_len=131078,
        output_root=output, case_name='h131072_b16', balanced_schedule=True,
        schedule_policy='earliest_ready', cost_profile=root/'b16_dynamic_l1_cost.json',
        qk_vector_indices=True, row_dependencies=True, qk_early_kv=True)
    report = {'case': str(case), 'audit': contract['qk_early_kv'], 'device_executed': False}
    (output/'early_kv_preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
