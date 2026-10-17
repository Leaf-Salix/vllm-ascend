# SPDX-License-Identifier: Apache-2.0
"""独立CPU准备入口，不修改默认catalog、生产代码或设备测试驱动。"""

import argparse
import json
from pathlib import Path

import shape_specialization as shapes
from indexer_l1_key_pairs import use_key_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base = json.loads((root/'b16_cached_row_deps_v1/specialization/bindings/h131072_b16_prepared.json').read_text())
    output = args.output.resolve()
    candidate, audit = use_key_pairs(base, shapes.catalog(), output/'math_variants'/'indexer_key_pairs')
    original = shapes.catalog
    try:
        shapes.catalog = lambda: candidate
        case, _ = shapes.build_variant(
            shapes.descriptors_from_roots(base['root_arguments']), max_seq_len=131078,
            output_root=output, case_name='h131072_b16', balanced_schedule=True,
            schedule_policy='earliest_ready', cost_profile=root/'b16_dynamic_l1_cost.json',
            qk_vector_indices=True, row_dependencies=True)
    finally:
        shapes.catalog = original
    result = {'case': str(case), 'audit': audit, 'device_executed': False}
    (output/'key_pairs_preparation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
