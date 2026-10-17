# SPDX-License-Identifier: Apache-2.0
"""只读检查既有CPP与源码快照，不重新编译，不执行NPU。"""

import argparse
import ast
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def functions(path):
    tree = ast.parse(path.read_text())
    result = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                node.body = node.body[1:]
            result[node.name] = ast.dump(node)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert functions(SOURCE / 'indexer_group_stages.py') == functions(args.run / 'indexer_group_stages.py')
    rows = []
    for stage in ('geometry', 'coeff2', 'coeff6'):
        kernel = next((args.run / stage / 'kernels/aiv').glob('*.cpp'))
        text = kernel.read_text()
        dimensions = re.findall(r'// Extract dynamic dim: (\w+)', text)
        assert 'CSA_GEOMETRY_REQUESTS' in dimensions and 'CSA_GEOMETRY_QUERY_BOUNDS' in dimensions
        assert any(x.startswith('CSA_INDEX_GROUP_T') for x in dimensions)
        row = {'stage': stage, 'kernel': str(kernel), 'dynamic_dimensions': dimensions,
               'store_sites': text.count('TSTORE('), 'cast_sites': text.count('TCVT('),
               'multiply_sites': text.count('TMUL('), 'load_sites': text.count('TLOAD(')}
        if stage.startswith('coeff'):
            group_size = int(stage[-1])
            assert row['store_sites'] == 1
            assert row['load_sites'] == row['cast_sites'] == 2 * group_size
            assert row['multiply_sites'] == group_size
            assert 'CSA_INDEX_GROUP_COEFF_COLS' in dimensions
            assert 'CSA_INDEX_GROUP_SCALE_ROWS' in dimensions and 'CSA_INDEX_GROUP_WEIGHT_ROWS' in dimensions
        else:
            # Scalar写不会产生TSTORE；查看真实plan/offset/report指针与赋值。
            assert 'plan__ssa_v0_tensor' in text and 'offsets__ssa_v0_tensor' in text
            assert 'report__ssa_v0_tensor' in text
            assignments = re.findall(r'\(v(\d+)\)\[[^\]]+\] = ', text)
            assert assignments.count('2') >= 2  # offsets循环及末端B
            assert assignments.count('4') == 8  # 固定八项plan
            assert assignments.count('6') == 10  # 每item十个报告Scalar
            row['scalar_store_sites_by_argument'] = {i: assignments.count(i) for i in ('2', '4', '6')}
        rows.append(row)
    result = {'passed': True, 'function_AST_matches_compiled_snapshot': True, 'stages': rows,
              'device_execution': False, 'limits': 'ABI和指令存在不能证明实际A3精度或跨核同步'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': True, 'stages': [r['stage'] for r in rows]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
