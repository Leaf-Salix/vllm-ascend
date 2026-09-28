"""复用地址解析器，核查长档双query生成码；只比较相关Score执行二进制。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / 'csa_score_key_l1_20260928'
KERNEL = 'kernels/aic/indexer_score_topk_native_pair_1_aic.cpp'


def main(root=ROOT, kernel=KERNEL, changed_binary='incore_43_aic_indexer_score_topk_native_pair_1_aic_a2a3.bin',
         scope='长档双query/M128/N128；生成地址/静态等待，不代表设备收益。'):
    if 'COMPILE_PASS ' not in (root / 'compile.log').read_text():
        raise ValueError('完整CPU编译/load尚未通过')
    spec = importlib.util.spec_from_file_location('score_codegen', BASELINE / 'codegen.py')
    inspector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspector)
    result = {'scope': scope,
              'baseline': inspector.inspect(BASELINE / 'compiled' / kernel),
              'candidate': inspector.inspect(root / 'compiled' / kernel)}
    comparisons = []
    for source in sorted((BASELINE / 'compiled/cache').glob('*indexer_score_topk_native_pair*.bin')):
        if source.name == changed_binary:
            continue
        candidate = root / 'compiled/cache' / source.name
        comparisons.append({'kernel': source.name, 'baseline': str(source), 'candidate': str(candidate),
                            'binary_equal': source.read_bytes() == candidate.read_bytes()})
    result['unchanged_score_kernels'] = comparisons
    result['binary_scope'] = ('只比较其他四组Score AIC/AIV及受影响组AIV共9份执行bin；'
                              '未扫描仓库或输入hash。.o含路径等元数据，不作为执行码一致判据。')
    if len(comparisons) != 9 or not all(row['binary_equal'] for row in comparisons):
        raise ValueError('预期不受影响的Score执行核出现变化')
    for label in ('baseline', 'candidate'):
        print(label, {k: v for k, v in result[label].items() if k not in ('source', 'key_and_score_ops')})
    if result['candidate']['op_counts'] != {'key': 8, 'qk': 8, 'ws': 8}:
        raise ValueError('每N1024面板的Key/QK/WS次数改变')
    (root / 'codegen_evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
