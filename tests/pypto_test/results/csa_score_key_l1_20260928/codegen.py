"""记录S6生成代码中的Key/Score地址和静态依赖，不代表设备stall计时。"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNEL = 'compiled/kernels/aic/indexer_score_topk_native_pair_aic.cpp'


def inspect(path):
    constants, tiles, operations = {}, {}, []
    source = path.read_text()
    for number, line in enumerate(source.splitlines(), 1):
        match = re.search(r'const (?:u?int64_t) (v\d+) = (-?\d+);', line)
        if match:
            constants[match[1]] = int(match[2])
        match = re.search(r'u?int(?:16|32|64)_t (v\d+) = (.*);', line)
        if match:
            references = re.findall(r'\bv\d+\b', match[2])
            if len(references) == 1 and references[0] in constants:
                if re.fullmatch(r'\(uint64_t\) v\d+|static_cast<uint(?:16|64)_t>\(v\d+\)', match[2]):
                    constants[match[1]] = constants[references[0]]
        match = re.search(r'Tile<TileType::(\w+), (\w+), (\d+), (\d+),.*?> (v\d+)', line)
        if match:
            tiles[match[5]] = {'space': match[1], 'dtype': match[2],
                               'shape': [int(match[3]), int(match[4])]}
        match = re.search(r'TASSIGN\((v\d+), (v\d+)\);', line)
        if match and match[1] in tiles and match[2] in constants:
            tiles[match[1]]['offset_bytes'] = constants[match[2]]
        match = re.match(r'\s*(TMOV|TMATMUL|TEXTRACT|TINSERT)(?:<.*>)?\((.*)\);', line)
        if match:
            variables = [value.strip() for value in match[2].split(',')]
            operations.append({'line': number, 'op': match[1],
                               'tiles': [dict(tiles[v]) for v in variables if v in tiles],
                               'extract_offsets': [constants.get(v) for v in variables[2:]]
                               if match[1] == 'TEXTRACT' else []})
    key = [op for op in operations if op['op'] in ('TMOV', 'TEXTRACT')
           and op['tiles'][0]['space'] == 'Right' and op['tiles'][0]['dtype'] == 'int8_t']
    score = [op for op in operations if op['op'] == 'TINSERT'
             and op['tiles'][0]['space'] == 'Mat' and op['tiles'][0]['dtype'] == 'half']
    qk = [op for op in operations if op['op'] == 'TMATMUL' and op['tiles'][0]['dtype'] == 'int32_t']
    ws = [op for op in operations if op['op'] == 'TMATMUL' and op['tiles'][0]['dtype'] == 'float']
    return {
        'source': str(path),
        'key_right_offsets': [op['tiles'][0]['offset_bytes'] for op in key],
        'key_mat_base_offsets': sorted({op['tiles'][1]['offset_bytes'] for op in key}),
        'score_mat_offsets': sorted({op['tiles'][0]['offset_bytes'] for op in score}),
        'ws_right_offsets': sorted({op['tiles'][2]['offset_bytes'] for op in ws}),
        'op_counts': {'key': len(key), 'qk': len(qk), 'ws': len(ws)},
        'mte1_fix_wait_count': source.count('wait_flag(PIPE_MTE1, PIPE_FIX,'),
        'mte1_fix_set_count': source.count('set_flag(PIPE_MTE1, PIPE_FIX,'),
        'key_and_score_ops': key + score,
    }


def main():
    result = {
        'scope': '每个N1024分数块的静态生成代码。Key/Score独立地址与等待计数不等于性能收益。',
        'failed_l0b_only': inspect(ROOT.parent / 'csa_score_key_prefetch_20260928' / KERNEL),
        'dedicated_key_l1': inspect(ROOT / KERNEL),
    }
    (ROOT / 'codegen_evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for label, value in result.items():
        if isinstance(value, dict):
            print(label, {k: v for k, v in value.items() if k not in ('source', 'key_and_score_ops')})


if __name__ == '__main__':
    main()
