# SPDX-License-Identifier: Apache-2.0
"""核对Indexer WS加载后移是否仍保留与当前QK互不重叠的Right区域。"""

import argparse
import json
import re
from pathlib import Path

from qk_phase_clock import close_brace


def cpp_issue_order(stage, qk, ws):
    """追踪生成C++对应set/wait，区别文本重排与被同步强制串行。

    只分析同一展开score块内这两个pipe的直接约束，不能证明实际硬件重叠。
    事件编号的配对基于这一段无分支的两个指令间区间及最近set；事件在
    其他迭代的可达性/全核行为仍由完整设备精度验证覆盖。
    """
    def op_position(name, op):
        m = re.search(r'// pto: '+re.escape(name)+r'\s*;\s*Tile<[^\n;]+> (\w+) =', stage)
        if not m:
            raise ValueError(f'没有找到实际C++ Tile：{name}')
        instruction = re.search(r'\b'+op+r'\('+re.escape(m[1])+r',', stage)
        if not instruction:
            raise ValueError(f'没有找到实际C++指令：{name}/{op}')
        return instruction.start()

    def waits(begin, end, producer, consumer, dependency_start):
        result = []
        pattern = r'wait_flag\('+producer+r', '+consumer+r', (EVENT_ID\d+)\)'
        for match in re.finditer(pattern, stage[begin:end]):
            wait_pos = begin+match.start()
            sets = list(re.finditer(r'set_flag\('+producer+r', '+consumer+r', '+match[1]+r'\)', stage[:wait_pos]))
            if not sets:
                raise ValueError('区间内wait没有可定位的前置set')
            set_pos = sets[-1].start()
            result.append({'event': match[1], 'wait_line': stage.count('\n', 0, wait_pos)+1,
                           'set_line': stage.count('\n', 0, set_pos)+1,
                           'set_after_dependency': set_pos > dependency_start})
        return result

    result = []
    for panel in range(1, 16):
        compute = op_position(qk[panel]['acc_name'], 'TMATMUL')
        load = op_position(ws[panel-1]['rhs_name'], 'TEXTRACT')
        before, after = sorted((compute, load))
        whole_pipe = 'pipe_barrier(PIPE_ALL)' in stage[before:after]
        if load < compute:
            links = waits(load, compute, 'PIPE_MTE1', 'PIPE_M', load)
            dependency = 'qk_waits_for_ws_right_load'
        else:
            links = waits(compute, load, 'PIPE_M', 'PIPE_MTE1', compute)
            dependency = 'ws_right_load_waits_for_qk'
        result.append({'qk_panel': panel, 'ws_panel': panel-1,
                       'qk_cpp_line': stage.count('\n', 0, compute)+1,
                       'ws_load_cpp_line': stage.count('\n', 0, load)+1,
                       'source_order': 'load_then_qk' if load < compute else 'qk_then_load',
                       'dependency_direction': dependency, 'wait_set_links': links,
                       'pipe_all_between': whole_pipe,
                       'direct_serialization_found': whole_pipe or any(x['set_after_dependency'] for x in links)})
    return result


def inspect(directory):
    directory = Path(directory)
    ir = (directory/'csa_static_group.pto').read_text().split('func.func @csa_static_aiv', 1)[0]
    constants = {n: int(v) for n, v in re.findall(r'(%\w+) = arith.constant (-?\d+) : i64', ir)}
    tiles, qk, ws = {}, [], []
    for line_number, line in enumerate(ir.splitlines(), 1):
        m = re.search(r'(%stage_16_\w+) = pto.alloc_tile addr = (%\w+).*'
                      r'!pto.tile_buf<loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)', line)
        if m:
            name, addr, memory, dtype, rows, cols = m.groups()
            tiles[name] = {'address': constants[addr], 'memory': memory, 'dtype': dtype,
                           'shape': [int(rows), int(cols)]}
        if 'pto.tmatmul ins(' in line:
            output = re.search(r'outs\((%\w+)', line)[1]
            if output not in tiles:
                continue
            rhs = re.search(r'ins\(%\w+, (%\w+)', line)[1]
            row = {'rhs': tiles[rhs], 'acc': tiles[output], 'rhs_name': rhs,
                   'acc_name': output, 'line': line_number}
            if tiles[output]['shape'] == [384, 64] and tiles[output]['dtype'] == 'i32':
                qk.append(row)
            elif tiles[output]['shape'] == [16, 64] and tiles[output]['dtype'] == 'f32':
                ws.append(row)
    if len(qk) != 16 or len(ws) != 16:
        raise ValueError(f'原N64流水应16QK/16WS，实际{len(qk)}/{len(ws)}')

    def disjoint(a, a_size, b, b_size):
        return a + a_size <= b or b + b_size <= a

    pairs = []
    for panel in range(1, 16):
        a, b = qk[panel], ws[panel-1]
        if (a['rhs']['memory'], a['rhs']['shape'], b['rhs']['memory'], b['rhs']['shape']) != (
                'right', [128, 64], 'right', [384, 64]):
            raise ValueError('Right规格改变')
        pairs.append({'qk_panel': panel, 'ws_panel': panel-1,
                      'qk_right': a['rhs']['address'], 'ws_right': b['rhs']['address'],
                      'right_disjoint': disjoint(a['rhs']['address'], 8192, b['rhs']['address'], 49152),
                      'acc_disjoint': disjoint(a['acc']['address'], 98304, b['acc']['address'], 4096)})
    cpp = (directory/'csa_static_group.cpp').read_text().split('AICORE void csa_static_aiv(', 1)[0]
    end = cpp.index('// pto: %done_16__tile\n')
    opening = list(re.finditer(r'^  if \([^\n]+\) \{', cpp[:end], re.M))[-1]
    stage = cpp[opening.start():close_brace(cpp, opening.end()-1)+1]
    counts = {'matmul': stage.count('TMATMUL('), 'pipe_all': stage.count('pipe_barrier(PIPE_ALL)'),
              'm_to_mte1_wait': stage.count('wait_flag(PIPE_M, PIPE_MTE1'),
              'mte1_to_m_wait': stage.count('wait_flag(PIPE_MTE1, PIPE_M'),
              'set_flags': stage.count('set_flag('), 'wait_flags': stage.count('wait_flag(')}
    issue_order = cpp_issue_order(stage, qk, ws)
    return {'artifact': str(directory.resolve()), 'pairs': pairs, 'counts': counts,
            'cpp_issue_order': issue_order,
            'directly_serialized_pair_count': sum(p['direct_serialization_found'] for p in issue_order),
            'all_right_pairs_disjoint': all(p['right_disjoint'] for p in pairs),
            'all_acc_pairs_disjoint': all(p['acc_disjoint'] for p in pairs),
            'scope': '物理区域与静态同步位点；不是硬件重叠或性能证明'}, stage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result, source = inspect(args.artifact)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    args.output.with_suffix('.cpp.txt').write_text(source+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}, ensure_ascii=False))
