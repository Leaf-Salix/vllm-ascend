# SPDX-License-Identifier: Apache-2.0
"""将在线oi状态本身拆成四份16KiB，避免DPS分段更新导致整状态保护复制。

不再对32头父状态做assemble；各16头状态使用原函数式加乘和loop yield。
保持32头softmax/m/l与所有归约，只改变独立oi元素的资源组织。
"""

import ast
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new)


def initial(prefix):
    return ''.join(f'    {prefix}_{side}{half} = pl.tile.full([FINAL_HEAD_TILE, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)\n'
                   for side in ('left', 'right') for half in (0, 1))


def merge():
    out = """                # oi状态自身拆为四份16KiB，每份独立采用原函数式更新。
                # 不修改32头父Tile，不让别名保护为每拍复制完整32头状态。
                # 每个元素仍按(old*alpha)+(pv*beta)运算；m/l与softmax不拆。
"""
    for half in (0, 1):
        for scalar in ('alpha','beta'):
            out += f"""                merge_{scalar}{half} = pl.reshape(
                    pl.tile.extract(pl.reshape({scalar}, [1, H // 2]), 0, {half} * FINAL_HEAD_TILE,
                                    [1, FINAL_HEAD_TILE], target_memory=pl.MemorySpace.Vec),
                    [FINAL_HEAD_TILE, 1],
                )
"""
        for side,col in [('left','0'),('right','HEAD_DIM // 2')]:
            out += f"""                pv_{side}{half} = pl.load(
                    pv_transfer, [pv_transfer_row + qk_lane_head + {half} * FINAL_HEAD_TILE, {col}],
                    [FINAL_HEAD_TILE, HEAD_DIM // 2],
                )
                next_{side}{half} = pl.add(pl.row_expand_mul({side}{half}_iter, merge_alpha{half}),
                                          pl.row_expand_mul(pv_{side}{half}, merge_beta{half}))
"""
    return out


def main():
    base = HERE / 'source_attention_shared_kv'
    target = HERE / 'source_attention_state_halves'
    target.mkdir(exist_ok=False)
    for p in base.glob('*.py'):
        (target / p.name).write_bytes(p.read_bytes())
    path = target / 'attention_stream_stages.py'
    source = path.read_text()
    source = once(source, '''    running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
    running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
''', initial('running'))
    start = source.index('                pv_left = pl.load(')
    end = source.index('                m_valid, l_valid, left_valid, right_valid = pl.yield_', start)
    source = source[:start] + merge() + source[end:]
    source = once(source, '''                    m_left = pl.slice(left_valid, [FINAL_HEAD_TILE, HEAD_DIM // 2], [pub_local_head, 0])
                    m_right = pl.slice(right_valid, [FINAL_HEAD_TILE, HEAD_DIM // 2], [pub_local_head, 0])
''', '''                    # pub_h是固定模型头分块的unroll变量，编译期选择对应
                    # 16头状态；不创建完整32头父Tile或执行跨地址phi复制。
                    if pub_h == 0:
                        m_left = left0_valid
                        m_right = right0_valid
                    else:
                        m_left = left1_valid
                        m_right = right1_valid
''')
    old = '''                reset_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                reset_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
'''
    new = ''.join('            ' + line for line in initial('reset').splitlines(keepends=True))
    source = once(source, old, new)
    # 状态元组扩为m/l和四个oi半段；统计环仍是原最后一项，不漏任何分支回边。
    pairs = [
        ('running_left', 'running_right'), ('left_iter','right_iter'),
        ('next_left','next_right'), ('left_valid','right_valid'),
        ('reset_left','reset_right'), ('left_published','right_published'),
        ('left_after','right_after'),
    ]
    for left,right in pairs:
        if left.startswith('running_') or left.startswith('next_') or left.startswith('reset_'):
            names=[left+'0',left+'1',right+'0',right+'1']
        else:
            names=[left.replace('left_','left0_'),left.replace('left_','left1_'),
                   right.replace('right_','right0_'),right.replace('right_','right1_')]
        pattern = r'\b'+left+r'\s*,\s*'+right+r'\b'
        source,n=re.subn(pattern,', '.join(names),source)
        assert n>=1, (left,right)
    # 所有旧完整oi状态名称均应消失；拒绝留下遗漏的分支或隐式旧shape。
    tree=ast.parse(source)
    old_names={x for pair in pairs for x in pair}
    assert not any(isinstance(n,ast.Name) and n.id in old_names for n in ast.walk(tree))
    marker='    尚须完整CPU确认同址与分离，随后真机冻结检查和同轮完整L1决定采用。\n'
    assert source.count(marker)==1
    source=source.replace(marker,'''    32头父状态内DPS分段更新会产生两份32KiB保护复制，第二版仍超限。
    本版把oi状态本身分成四份16KiB，各自使用原函数式更新和loop yield，
    避免修改父状态的别名；32头softmax、m/l归约与原通知顺序全部保留。
'''+marker)
    path.write_text(source)
    report=dict(base=str(base), changed=['attention_stream_stages.py'],
                oi_state_tiles=4, oi_state_bytes_each=16384, total_oi_bytes=65536,
                row_reduce_unchanged=True, shared_kv_bytes=65536,
                formal_changed=False, device_validated=False)
    (HERE/'ATTENTION_STATE_HALVES_PREPARE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    main()
