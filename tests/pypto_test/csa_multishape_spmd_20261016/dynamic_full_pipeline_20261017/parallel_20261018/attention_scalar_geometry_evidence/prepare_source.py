"""从正式 66882634 冻结 33 份源，只增加 Attention 本核四 Scalar 几何缓存。

本工具只处理文本/AST，不导入 PyPTO、torch 或设备库。缓存属于一次核调用，
不生成 Host tiling、GM descriptor 或跨核共享状态；数学、地址保护和事件保留。
输出已存在时拒绝覆盖，防止运行中的候选或原始证据被意外替换。
"""

import ast
import difflib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1')
PREFIX = 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd'
BASE = '66882634'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'源文本定位不是唯一项: {before[:90]!r}')
    return text.replace(before, after, 1)


def main():
    reference = HERE / 'source_reference'
    candidate = HERE / 'source_candidate'
    if reference.exists() or candidate.exists():
        raise RuntimeError('已有冻结源；拒绝覆盖。')
    names = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', BASE, '--', PREFIX], cwd=REPO, text=True
    ).splitlines()
    names = [n for n in names if n.endswith('.py')]
    assert len(names) == 33
    reference.mkdir()
    candidate.mkdir()
    for name in names:
        body = subprocess.check_output(['git', 'show', f'{BASE}:{name}'], cwd=REPO)
        (reference / Path(name).name).write_bytes(body)
        (candidate / Path(name).name).write_bytes(body)

    path = candidate / 'attention_stream_stages.py'
    original = path.read_text()
    text = replace_once(original,
        '    attention_qk_pv_cube_stage 执行QK与滞后两拍的PV；',
        '''    本候选只缓存AIV本次调用当前gather query的四个Scalar：tag/owner/position/active。
    数学、Q/L/position/page的device输入、动态T/B、GM容量和全部事件合同不变。
    首个有效块惰性调用原geometry；同query其余有效块复用结果，active=0也缓存。
    四个标量加入原tick循环携带，不增加Tile、GM、根入参或phase，也不跨调用保存。
    它只缓存gather几何，不缓存较晚的softmax/PV query；PV末块发布不能清除此缓存。
    本地标量值复用依赖本次stage中Q/L/positions保持只读，不能替代原入口获取维护。
    本候选尚未PyPTO编译或真机验证，寄存器/分配/实际速度均未取得证据。

    attention_qk_pv_cube_stage 执行QK与滞后两拍的PV；''')
    text = replace_once(text,
        '''    # 多走三拍排空末块归并；循环携带的m/l/oi仅属于当前query，
    # 原始t_dim决定提前KV策略，不能用pad容量代替。
    for qk_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(
        qk_block_count + QK_PRE_LAUNCH + 1,
        init_values=(running_m, running_l, running_left, running_right),
    ):''',
        '''    # 每次进入helper重新初始化，图重放或下一次decode的Q/L变化不能沿用旧值。
    # INDEX保留完整位置范围，active保持原INT32；只增加Scalar循环值，未建立UB表。
    geometry_query = pl.cast(-1, pl.INDEX)
    geometry_owner = pl.cast(-1, pl.INDEX)
    geometry_position = pl.cast(0, pl.INDEX)
    geometry_active = pl.cast(0, pl.INT32)
    # 多走三拍排空末块归并；m/l/oi描述滞后三拍的PV，而geometry描述当前gather。
    # 两者生命周期不同：末query发布只重置数学状态，不能清掉已进入下一query的几何。
    # 原始t_dim决定提前KV策略，不能用pad容量代替。
    for qk_tick, (
        m_iter, l_iter, left_iter, right_iter,
        geometry_query_iter, geometry_owner_iter, geometry_position_iter, geometry_active_iter,
    ) in pl.range(
        qk_block_count + QK_PRE_LAUNCH + 1,
        init_values=(
            running_m, running_l, running_left, running_right,
            geometry_query, geometry_owner, geometry_position, geometry_active,
        ),
    ):''')
    text = replace_once(text,
        '''                qk_b, qk_pos, q_active = attention_query_geometry(
                    position_ids, bounds, lengths, valid_block_mask, qk_t, CACHE_QUERY_METADATA
                )''',
        '''                # 只在真实tick且原mask有效时访问几何。不能用sb==0作为刷新条件：
                # 首块可能无效，而该query后续压缩块仍有效；此处以实际query tag判断。
                # active=0同样更新tag，否则dummy/陈旧causal会在每块反复扫描。
                if qk_t != geometry_query_iter:
                    loaded_owner, loaded_position, loaded_active = attention_query_geometry(
                        position_ids, bounds, lengths, valid_block_mask, qk_t, CACHE_QUERY_METADATA
                    )
                    cached_query, cached_owner, cached_position, cached_active = pl.yield_(
                        qk_t, loaded_owner, loaded_position, loaded_active
                    )
                else:
                    cached_query, cached_owner, cached_position, cached_active = pl.yield_(
                        geometry_query_iter, geometry_owner_iter, geometry_position_iter, geometry_active_iter
                    )
                qk_b = cached_owner
                qk_pos = cached_position
                q_active = cached_active''')
    text = replace_once(text,
        '''                pl.store(qk_kv_half, [qk_kv_row + qk_lane_kv, 0], kv_transfer)
        # 大档先消费上一块SCORE通知''',
        '''                pl.store(qk_kv_half, [qk_kv_row + qk_lane_kv, 0], kv_transfer)
                geometry_valid_query, geometry_valid_owner, geometry_valid_position, geometry_valid_active = pl.yield_(
                    cached_query, cached_owner, cached_position, cached_active
                )
            else:
                # 全空query不扫描；上一query的tag可暂留，但下一有效query必须tag不等再刷新。
                geometry_valid_query, geometry_valid_owner, geometry_valid_position, geometry_valid_active = pl.yield_(
                    geometry_query_iter, geometry_owner_iter, geometry_position_iter, geometry_active_iter
                )
            geometry_after_query, geometry_after_owner, geometry_after_position, geometry_after_active = pl.yield_(
                geometry_valid_query, geometry_valid_owner, geometry_valid_position, geometry_valid_active
            )
        else:
            # 最后三拍只排空旧softmax/PV，qk_t可能已越过T，禁止读取Q/L/position。
            geometry_after_query, geometry_after_owner, geometry_after_position, geometry_after_active = pl.yield_(
                geometry_query_iter, geometry_owner_iter, geometry_position_iter, geometry_active_iter
            )
        # 大档先消费上一块SCORE通知''')
    text = replace_once(text,
        '''        running_m, running_l, running_left, running_right = pl.yield_(m_after, l_after, left_after, right_after)''',
        '''        # 几何从当前gather分支带到下一拍；不使用pv_t，也不跟随PV末块reset。
        running_m, running_l, running_left, running_right, geometry_query, geometry_owner, geometry_position, geometry_active = pl.yield_(
            m_after, l_after, left_after, right_after,
            geometry_after_query, geometry_after_owner, geometry_after_position, geometry_after_active,
        )''')
    ast.parse(text)
    path.write_text(text)
    changed = [f.name for f in candidate.iterdir() if f.read_bytes() != (reference / f.name).read_bytes()]
    assert changed == ['attention_stream_stages.py']
    (HERE / 'candidate.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), text.splitlines(True),
        fromfile='source_reference/attention_stream_stages.py',
        tofile='source_candidate/attention_stream_stages.py')))
    (HERE / 'SOURCE_PREPARATION.json').write_text(json.dumps({
        'baseline': subprocess.check_output(['git','rev-parse',BASE],cwd=REPO,text=True).strip(),
        'source_file_count': len(names), 'changed_files': changed,
        'source_origin': str(REPO), 'pypto_compile': False, 'device_execution': False,
    }, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'prepared': True, 'changed': changed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
