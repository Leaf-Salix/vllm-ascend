#!/usr/bin/env python3
"""解释当前 RoPE/HAD 分工，枚举实际动态 T 的生产者集合。

这里以生产者的真实轮转循环逐项生成 unit→worker，再按 Hadamard 的
token轮转反查全部64heads；不先假定待证的16核结论。有限枚举用于发现
边界错误，任意T的正确性仍由README中的模数推导及整阶段发布合同保证。
仅标准库模型，不模拟A3流水、缓存或编译器，也不声称设备精度已通过。
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'source_reference'
BOUNDARIES = {0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 49,
              71, 72, 73, 95, 96, 97, 127, 128, 129, 168, 192, 383, 384, 385, 4096}


def wait_row_owners(first, count, tile, workers, base, stride):
    if count <= 0:
        return set()
    first_tile = first//tile
    end_tile = (first+count+tile-1)//tile
    return {base+((first_tile+i)%workers)*stride for i in range(min(end_tile-first_tile, workers))}


def one_case(tokens):
    blocks = (tokens+7)//8
    units = blocks*16
    producers = min(units, 48)
    # 独立逐项生成生产映射，核对没有重复/缺漏 unit。T=0无生产循环。
    unit_owner = [-1]*units
    for vector in range(producers):
        for unit in range(vector, units, producers):
            assert unit_owner[unit] == -1
            unit_owner[unit] = vector
    assert all(x >= 0 for x in unit_owner)
    block_owners = [set(unit_owner[b*16:(b+1)*16]) for b in range(blocks)]
    # 每unit写4heads，16unit覆盖64heads；完整8行与不足8尾行采用同一owner。
    actual = [set() for _ in range(24)]
    token_counts = [0]*24
    for token in range(tokens):
        cube = token%24
        actual[cube].update(24+x for x in block_owners[token//8])
        token_counts[cube] += 1
        # 两个合法GM别名必须将同一token/head映到同一线性元素地址。
        for head in (0, 3, 4, 31, 32, 60, 63):
            for dim in (0, 63, 64, 127):
                assert token*8192+head*128+dim == (token*64+head)*128+dim
        assert (token*64+64)*128 <= tokens*64*128
    old_total, new_total = 0, 0
    details = []
    for cube in range(24):
        old = wait_row_owners(0, units, 1, 48, 24, 1)
        new = wait_row_owners((cube//8)*16, 16, 1, 48, 24, 1) if cube < tokens else set()
        assert actual[cube] == new, (tokens, cube, actual[cube], new)
        assert new <= old
        assert all(24 <= x < 24+producers for x in new)
        assert token_counts[cube] == len(range(cube, tokens, 24))
        old_total += len(old)
        new_total += len(new)
        if tokens in BOUNDARIES:
            details.append(dict(cube=cube, token_count=token_counts[cube], owners=sorted(new)))
    return dict(tokens=tokens, rope_units=units, rope_workers=producers, active_had_workers=min(tokens,24),
                original_logical_waits=old_total, candidate_logical_waits=new_total,
                eliminated_logical_waits=old_total-new_total, owners=details)


def source_contract():
    src = (SOURCE/'indexer_query_stages.py').read_text()
    t = ast.parse(src)
    constants = {n.targets[0].id: n.value.value for n in t.body if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    assert constants['QH_MM_TILE'] == 64 and constants['DEQUANT_T_TILE'] == 8 and constants['DQ_ROPE_H_TILE'] == 4
    functions = {n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
    qh = functions['indexer_query_hadamard_stage']
    loops = [ast.unparse(n.iter) for n in ast.walk(qh) if isinstance(n,ast.For)]
    assert loops == ['pl.range(qh_worker, bs_heads // QH_MM_TILE, WORKERS)']
    rope = functions['indexer_query_dequant_rope_stage']
    rope_loops = [ast.unparse(n.iter) for n in ast.walk(rope) if isinstance(n,ast.For)]
    assert 'pl.range(dq_rope_worker, dq_rope_units, dq_rope_workers)' in rope_loops
    assert 'pl.range(dq_t0, bs)' in rope_loops
    assign = {n.targets[0].id:ast.unparse(n.value) for n in ast.walk(rope)
              if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert assign['dq_rope_workers'] == 'pl.min(dq_rope_units, WORKERS)'
    assert assign['dq_t0'] == 'dq_unit // (IDX_N_HEADS // DQ_ROPE_H_TILE) * DEQUANT_T_TILE'
    assert assign['hg'] == 'dq_unit % (IDX_N_HEADS // DQ_ROPE_H_TILE) * DQ_ROPE_H_TILE'
    vector = (SOURCE/'vector_schedule.py').read_text()
    assert vector.index('idx_query_rope = indexer_query_dequant_rope_stage(') < vector.index('publish_phase(signal, PHASE_IDX_QUERY_ROPE')
    assert vector.count('publish_phase(signal, PHASE_IDX_QUERY_ROPE') == 1
    binding = (SOURCE/'binding.py').read_text()
    assert 'idx_query_rope_heads=self.buffers["idx_query_rope"].view(p16 * 64, 128)' in binding
    assert '"idx_query_had": ((t * 64, 128), torch.float32)' in binding
    cube = (SOURCE/'cube_schedule.py').read_text()
    assert cube.index('PHASE_QR,') < cube.index('indexer_query_hadamard_stage(') < cube.index('q_proj = q_b_stage(')
    return dict(constant_values=constants, qh_loop=loops[0], rope_loops=rope_loops,
                producer_assignments={key:assign[key] for key in ('dq_rope_units','dq_rope_workers','dq_t0','hg')},
                producer_publishes_after_all_own_waves=True, flatten_alias_confirmed=True,
                had_output_separate_shape=["T*64",128], source_constants_used=[24,48,64,8,4])


def main():
    contract = source_contract()
    selected = []
    checks, consumed_tokens = 0, 0
    for tokens in range(0,4097):
        data = one_case(tokens)
        checks += 24
        consumed_tokens += tokens
        if tokens in BOUNDARIES:
            selected.append(data)
    result = dict(status='PASS', scope='T=0..4096逐项生产映射/消费集合枚举；未编译/未上卡。',
                  token_cases=4097, cube_cases=checks, consumed_token_instances=consumed_tokens,
                  no_missing_or_extra_candidate_owner=True, all_selected_owners_active=True,
                  original_wait_set_is_superset=True, source_contract=contract, boundaries=selected,
                  limitations=['CPU模型不验证A3缓存、流水、生成代码或性能。',
                               'T=0只核对空循环，不新增正式入口T=0支持承诺。',
                               '任意T依赖证明要求24AIC/48AIV、64heads、8行×4head和现行循环步长。',
                               '工作区遵守原合法独立分配合同，仅允许声明的rope/rope_heads连续view别名。'])
    (HERE/'owner_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_contract','boundaries')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
