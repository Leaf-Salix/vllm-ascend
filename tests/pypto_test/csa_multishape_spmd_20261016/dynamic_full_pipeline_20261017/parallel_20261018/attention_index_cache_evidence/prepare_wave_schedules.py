# 历史 v1 接线生成器：后续显式 while/SSA 修复已直接进入 source_candidate，禁止直接重跑覆盖现有源码。
from pathlib import Path
from textwrap import dedent, indent
E=Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_candidate')
SRC=E/'source_candidate'
REF=E/'source_reference'
common_import='''from .wave_boundaries import next_wave_boundary
from .runtime_abi import WAVE_PHASE_BASE, WAVE_PHASE_STRIDE, WAVE_ROOT, WAVE_TOPK, WAVE_PLAN
from .wave_dependencies import wait_wave_plan_for_attention
'''
select=dedent('''
    # 只在既有G6/N64平衡长档且实际metadata能切成多wave时交错。
    # 此判定不匹配B/S测试点；无内部M8边界、空输入或其它策略保留原执行路径。
    wave_groups = pl.cast(pl.read(group_plan, [0]), pl.INDEX)
    use_interleave = pl.cast(0, pl.INT32)
    if QUERY_GROUP == 6 and QK_PANEL == 64 and SCORE_TILE == 1024 and PREFETCH == 1 and TO_L0 and BALANCED and ROOTS == 1:
        first_group_end, first_query_end = next_wave_boundary(group_offsets, idx_lengths, tokens, pl.cast(0, pl.INDEX))
        if first_group_end < wave_groups:
            use_interleave = pl.cast(1, pl.INT32)
''').lstrip('\n')
cube=dedent('''
    wave = pl.cast(0, pl.INDEX)
    group_begin = pl.cast(0, pl.INDEX)
    query_begin = pl.cast(0, pl.INDEX)
    previous_query_begin = pl.cast(0, pl.INDEX)
    previous_query_end = pl.cast(0, pl.INDEX)
    # 首wave只产Score；中间wave产当前Score并消费上一wave Attention。
    # 每核独立推进，不把不同核的本地round或叶游标强行对齐。
    while group_begin < wave_groups:
        group_end, query_end = next_wave_boundary(group_offsets, idx_lengths, tokens, group_begin)
        if wave > 0:
            signal = wait_wave_plan_for_attention(
                signal, WAVE_PHASE_BASE + (wave - 1) * WAVE_PHASE_STRIDE + WAVE_PLAN,
                generation, worker, previous_query_begin, previous_query_end,
            )
            if wave == 1:
                # QUERY的原head分核不变，只在首次真实Attention前获取一次。
                signal = wait_row_owners(signal, PHASE_QUERY, CORES, generation, 0, VECTORS, 1, VECTORS, GROUPS, 1)
        indexer_transfer, score_ring, pv_ring = interleave_cube_wave_g6(
            idx_query_i8, coefficients, idx_key0, idx_key64, idx_cache, indexer_table,
            positions, idx_lengths, idx_bounds, group_offsets, group_plan, indexer_transfer,
            q_heads, attention_mask, kv_ring, score_ring, probability_ring, pv_ring, ffts,
            worker, group_begin, group_end, previous_query_begin, previous_query_end,
            GROUPS, 0, 1, 2,
        )
        previous_query_begin = query_begin
        previous_query_end = query_end
        query_begin = query_end
        group_begin = group_end
        wave = wave + 1
    signal = publish_phase(signal, PHASE_SCORE_CUBE, worker, CORES, generation)
    # 尾wave没有后续Score；仍完整排空最后一段Attention的两个AIC尾拍。
    signal = wait_wave_plan_for_attention(
        signal, WAVE_PHASE_BASE + (wave - 1) * WAVE_PHASE_STRIDE + WAVE_PLAN,
        generation, worker, previous_query_begin, previous_query_end,
    )
    indexer_transfer, score_ring, pv_ring = interleave_cube_wave_g6(
        idx_query_i8, coefficients, idx_key0, idx_key64, idx_cache, indexer_table,
        positions, idx_lengths, idx_bounds, group_offsets, group_plan, indexer_transfer,
        q_heads, attention_mask, kv_ring, score_ring, probability_ring, pv_ring, ffts,
        worker, wave_groups, wave_groups, previous_query_begin, previous_query_end,
        GROUPS, 0, 1, 2,
    )
    signal = publish_phase(signal, PHASE_ATTN_CUBE, worker, CORES, generation)
''').lstrip('\n')
vector_call='''pair_arena, kv_ring, probability_ring, mi_ring, li_ring, packed, root_wrote = interleave_vector_wave_g6(
    idx_cache, indexer_table, positions, idx_lengths, idx_bounds, group_offsets, group_plan,
    indexer_transfer, pair_arena, cache_rows, main_cache, original_table, compressed_table,
    attention_indices, attention_bias, attention_mask, attn_sink, rope_cos, inverse_sin,
    kv_ring, score_ring, probability_ring, pv_ring, mi_ring, li_ring, packed, ffts,
    {group_begin}, {group_end}, previous_query_begin, previous_query_end,
    pl.tile.get_block_idx(), pl.tile.get_subblock_idx(), 0, 1, 2, ATTENTION_QUERY_METADATA,
)
'''
vector=dedent('''
    wave = pl.cast(0, pl.INDEX)
    group_begin = pl.cast(0, pl.INDEX)
    query_begin = pl.cast(0, pl.INDEX)
    previous_query_begin = pl.cast(0, pl.INDEX)
    previous_query_end = pl.cast(0, pl.INDEX)
    while group_begin < wave_groups:
        group_end, query_end = next_wave_boundary(group_offsets, idx_lengths, tokens, group_begin)
        if wave > 0:
            signal = wait_wave_plan_for_attention(
                signal, WAVE_PHASE_BASE + (wave - 1) * WAVE_PHASE_STRIDE + WAVE_PLAN,
                generation, pl.tile.get_block_idx(), previous_query_begin, previous_query_end,
            )
            if wave == 1:
                # 仅首次Attention取得原cache/rope依赖，Score首wave不额外前移等待。
                signal = wait_row_owners(signal, PHASE_ORIGINAL_CACHE, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1)
                signal = wait_row_owners(signal, PHASE_MAIN_OUTPUT, CORES, generation, 0, main_requests, 1, VECTORS, GROUPS, 1)
                signal = wait_row_owners(signal, PHASE_ROPE, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1)
''').lstrip('\n')
vector+=indent(vector_call.format(group_begin='group_begin',group_end='group_end'),'    ')
vector+=dedent('''
        # 每个root writer完成整个wave所有有效叶后才发布一次；空lane不伪报就绪。
        root_phase = WAVE_PHASE_BASE + wave * WAVE_PHASE_STRIDE + WAVE_ROOT
        topk_phase = WAVE_PHASE_BASE + wave * WAVE_PHASE_STRIDE + WAVE_TOPK
        plan_phase = WAVE_PHASE_BASE + wave * WAVE_PHASE_STRIDE + WAVE_PLAN
        if root_wrote > 0:
            signal = publish_phase(signal, root_phase, core, CORES, generation)
        signal = wait_wave_roots(
            signal, idx_bounds, idx_lengths, positions, group_offsets, group_plan,
            root_phase, generation, worker, query_begin, query_end,
        )
        topk_scores, topk_indices = indexer_pair_publish_range(
            idx_bounds, idx_lengths, positions, group_plan, pair_arena, topk_scores, topk_indices,
            query_begin, query_end, worker, VECTORS, ROOTS, BALANCED,
        )
        first_topk = query_begin + (worker - query_begin % VECTORS + VECTORS) % VECTORS
        if first_topk < query_end:
            signal = publish_phase(signal, topk_phase, core, CORES, generation)
        signal = wait_wave_topk_for_plan(signal, topk_phase, generation, worker, query_begin, query_end)
        attention_indices, attention_bias, attention_mask = attention_slot_plan_range(
            topk_indices, positions, idx_bounds, idx_lengths, original_table, compressed_table,
            attention_indices, attention_bias, attention_mask,
            pl.tensor.dim(cache_rows, 0) // 32, pl.tensor.dim(main_cache, 0) // 32,
            query_begin, query_end, worker, VECTORS, ATTENTION_QUERY_METADATA,
        )
        begin_plan_tile = query_begin // 8
        first_plan_tile = begin_plan_tile + (worker - begin_plan_tile % VECTORS + VECTORS) % VECTORS
        if first_plan_tile * 8 < query_end:
            signal = publish_phase(signal, plan_phase, core, CORES, generation)
        previous_query_begin = query_begin
        previous_query_end = query_end
        query_begin = query_end
        group_begin = group_end
        wave = wave + 1
''')
# 上面dedent保留的段落需仍处于while内。
# 明确从第一行的8空格改到局部block的4空格，再由外围统一增加8空格。
# dedent后是0层，因此修正刚拼接的root..wave这一段为while体。
marker='# 每个root writer'
pos=vector.index(marker)
vector=vector[:pos]+indent(vector[pos:],'    ')
vector+=dedent('''
    # 原整阶段通知保留供后续与诊断读取，其语义仍是全部业务输出已完成。
    signal = publish_phase(signal, PHASE_SCORE_VECTOR, core, CORES, generation)
    signal = publish_phase(signal, PHASE_TOPK, core, CORES, generation)
    signal = publish_phase(signal, PHASE_ATTN_PLAN, core, CORES, generation)
    signal = wait_wave_plan_for_attention(
        signal, WAVE_PHASE_BASE + (wave - 1) * WAVE_PHASE_STRIDE + WAVE_PLAN,
        generation, pl.tile.get_block_idx(), previous_query_begin, previous_query_end,
    )
''').lstrip('\n')
vector+=vector_call.format(group_begin='wave_groups',group_end='wave_groups')
vector+='signal = publish_phase(signal, PHASE_ATTN_VECTOR, core, CORES, generation)\n'
for name,begin,end,new,imports in [
 ('cube_schedule.py','    indexer_transfer = indexer_score_cube_stage(', '    signal = publish_phase(signal, PHASE_ATTN_CUBE, worker, CORES, generation)',cube,'from .interleave_cube_wave import interleave_cube_wave_g6\n'),
 ('vector_schedule.py','    score_arena, pair_arena = indexer_score_vector_lane_stage(', '    signal = publish_phase(signal, PHASE_ATTN_VECTOR, core, CORES, generation)',vector,'from .interleave_vector_wave import interleave_vector_wave_g6\nfrom .indexer_publish_stages import indexer_pair_publish_range\nfrom .attention_plan_stages import attention_slot_plan_range\nfrom .wave_dependencies import wait_wave_roots, wait_wave_topk_for_plan\n'),
]:
    source=(REF/name).read_text()
    lo=source.index(begin); hi=source.index(end,lo)+len(end)
    old=source[lo:hi]
    replacement=indent(select,'    ')+'    if use_interleave > 0:\n'+indent(new,'        ')+'    else:\n'+indent(old,'    ')
    source=source[:lo]+replacement+source[hi:]
    import_pos=source.index('from .attention_')
    source=source[:import_pos]+common_import+imports+source[import_pos:]
    (SRC/name).write_text(source)
    import ast
    ast.parse(source)
    print(name, 'Python syntax PASS')
(E/'prepare_wave_schedules.py').write_text(Path(__file__).read_text())
