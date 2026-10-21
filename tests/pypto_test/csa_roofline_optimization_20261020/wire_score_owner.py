from pathlib import Path
import ast,json,importlib.util,sys
here=Path('tests/pypto_test/csa_roofline_optimization_20261020');root=here/'source_score_owner_balance'
def change(name,old,new,count=1):
 p=root/name;s=p.read_text();assert s.count(old)==count,(name,s.count(old),old[:80]);s=s.replace(old,new);ast.parse(s);p.write_text(s)
change('policy.py','    split_score_rows: bool = False','    split_score_rows: bool = False\n    score_prefix_balance: bool = False')
change('policy.py','            "SPLIT_SCORE_ROWS": self.split_score_rows,','            "SPLIT_SCORE_ROWS": self.split_score_rows,\n            "SCORE_PREFIX_BALANCE": self.score_prefix_balance,')
change('policy.py','        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,','''        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,
        # 两个Q_B M128盒的资源区间；实际group/visible和是否存在工作差由META判断。
        # 不用B28白名单，不改变原叶数/数学；其它区间不分配映射或执行统计。
        score_prefix_balance=long_history and group == 6 and 128 < tokens <= 256,''')
change('binding.py','from .policy import ', 'from .score_owner_stages import SCORE_BALANCE_WORDS\nfrom .policy import ')
change('binding.py','(2 * b if self.policy.scale_page_coalescing else 0)','(2 * b if self.policy.scale_page_coalescing or self.policy.score_prefix_balance else 0) + (SCORE_BALANCE_WORDS if self.policy.score_prefix_balance else 0)')
change('entry.py','    SPLIT_SCORE_ROWS: pl.constexpr = False,','    SPLIT_SCORE_ROWS: pl.constexpr = False,\n    SCORE_PREFIX_BALANCE: pl.constexpr = False,',2)
change('entry.py','            SPLIT_SCORE_ROWS,\n','            SPLIT_SCORE_ROWS,\n            SCORE_PREFIX_BALANCE,\n',2)
change('entry.py','            PLAN_PHYSICAL_ROWS,\n        )','            PLAN_PHYSICAL_ROWS,\n            SCORE_PREFIX_BALANCE,\n        )')
change('cube_schedule.py','from .indexer_score_stages import indexer_score_cube_stage','from .indexer_score_stages import indexer_score_cube_stage\nfrom .score_owner_stages import SCORE_BALANCE_BRANCH_GROUPS, score_logical_worker')
change('cube_schedule.py','    SPLIT_SCORE_ROWS: pl.constexpr = False,','    SPLIT_SCORE_ROWS: pl.constexpr = False,\n    SCORE_PREFIX_BALANCE: pl.constexpr = False,')
change('cube_schedule.py','idx_normed, idx_had, wkv, kv_fp32, signal, worker, generation, 4,','idx_normed, idx_had, wkv, kv_fp32, signal, worker, generation, SCORE_BALANCE_BRANCH_GROUPS,')
change('cube_schedule.py','    indexer_transfer = indexer_score_cube_stage(','    score_worker = worker\n    if SCORE_PREFIX_BALANCE:\n        score_worker = score_logical_worker(group_offsets, pl.tensor.dim(idx_lengths, 0), tokens, worker)\n    indexer_transfer = indexer_score_cube_stage(')
change('cube_schedule.py','        group_plan,\n        indexer_transfer,\n        worker,','        group_plan,\n        indexer_transfer,\n        score_worker,')
change('vector_schedule.py','from .indexer_scale_layout_stages import indexer_scale_layout_plan_stage','from .indexer_scale_layout_stages import indexer_scale_layout_plan_stage\nfrom .score_owner_stages import score_balance_plan_stage, score_logical_worker')
change('vector_schedule.py','    PLAN_PHYSICAL_ROWS: pl.constexpr = False,','    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n    SCORE_PREFIX_BALANCE: pl.constexpr = False,')
change('vector_schedule.py','        # publish先排空业务写回并维护缓存，随后发单写者完成信号；不是atomic计数器。','''        if SCORE_PREFIX_BALANCE:
            group_offsets = score_balance_plan_stage(
                group_offsets, group_plan, pl.tensor.dim(idx_lengths, 0), tokens,
            )
        # publish先排空业务写回并维护缓存，随后发单写者完成信号；不是atomic计数器。''')
change('vector_schedule.py','    score_arena, pair_arena = indexer_score_vector_lane_stage(','    score_worker = pl.tile.get_block_idx()\n    if SCORE_PREFIX_BALANCE:\n        score_worker = score_logical_worker(group_offsets, pl.tensor.dim(idx_lengths, 0), tokens, score_worker)\n    score_arena, pair_arena = indexer_score_vector_lane_stage(')
change('vector_schedule.py','        pl.tile.get_subblock_idx(),\n        pl.tile.get_block_idx(),\n        QUERY_GROUP,','        pl.tile.get_subblock_idx(),\n        score_worker,\n        QUERY_GROUP,')
change('vector_schedule.py','        QUERY_GROUP,\n        EARLY_ATTENTION_GEOMETRY,\n    )','        QUERY_GROUP,\n        EARLY_ATTENTION_GEOMETRY,\n        SCORE_PREFIX_BALANCE,\n    )')
change('indexer_publish_stages.py','from .signals import ', 'from .score_owner_stages import wait_balanced_score_roots\nfrom .signals import ') if 'from .signals import ' in (root/'indexer_publish_stages.py').read_text() else change('indexer_publish_stages.py','import pypto.language as pl','import pypto.language as pl\n\nfrom .score_owner_stages import wait_balanced_score_roots')
change('indexer_publish_stages.py','    EARLY_ATTENTION_GEOMETRY: pl.constexpr = False,','    EARLY_ATTENTION_GEOMETRY: pl.constexpr = False,\n    SCORE_PREFIX_BALANCE: pl.constexpr = False,')
old='''            signals = wait_indexer_query_roots(
                signals, plan, group, local_query % QUERY_GROUP, root_count,
                generation, QUERY_GROUP, BALANCED,
            )'''
new='''            if SCORE_PREFIX_BALANCE:
                signals = wait_balanced_score_roots(
                    signals, offsets, plan, pl.tensor.dim(lengths, 0), tokens,
                    group, local_query % QUERY_GROUP, root_count, generation, QUERY_GROUP,
                )
            else:
                signals = wait_indexer_query_roots(
                    signals, plan, group, local_query % QUERY_GROUP, root_count,
                    generation, QUERY_GROUP, BALANCED,
                )'''
change('indexer_publish_stages.py',old,new)
spec=importlib.util.spec_from_file_location('score_owner_policy',root/'policy.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
for b in (24,28):
 controls=m.select_policy(tokens=b*6,requests=b,max_query_tokens=6,max_seq_len=131078).constexprs();(here/f'controls_score_owner_b{b}.json').write_text(json.dumps(controls,indent=2)+'\n')
changed=[p.name for p in root.glob('*.py') if not (here/'source_after_b1_half_v2'/p.name).exists() or p.read_bytes()!=(here/'source_after_b1_half_v2'/p.name).read_bytes()]
(here/'SCORE_OWNER_SOURCE.json').write_text(json.dumps({'reference':'source_after_b1_half_v2','candidate':'source_score_owner_balance','changed_files':sorted(changed),'device_validated':False,'performance_measured':False},ensure_ascii=False,indent=2)+'\n')
print(changed)
