# 历史首次AST抽取脚本，只复现最初待修正源码，不是当前生产文件生成器。
# 后续INT8 full修正、恒假分支删除与中文说明由正式文件及full_v1/full_v2快照保存。
# SPDX-License-Identifier: Apache-2.0
"""一次性提取当前高层Cube/Vector数学，生产文件不导入本脚本或历史IR。

只替换请求几何、容量、显式worker与尾query/页地址保护；数学节点来自保存的
canonical源码。运行拒绝覆盖生产文件，后续API修复直接在生产文件审阅并保留编译快照。
"""
import ast
import copy
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'reference_decode_indexer.py'
TARGET = HERE.parents[3] / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_score_stages.py'

def parse(s):
    return ast.parse(textwrap.dedent(s)).body

def assign_name(n):
    return n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else ''

def scalar(expr):
    return ast.parse(expr,mode='eval').body

class DynamicReads(ast.NodeTransformer):
    """合法query的visibility统一来自正式helper；先guard live，再读position。"""
    def visit_Assign(self, n):
        n = self.generic_visit(n)
        name = assign_name(n)
        if name in ['single_position','buf_position','stream_position']:
            return None
        prefixes = {
            'single_query_visible':('single_query + single_query_lane','single_batch'),
            'buf_query_visible':('buf_query + buf_query_lane','buf_batch_idx'),
            'stream_visible':('buf_query + stream_lane','buf_batch_idx'),
        }
        if name in prefixes:
            q,r = prefixes[name]
            return parse(f'{name}, _causal = indexer_query_visibility(bounds, kv_seq_lens, position_ids, {r}, {q})')[0]
        return n

    def visit_For(self, n):
        n = self.generic_visit(n)
        var = ast.unparse(n.target)
        if var in ['buf_query_lane','stream_lane']:
            n.body = [ast.If(test=scalar(f'{var} < buf_live'),body=n.body,orelse=[])]
        elif var == 'single_local':
            # 仅消费原始query位置或发布真实query根的循环需要guard；分数FIFO消费不能跳过。
            split = next((i for i,s in enumerate(n.body) if assign_name(s)=='single_query_lane'),None)
            if split is not None:
                n.body = n.body[:split+1] + [ast.If(test=scalar('single_query_lane < buf_live'),body=n.body[split+1:],orelse=[])]
        return n


def guard_key_pages(body):
    """保留原safe_page映射；页表列和物理页guard必须在对应读取前。"""
    class Guard(ast.NodeTransformer):
        def visit_For(self,n):
            n=self.generic_visit(n)
            var=ast.unparse(n.target)
            choices={
                'buf_prime_key_page':('buf_prime_safe_page','buf_prime_physical_page'),
                'buf_key_page':('buf_safe_page','buf_physical_page'),
            }
            if var not in choices:return n
            safe,physical=choices[var]
            p=next(i for i,s in enumerate(n.body) if assign_name(s)==physical)
            # 原max(page,0)会把负页误读为page0；这里只安全跳过非法页，不定义非法metadata数学。
            n.body[p]=parse(f'{physical} = pl.cast(pl.read(idx_block_table, [buf_batch_idx, {safe}]), pl.INDEX)')[0]
            tail=ast.If(test=scalar(f'{physical} >= 0 and {physical} < pl.tensor.dim(idx_native_kv_cache, 0)'),body=n.body[p+1:],orelse=[])
            n.body=n.body[:p]+[ast.If(test=scalar(f'{safe} >= 0 and {safe} < pl.tensor.dim(idx_block_table, 1)'),body=[n.body[p],tail],orelse=[])]
            return n
    return [Guard().visit(n) for n in body]


def guard_scale_pages(body):
    class Guard(ast.NodeTransformer):
        def visit_Assign(self,n):
            n=self.generic_visit(n)
            name=assign_name(n)
            if name in ['single_scale_bytes','buf_scale_bytes'] and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='pl.tile.create':
                n.value.func=scalar('pl.tile.full');n.value.keywords.append(ast.keyword(arg='value',value=ast.Constant(0)))
            return n
        def visit_For(self,n):
            n=self.generic_visit(n)
            var=ast.unparse(n.target)
            if var not in ['single_scale_page','buf_scale_page']:return n
            if var=='single_scale_page':safe,physical,request='single_scale_logical','single_scale_physical','single_batch'
            else:safe,physical,request='buf_scale_safe_page','buf_scale_physical_page','buf_batch_idx'
            p=next(i for i,s in enumerate(n.body) if assign_name(s)==physical)
            n.body[p]=parse(f'{physical}=pl.cast(pl.read(idx_block_table,[{request},{safe}]),pl.INDEX)')[0]
            tail=ast.If(test=scalar(f'{physical} >= 0 and {physical} < pl.tensor.dim(idx_native_kv_cache,0)'),body=n.body[p+1:],orelse=[])
            n.body=n.body[:p]+[ast.If(test=scalar(f'{safe}>=0 and {safe}<pl.tensor.dim(idx_block_table,1)'),body=[n.body[p],tail],orelse=[])]
            return n
    return [Guard().visit(n) for n in body]


def render(body,indent=0):
    module=ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]))
    return textwrap.indent(ast.unparse(module), ' '*indent)

old=ast.parse(SOURCE.read_text())
f=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='indexer_score_topk_native_cube')
sp=next(n for n in f.body if isinstance(n,ast.With))
cube_loop=sp.body[10]
cube_if=next(n for n in cube_loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='buf_logical_begin < buf_visible_count')
cube_body=copy.deepcopy(cube_if.body)
# 查询真实行与WS有效K两端同时收窄，不能仅限制第一处TLOAD。
class TailQuery(ast.NodeTransformer):
    def visit_Assign(self,n):
        n=self.generic_visit(n);name=assign_name(n)
        if name=='buf_query_vector':
            n.value.keywords.append(ast.keyword(arg='valid_shape',value=scalar('[buf_live*IDX_N_HEADS,IDX_HEAD_DIM]')))
        if name=='buf_coefficients_l1':
            n.value.args[1]=scalar('[buf_group*NATIVE_QLI_WEIGHT_ROWS,0]')
            n.value.keywords.append(ast.keyword(arg='valid_shape',value=scalar('[NATIVE_QLI_WEIGHT_ROWS,buf_live*IDX_N_HEADS]')))
        if name in ['buf_previous_l1','buf_scores_l1'] and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='pl.tile.create':
            return [n,*parse(f'{name}=pl.set_validshape({name},buf_live*IDX_N_HEADS,qk_panel_cols)')]
        if name in ['buf_previous_pair','buf_last_pair']:
            return [n,*parse(f'{name}=pl.set_validshape({name},buf_live*IDX_N_HEADS,qk_panel_cols)')]
        return n
cube_body=[TailQuery().visit(n) for n in cube_body]
cube_body=[x for n in cube_body for x in (n if isinstance(n,list) else [n])]
cube_body=guard_key_pages(cube_body)
# AIV两个算法分支原数学整体保留，固定S/原query编号统一替换为动态计划。
vector_branch=sp.body[11].body[0]
balanced_loop=vector_branch.body[0]
balanced_if=next(n for n in balanced_loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='single_begin < single_visible')
plain_loop=vector_branch.orelse[0]
plain_if=next(n for n in plain_loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='buf_logical_begin < buf_visible_count')
balanced_body=copy.deepcopy(balanced_if.body)
plain_body=copy.deepcopy(plain_if.body)
# 原balanced外层内还有重复几何的旧if balance分支，在constexpr=False下不可达；直接移除。
class RemoveDeadBalance(ast.NodeTransformer):
    def visit_If(self,n):
        if ast.unparse(n.test)=='balance_leaves':return self.generic_visit(ast.Module(body=n.orelse,type_ignores=[])).body
        return self.generic_visit(n)
plain_body=[RemoveDeadBalance().visit(n) for n in plain_body]
plain_body=[x for n in plain_body for x in (n if isinstance(n,list) else [n])]
balanced_body=[DynamicReads().visit(n) for n in balanced_body]
plain_body=[DynamicReads().visit(n) for n in plain_body]
balanced_body=guard_scale_pages(balanced_body)
plain_body=guard_scale_pages(plain_body)
license='\n'.join(SOURCE.read_text().splitlines()[:7])+'\n'
header='''"""动态CSA Indexer的成对Cube/Vector score流水：请求内分组、两槽READY/ACK。

阅读顺序与范围
    indexer_score_cube_stage在AIC执行INT8 QK、FIX缩放/ReLU至FP16，再用
    FP16 head系数在Cube归约为FP32；indexer_score_vector_lane_stage由同组
    两AIV读取分数、乘Native key scale并生成叶Top512根。二者必须在一个
    MIX内并发推进，不能先等完整AIC阶段后才调用AIV。本模块不含short Vector
    score、最终query Top-K发布或完整CSA入口；不创建spmd/task/阶段通知。

动态布局和分组
    Q/L/positions与group_offsets/plan来自正式indexer_group_stages，T/B、
    每请求query数、历史和所有arena容量保留动态。组在单请求内切G=2/6，
    最后live=min(G,request_end-first)，不读下一请求或padding query来凑整组。
    AIC query INT8[HEAD_ROWS,128]按原token*64排列；系数FP16[COEFF_ROWS,G*64]
    每组16行。尾组限制query的有效M和系数/Score的有效K为live*64，物理tile
    仍为G*64；完整组保留原QK/WS顺序。coeff中失效query行已由生产者置零。
    cache沿Native INT8[pages,page_bytes]布局：4096B key后是64B FP16 scale。
    key0/key64是外层绑定的只读二维别名；page_bytes>=4160且64B对齐，不在
    incore内reshape整块GM。页表列和物理页先检查再读取；无效页保护不表示
    非法metadata具有已定义的正确Top-K数学，真实visible历史页必须合法。

策略、分核与工作区
    G、QK panel、score_tile768/1024、prefetch和balanced均为显式策略，
    必须与plan的G/balanced/roots配置一致，不能按T重新猜。常驻核组数固定24。
    所有角色用同一work_item/leaf_range计算core按24条带领取的(group,leaf)。
    score_transfer FP32[>=24*2*G,score_tile]每core双槽；仅AIC写、两AIV读。
    score_arena FP32[>=24*2*G,8192]是非balanced AIV私有半叶scratch，
    行号core*2*G+query_lane*2+lane，各自覆盖有效前缀后才排序，不读取旧尾。
    pair_arena FP32[>=T*64,1024]每query固定64行：两根布局写leaf*2+lane，
    一根布局写leaf；后继按同一plan和真实query visibility只合并有效根。
    balanced AIV各管G/2个完整query，按两个1024连续块排序再以前新后旧次序
    合并根，保留同分规则。未实际产出的arena容量尾部不能作为有效数据消费。

同步、资源与边界
    外层先等query、系数、Native cache的key/scale和plan全部发布，并在同组
    AIC/双AIV设置同一FFTS workspace。READY事件由FIX发布；两AIV MTE2等待
    后读取自己的FIFO片段，再以MTE2各发ACK。AIC第2轮起等ACK后复用槽，
    每leaf末排空min(steps,2)。尾组中无有效query的AIV也必须读取FIFO并ACK，
    只能跳过query数学/根写回，否则AIC死锁。外层完成phase不是这个消费ACK。
    同时存活的QK/WS与两Key L0B沿原tile布局，不增加永久L1/L0 reserve。
    所有输入metadata在本次阶段中固定；工作区只有协议规定的owner写。
    返回别名不等于跨核完成。当前源码/CPU证据不能代替设备同步、精度和性能。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import FP32_NEG_INF
from .indexer_group_stages import (
    COEFF_COLS_DYN, COEFF_ROWS_DYN, OFFSET_ROWS_DYN, PLAN_WORDS, TOKENS_DYN,
    indexer_decode_group, indexer_leaf_range, indexer_query_visibility, indexer_work_item,
)
from .indexer_topk_stages import PAIR_ROWS_DYN, SCORE_ROWS_DYN, indexer_topk_half_leaf
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN

QUERY_HEAD_ROWS_DYN = pl.dynamic("CSA_SCORE_QUERY_HEAD_ROWS")
CACHE_PAGES_DYN = pl.dynamic("CSA_SCORE_CACHE_PAGES")
PAGE_BYTES_DYN = pl.dynamic("CSA_SCORE_PAGE_BYTES")
KEY0_ROWS_DYN = pl.dynamic("CSA_SCORE_KEY0_ROWS")
KEY64_ROWS_DYN = pl.dynamic("CSA_SCORE_KEY64_ROWS")
TABLE_COLUMNS_DYN = pl.dynamic("CSA_SCORE_TABLE_COLUMNS")
TRANSFER_ROWS_DYN = pl.dynamic("CSA_SCORE_TRANSFER_ROWS")
TRANSFER_COLS_DYN = pl.dynamic("CSA_SCORE_TRANSFER_COLS")
IDX_N_HEADS = 64
IDX_HEAD_DIM = 128
BLOCK_SIZE = 32
COMPRESS_RATIO = 4
INDEXER_KEY_BYTES = 4096
TOPK_CANDIDATES_PER_LEAF = 8192
TOPK_PAIR_WIDTH = 1024
TOPK_ROWS_PER_QUERY = 64
TOPK_SCORE_WORKERS = 24
BUFFERED_LONG_SCORE_TILE = 1024
BUFFERED_SCORE_SCALE = 1024.0
NATIVE_QLI_WEIGHT_ROWS = 16
'''
common='''    idx_native_kv_cache: pl.Tensor[[CACHE_PAGES_DYN, PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[TOKENS_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
'''
config='''    buf_worker: pl.Scalar[pl.INDEX],
    query_group_size: pl.constexpr,
    qk_panel_cols: pl.constexpr,
    score_tile: pl.constexpr,
    key_prefetch_panels: pl.constexpr,
    key_prefetch_to_l0: pl.constexpr,
    balance_leaves: pl.constexpr,
    SCORE_READY_EVENT: pl.constexpr,
    SCORE_CONSUMED_EVENT: pl.constexpr,
'''
prelude='''    native_page_bytes = pl.tensor.dim(idx_native_kv_cache, 1)
    groups = pl.cast(pl.read(plan, [0]), pl.INDEX)
    buf_max_leaves = pl.cast(pl.read(plan, [2]), pl.INDEX)
    buf_leaf_tiles = pl.cast(pl.read(plan, [3]), pl.INDEX)
    buf_extra_leaves = pl.cast(pl.read(plan, [4]), pl.INDEX)
    # 每个core独立读取同一device计划；空core不触发任何FFTS事件。
    for buf_item in pl.range(buf_worker, groups * buf_max_leaves, TOPK_SCORE_WORKERS):
        buf_group, buf_leaf = indexer_work_item(buf_item, groups, buf_max_leaves)
        buf_batch_idx, buf_query, buf_live, buf_visible_count = indexer_decode_group(
            bounds, kv_seq_lens, position_ids, offsets, buf_group, query_group_size
        )
        buf_logical_begin, buf_leaf_capacity, work_valid = indexer_leaf_range(
            buf_leaf, buf_leaf_tiles, buf_extra_leaves, buf_visible_count
        )
        if work_valid > 0:
            buf_cache_len = pl.cast(pl.read(kv_seq_lens, [buf_batch_idx]), pl.INDEX) // COMPRESS_RATIO
'''
cube='''

@pl.jit.inline(auto_scope=False)
def indexer_score_cube_stage(
    qr_hadamard_i8: pl.Tensor[[QUERY_HEAD_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    coefficients: pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16],
    idx_kv_cache: pl.Tensor[[KEY0_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_cache_shift64: pl.Tensor[[KEY64_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
'''+common+'''    buf_score_transfer: pl.InOut[pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32]],
'''+config+'''):
    """AIC生产FP32 score双槽；须与两个vector_lane同MIX并发，外层已设置FFTS。"""
'''+prelude+render(cube_body,12)+'''
    return buf_score_transfer
'''
vector='''

@pl.jit.inline(auto_scope=False)
def indexer_score_vector_lane_stage(
'''+common+'''    buf_score_transfer: pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32],
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    buf_score_lane: pl.Scalar[pl.INDEX],
'''+config+'''):
    """AIV lane0/1都完整消费同一core事件；只有真实query可写根，空尾lane仍ACK。"""
'''+prelude+'''            if balance_leaves:
                single_query = buf_query
                single_leaf = buf_leaf
                single_batch = buf_batch_idx
                single_cache_len = buf_cache_len
                single_visible = buf_visible_count
                single_begin = buf_logical_begin
                single_capacity = buf_leaf_capacity
'''+render(balanced_body,16)+'''
            else:
'''+render(plain_body,16)+'''
    return score_arena, pair_arena
'''
text=license+header+cube+vector
# AST unparse不保留原注释；在关键调度/math节点恢复中文原因说明。
notes={
 'for buf_score_step in pl.range(buf_score_iters):':'# 双槽最多允许两份未消费分数；第2轮起等两个AIV的MTE2 ACK后才重用。',
 'for buf_qk_panel in pl.unroll(score_tile // qk_panel_cols):':'# 固定panel顺序：本panel QK与前panel head归约同时存活，保留FP16舍入位置。',
 'for buf_score_drain in pl.range(pl.min(buf_score_iters, 2)):':'# 即使只有1轮也排空已发消息；下个leaf才能重新从slot0开始，不携带旧ACK。',
 'for single_step in pl.range(single_steps):':'# balanced每AIV持G/2条完整query；连续两轮凑2048候选，尾轮仍按实际valid排序。',
 'for buf_scale_page in pl.range(score_tile // 2 // BLOCK_SIZE):':'# scale读取独立于Cube结果，放在wait之前；无效页不解引用，输入合同仍须保证历史页合法。',
 'for single_scale_page in pl.range(score_tile // BLOCK_SIZE):':'# 两AIV读取同一候选区间的scale；页表与slab容量检查先于gather。',
 'single_new_root = pl.tile.extract(tiny_pairs, 0, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec)':'# 提取512对(score,index)，限制循环root存储寿命；保留原sort32→merge输入顺序。',
 'single_merged = pl.tile.mrgsort(single_new_root, single_old_root, tmp=single_merge_tmp)':'# 新2048段在旧根之前，保留同分时后到段优先，不交换两输入。',
 'for buf_query_lane in pl.unroll(query_group_size):':'# 每query单独causal/visible；不能把组内最大历史当成所有query的有效候选数。',
}
lines=[]
for line in text.splitlines():
 key=line.strip()
 if key in notes:lines.append(line[:len(line)-len(line.lstrip())]+notes[key])
 lines.append(line)
text='\n'.join(lines)+'\n'
ast.parse(text)
if TARGET.exists():raise SystemExit('拒绝覆盖已存在生产文件')
TARGET.write_text(text)
print(TARGET)
