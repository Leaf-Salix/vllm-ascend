# SPDX-License-Identifier: Apache-2.0
"""历史首次提取工具；只写本目录候选，未包含后续FFTS/Tile接口修正，禁止用作正式生成器。"""

import ast
import re
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_sparse_attn_csa.py"
NEW = HERE / "initial_extraction_candidate.py"
source = OLD.read_text()
license = source[: source.index('"""')]
header = '''"""动态 CSA Attention 的成对 AIC / 双 AIV 三槽流水与最终头向量发布。

整体数据流与阅读顺序
    attention_publish_heads_tile 是最终归一化、逆RoPE、BF16和8组packed写回；
    attention_qk_pv_cube_stage 执行QK与滞后两拍的PV；
    attention_qk_pv_vector_lane_stage 执行采KV、滞后一拍的局部softmax、
    滞后三拍的在线归并与发布。数学来自canonical decode_sparse_attn_csa，
    首版保留其5块/3槽/提前KV分支，不用全阶段barrier替代循环流水。

真实动态输入、布局与工作区
    q_flat为BF16[Q_ROWS,512]，Q_ROWS=64*T；两份cache直接使用Native连续
    BF16[pages*32,512]别名。positions INT64[T]、原始Q INT32[B+1]、L INT32[B]
    和两份INT32[B,columns]页表保持device输入，不用token//S或Host回读。
    plan必须来自同一批Q/L/positions/页表/页容量：indices INT32[PLAN_CAP,512]，
    bias FP32[PLAN_CAP,640]，mask INT32[PLAN_CAP,16]只读取前5列，PLAN_CAP>=T。
    sink FP32[64,1]，cos FP32[T,64]，逆RoPE signed_sin FP32[ROPE_CAP,64]。
    cos/sin/sink需覆盖所有实际T行且有限，包含约定写出零attention的dummy行。
    packed为BF16[PACK_ROWS,4096]，PACK_ROWS须整除8；P_PACK=PACK_ROWS//8>=T，
    第g组第t行地址为g*P_PACK+t。只写实际T，不触碰每组的pad。
    所有容量轴均为pl.dynamic；64头、512列、128候选块、5块和3槽是模型/流水
    常量，不是冻结batch/sequence。FFTS INT64[FFTS_WORDS]容量至少256。
    ring独立GM：KV BF16[KV_RING_ROWS,512]至少G*3*128行；score FP32[HEAD_RING_ROWS,128]、
    probability BF16[HEAD_RING_ROWS,128]、PV FP32[HEAD_RING_ROWS,512]、mi/li
    FP32[HEAD_RING_ROWS,1]各至少G*3*64行。描述符须连续，所有scratch互不非法alias。

核分工、事件与槽位生命周期
    调用者在同一个MIX内并发运行G个AIC和2G个AIV；core_id为组号，lane_id为0/1。
    query t归t%G；每组work跨全部query连续计数，slot=core_id*3+work%3。
    两AIV各管32heads/64个KV候选，不能先等待一个完整helper结束再启动另一个。
    EVENT_BASE..EVENT_BASE+3分别为KV_READY、SCORE_READY、PROB_READY、PV_READY；
    基址须让4个ID均落在0..13，14/15为编译器保留。生产者pipe沿用MTE3/FIX，
    消费者沿用MTE2，ffts_mode=2。调用者保证此期间其它阶段不复用这4个事件。
    两角色各自set_ffts到同一workspace；同组两AIV的半块写入都就绪后AIC才可读，
    AIC通知需被两个lane分别消费。这里只保留canonical协议，不另造GM flag。
    AIC第tick拍做QK(tick)、PV(tick-2)；AIV做gather(tick)、softmax(tick-1)、
    merge(tick-3)。槽序跨query延续，只有最后一次排空。有效块由同一plan判断，
    失效块在两侧同时跳过事件与ring访存，但query末块仍发布/清零运行状态。
    T>=144时必须先consume前一块SCORE，才发布本块KV；小档先做前一块softmax，
    再发布KV。不要交换这些语句：它们也限制生产者前进速度，保护三槽复用。
    阶段级phase没有逐槽ACK，不能拿它替代这些通知；此处没有创建task或新SPMD。

算术边界、安全与外层责任
    KV进一次L1，QK用转置view，PV复用原L1行；PV N128/K128保留交替累加器。
    AIV保持FP32块内softmax和在线m/l/左右oi顺序，probability及最终输出BF16均
    rint。最终归一化仍加入sink，RoPE读取未cast的FP32值，再执行BF16边界。
    gather再次检查Q/L活动、causal position、页表列和物理页上下界；inactive
    不访问cache。plan与metadata必须保持一致，不能一侧私自改mask而破坏事件对数。
    调用前等待Q/plan/原压缩cache及cos/sin可见；各组完成尾排空后，外层再发布
    packed-ready供O_A读取。FFTS生命周期/事件共存、真实A3同步、精度与性能
    仍须后续整合设备验证；CPU编译和事件模型都不能替代该证据。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import FLASH as M
from .attention_plan_stages import (
    COMPRESSED_COLUMNS_DYN, ORIGINAL_COLUMNS_DYN, PLAN_CAP_DYN, ROPE_CAP_DYN,
    T_DYN, QUERY_BOUNDS_DYN, REQUESTS_DYN,
)
from .request_geometry import query_request

Q_ROWS_DYN=pl.dynamic("CSA_ATTN_STREAM_Q_ROWS")
ORIGINAL_ROWS_DYN=pl.dynamic("CSA_ATTN_STREAM_ORIGINAL_ROWS")
COMPRESSED_ROWS_DYN=pl.dynamic("CSA_ATTN_STREAM_COMPRESSED_ROWS")
KV_RING_ROWS_DYN=pl.dynamic("CSA_ATTN_STREAM_KV_RING_ROWS")
HEAD_RING_ROWS_DYN=pl.dynamic("CSA_ATTN_STREAM_HEAD_RING_ROWS")
PACK_ROWS_DYN=pl.dynamic("CSA_ATTN_STREAM_PACK_ROWS")
FFTS_WORDS_DYN=pl.dynamic("CSA_ATTN_STREAM_FFTS_WORDS")
H=64
HEAD_DIM=512
ROPE_DIM=64
NOPE_DIM=448
WIN=128
CMP_TOPK=512
BLOCK_SIZE=32
ATTN_K_TILE=128
SPARSE_BLOCKS=5
QK_PRE_LAUNCH=2
QK_TRANSFER_SLOTS=3
PV_N_TILE=128
SWA_RUNS=5
FINAL_HEAD_TILE=16
O_GROUPS=8
HEADS_PER_GROUP=8
O_GROUP_IN=4096
PUBLISH_GROUPS=2
EARLY_KV_MIN_TOKENS=144
SOFTMAX_SCALE=M.softmax_scale
QK_KV_READY_EVENT=0
QK_SCORE_READY_EVENT=1
QK_PROB_READY_EVENT=2
QK_PV_READY_EVENT=3
'''

publish = '''

@pl.jit.inline(auto_scope=False)
def attention_publish_heads_tile(
    m_valid: pl.Tile[[H//2,1],pl.FP32],
    l_valid: pl.Tile[[H//2,1],pl.FP32],
    left_valid: pl.Tile[[H//2,HEAD_DIM//2],pl.FP32],
    right_valid: pl.Tile[[H//2,HEAD_DIM//2],pl.FP32],
    m_swap_idx: pl.Tile[[1,FINAL_HEAD_TILE*ROPE_DIM],pl.INT32],
    attn_sink_col: pl.Tensor[[H,1],pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN,ROPE_DIM],pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_CAP_DYN,ROPE_DIM],pl.FP32],
    o_packed_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN,O_GROUP_IN],pl.BF16]],
    pv_t: pl.Scalar[pl.INDEX],
    qk_lane_head: pl.Scalar[pl.INDEX],
):
    """消费本lane已合并32头的live Tile；不新增整T FP32中间结果。"""
    packed_stride=pl.tensor.dim(o_packed_heads,0)//O_GROUPS
'''
pub = source[
    source.index("                        m_gather_tmp =") : source.index("                        # Reset only")
]
pub = textwrap.indent(textwrap.dedent(pub), "    ").replace("T_PAD", "packed_stride")
publish += pub + "    return o_packed_heads\n"

cube = '''

@pl.jit.inline(auto_scope=False)
def attention_qk_pv_cube_stage(
    q_flat: pl.Tensor[[Q_ROWS_DYN,HEAD_DIM],pl.BF16],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN,16],pl.INT32],
    kv_transfer: pl.Tensor[[KV_RING_ROWS_DYN,HEAD_DIM],pl.BF16],
    score_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN,ATTN_K_TILE],pl.FP32]],
    probability_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN,ATTN_K_TILE],pl.BF16],
    pv_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN,HEAD_DIM],pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS_DYN],pl.INT64],
    core_id: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr=0,
):
    """AIC侧：QK领先PV两拍；必须与本组两个AIV helper并发。"""
    t_dim=pl.tensor.dim(q_flat,0)//H
    qk_core=core_id
    pl.system.set_ffts(ffts_workspace)
'''
cube_body = source[source.index("        qk_query_count =") : source.index("        for qk_aiv in pl.split_aiv")]
cube_body = textwrap.indent(textwrap.dedent(cube_body), "    ")
cube += cube_body + "    return score_transfer,pv_transfer\n"

vector = '''

@pl.jit.inline(auto_scope=False)
def attention_qk_pv_vector_lane_stage(
    ori_kv_flat: pl.Tensor[[ORIGINAL_ROWS_DYN,HEAD_DIM],pl.BF16],
    cmp_kv_flat: pl.Tensor[[COMPRESSED_ROWS_DYN,HEAD_DIM],pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN,ORIGINAL_COLUMNS_DYN],pl.INT32],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN,COMPRESSED_COLUMNS_DYN],pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN],pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN],pl.INT32],
    position_ids: pl.Tensor[[T_DYN],pl.INT64],
    cmp_sparse_indices: pl.Tensor[[PLAN_CAP_DYN,CMP_TOPK],pl.INT32],
    sparse_bias: pl.Tensor[[PLAN_CAP_DYN,640],pl.FP32],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN,16],pl.INT32],
    attn_sink_col: pl.Tensor[[H,1],pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN,ROPE_DIM],pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_CAP_DYN,ROPE_DIM],pl.FP32],
    kv_transfer: pl.InOut[pl.Tensor[[KV_RING_ROWS_DYN,HEAD_DIM],pl.BF16]],
    score_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN,ATTN_K_TILE],pl.FP32],
    probability_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN,ATTN_K_TILE],pl.BF16]],
    pv_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN,HEAD_DIM],pl.FP32],
    mi_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN,1],pl.FP32]],
    li_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN,1],pl.FP32]],
    o_packed_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN,O_GROUP_IN],pl.BF16]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS_DYN],pl.INT64],
    core_id: pl.Scalar[pl.INDEX],
    lane_id: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr=0,
):
    """双AIV各32头/64候选；同一mask和事件次序保证与AIC配对。"""
    t_dim=pl.tensor.dim(position_ids,0)
    qk_core=core_id
    qk_aiv=lane_id
    qk_query_count=(t_dim+CORE_GROUPS-1-qk_core)//CORE_GROUPS
    qk_block_count=qk_query_count*SPARSE_BLOCKS
'''
vbody = source[
    source.index(
        "            pl.system.set_ffts(ffts_workspace)", source.index("        for qk_aiv in pl.split_aiv")
    ) : source.index("\n    return o_packed_heads, qk_tid")
]
vbody = textwrap.dedent(vbody)
vbody = vbody.replace("    qk_b = qk_t // S\n", "")
old_gather = vbody[vbody.index("            if qk_sb == 0:") : vbody.index("            pl.store(qk_kv_half")]
new_gather = """            # mask是两角色共同事件谓词；额外地址guard只限制gather，不改变
            # 事件对数。合法调用必须使用与plan同一批Q/L/positions/页表。
            qk_b, q_begin, q_end, q_active = query_request(bounds, lengths, qk_t, t_dim)
            if q_active != 0:
                qk_pos = pl.cast(pl.read(position_ids, [qk_t]), pl.INDEX)
                q_length = pl.cast(pl.read(lengths, [qk_b]), pl.INDEX)
                if qk_pos == q_length - (q_end - q_begin) + qk_t - q_begin:
                    if qk_sb == 0:
                        qk_win_len = pl.min(qk_pos + 1, WIN)
                        qk_win_start = qk_pos - qk_win_len + 1
                        qk_head = qk_win_start % BLOCK_SIZE
                        qk_rows = pl.min(pl.max(qk_win_len - qk_lane_kv, 0), ATTN_K_TILE // 2)
                        # 最多五个页run，每个lane只采自己的64候选；尾run只读
                        # qk_hi-qk_lo行，先检查页表列再检查物理page容量。
                        for qk_run in pl.unroll(SWA_RUNS):
                            qk_lo = pl.max(qk_run * BLOCK_SIZE - qk_head - qk_lane_kv, 0)
                            qk_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head - qk_lane_kv, qk_rows)
                            if qk_hi > qk_lo:
                                qk_absolute = qk_win_start + qk_lane_kv + qk_lo
                                qk_column = qk_absolute // BLOCK_SIZE
                                if qk_column >= 0 and qk_column < pl.tensor.dim(ori_block_table, 1):
                                    qk_raw_page = pl.cast(pl.read(ori_block_table, [qk_b, qk_column]), pl.INDEX)
                                    if qk_raw_page >= 0 and qk_raw_page < pl.tensor.dim(ori_kv_flat, 0) // BLOCK_SIZE:
                                        qk_raw_row = qk_raw_page * BLOCK_SIZE + qk_absolute % BLOCK_SIZE
                                        qk_kv_half = pl.gather_row(
                                            qk_kv_half, ori_kv_flat, [qk_lo, 0], [qk_raw_row, 0],
                                            [ATTN_K_TILE // 2, HEAD_DIM], valid_shape=[qk_hi-qk_lo, HEAD_DIM],
                                        )
                    else:
                        # indices已经包含plan的causal/页mask，保留顺序并再做
                        # 地址上界保护；非法行留在本轮初始化的零KV中。
                        for qk_row in pl.range(ATTN_K_TILE // 2):
                            qk_cmp_k = (qk_sb - 1) * ATTN_K_TILE + qk_lane_kv + qk_row
                            if qk_cmp_k < CMP_TOPK:
                                qk_ridx = pl.cast(pl.read(cmp_sparse_indices, [qk_t, qk_cmp_k]), pl.INDEX)
                                if qk_ridx >= 0:
                                    qk_column = qk_ridx // BLOCK_SIZE
                                    if qk_column < pl.tensor.dim(cmp_block_table, 1):
                                        qk_page = pl.cast(pl.read(cmp_block_table, [qk_b, qk_column]), pl.INDEX)
                                        if qk_page >= 0 and qk_page < pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE:
                                            qk_src = qk_page * BLOCK_SIZE + qk_ridx % BLOCK_SIZE
                                            qk_kv_half = pl.gather_row(
                                                qk_kv_half, cmp_kv_flat, [qk_row, 0], [qk_src, 0], [1, HEAD_DIM]
                                            )
"""
vbody = vbody.replace(old_gather, new_gather)
start = vbody.index("            m_gather_tmp =")
end = vbody.index("            # Reset only", start)
vbody = (
    vbody[:start]
    + """            o_packed_heads = attention_publish_heads_tile(
                m_valid,l_valid,left_valid,right_valid,m_swap_idx,attn_sink_col,
                freqs_cos,rope_sin_signed,o_packed_heads,pv_t,qk_lane_head,
            )
"""
    + vbody[end:]
)
vector += (
    textwrap.indent(vbody, "    ")
    + "    return kv_transfer,probability_transfer,mi_transfer,li_transfer,o_packed_heads\n"
)
body = header + publish + cube + vector
body = body.replace("NUM_QK_CORES", "CORE_GROUPS")

body = re.sub(r"\bT\b", "t_dim", body)
for event in ("QK_KV_READY_EVENT", "QK_SCORE_READY_EVENT", "QK_PROB_READY_EVENT", "QK_PV_READY_EVENT"):
    body = body.replace("sync_wait(" + event, "sync_wait(EVENT_BASE + " + event).replace(
        "sync_set(\n                        " + event, "sync_set(\n                        EVENT_BASE + " + event
    )
# 事件调用的缩进不同，按源AST精确插入基址，不依赖空格。

parsed = ast.parse(body)
for call in sorted(
    (
        n
        for n in ast.walk(parsed)
        if isinstance(n, ast.Call) and ast.unparse(n.func) in ("pl.system.sync_set", "pl.system.sync_wait")
    ),
    key=lambda n: (n.lineno, n.col_offset),
    reverse=True,
):
    first = call.args[0]
    if isinstance(first, ast.Name) and first.id.startswith("QK_"):
        lines = body.splitlines(keepends=True)
        line = lines[first.lineno - 1]
        lines[first.lineno - 1] = line[: first.col_offset] + "EVENT_BASE + " + line[first.col_offset :]
        body = "".join(lines)
assert not ({n.id for n in ast.walk(ast.parse(body)) if isinstance(n, ast.Name)} & {"S", "T_PAD", "T"})
NEW.write_text(license + body)
