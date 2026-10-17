# SPDX-License-Identifier: Apache-2.0
"""动态O投影/HCpost单MIX组合探针：不是完整CSA，不执行NPU。

阶段DAG：AIC O_A → AIV分组量化（含OB所需尾零）→ AIC O_B → AIV dequant/HCpost。
24 AIC和48 AIV各begin一次；阶段信号1/2/3/4分别对应上述四段，0保存代际。
当前组合先等待该阶段实际全部owner，建立无环依赖；没有逐tile槽复用或ACK。
输入packed heads、HC的post/comb/原始residual必须由外部保证已经就绪。

T来自原始residual实际行，PACK是8个组的合并行容量，MM/scale容量独立dynamic。
要求PACK%8=0、PACK/8>=T、MM>=ceil(T/96)*96、scale>=T；所有GM在外部绑定。
本探针固定RB96/N256数学tile，但T/B并不实值化，任意T都由核内循环决定尾块。
ND/NZ独立helper已各自编译，本次只组合正式NZ2路径；不重复之前的独立阶段验证。
所有中间GM按原入口返回别名保留，编译后必须核对真实store而不是只看返回参数。
"""

import pypto.language as pl
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.output_projection_a_stages import (
    o_projection_a_stage_nz, o_projection_group_quant_stage,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.output_projection_b_stages import (
    o_b_stage, o_dequant_hc_post_stage,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.phase_sync import (
    begin_invocation, publish_phase, wait_row_owners,
)

T = pl.dynamic("OUT_PIPE_T")
PACK = pl.dynamic("OUT_PIPE_PACK")
MM = pl.dynamic("OUT_PIPE_MM")
SCALE = pl.dynamic("OUT_PIPE_SCALE")
WORDS = pl.dynamic("OUT_PIPE_WORDS")
GROUPS = 24
VECTORS = 48
CORES = 72
RB = 96
PHASE_OA = 1
PHASE_QUANT = 2
PHASE_OB = 3
PHASE_OUT = 4

@pl.jit.incore
def cube(
    packed: pl.Tensor[[PACK,4096],pl.BF16],
    wo_a: pl.Tensor[[8,4096,1024],pl.BF16,pl.NZ],
    o_r: pl.InOut[pl.Tensor[[MM,8192],pl.FP32]],
    quant: pl.Tensor[[MM,8192],pl.INT8],
    scales: pl.Tensor[[8,SCALE],pl.FP32],
    wo_b: pl.Tensor[[8192,4096],pl.INT8,pl.NZ],
    partials: pl.InOut[pl.Tensor[[MM,32768],pl.INT32]],
    weight_scale: pl.Tensor[[4096],pl.FP32],
    residual: pl.Tensor[[T,16384],pl.BF16],
    post: pl.Tensor[[T,4],pl.FP32],
    comb: pl.Tensor[[T,16],pl.FP32],
    output: pl.Tensor[[T,16384],pl.BF16],
    signal: pl.InOut[pl.Tensor[[WORDS],pl.INT32]],
):
    o_r_gm = o_r
    partials_gm = partials
    signal_gm = signal
    worker = pl.tile.get_block_idx()
    signal, generation = begin_invocation(signal,worker)
    tokens = pl.tensor.dim(residual,0)
    o_r = o_projection_a_stage_nz(packed,wo_a,o_r,tokens,worker,GROUPS,256)
    signal = publish_phase(signal,PHASE_OA,worker,CORES,generation)
    # 量化逻辑任务=(group,32行块)，只等实际存在的owner；零T时没有等待。
    quant_tasks = 8*((tokens+31)//32)
    signal = wait_row_owners(signal,PHASE_QUANT,CORES,generation,0,quant_tasks,1,VECTORS,GROUPS,1)
    partials = o_b_stage(quant,wo_b,partials,tokens,worker,GROUPS,RB)
    signal = publish_phase(signal,PHASE_OB,worker,CORES,generation)
    return o_r_gm,partials_gm,signal_gm


@pl.jit.incore
def vector(
    packed: pl.Tensor[[PACK,4096],pl.BF16],
    wo_a: pl.Tensor[[8,4096,1024],pl.BF16,pl.NZ],
    o_r: pl.Tensor[[MM,8192],pl.FP32],
    quant: pl.InOut[pl.Tensor[[MM,8192],pl.INT8]],
    scales: pl.InOut[pl.Tensor[[8,SCALE],pl.FP32]],
    wo_b: pl.Tensor[[8192,4096],pl.INT8,pl.NZ],
    partials: pl.Tensor[[MM,32768],pl.INT32],
    weight_scale: pl.Tensor[[4096],pl.FP32],
    residual: pl.Tensor[[T,16384],pl.BF16],
    post: pl.Tensor[[T,4],pl.FP32],
    comb: pl.Tensor[[T,16],pl.FP32],
    output: pl.InOut[pl.Tensor[[T,16384],pl.BF16]],
    signal: pl.InOut[pl.Tensor[[WORDS],pl.INT32]],
):
    pl.func_attr({"split_aiv":True})
    quant_gm = quant
    scales_gm = scales
    output_gm = output
    signal_gm = signal
    worker = pl.tile.get_block_idx()*2+pl.tile.get_subblock_idx()
    core = GROUPS+worker
    signal, generation = begin_invocation(signal,core)
    tokens = pl.tensor.dim(residual,0)
    # O_A任务=(8组,ceil(T/128),4列块)，与正式N256数学owner完全相同。
    oa_tasks = 8*((tokens+127)//128)*4
    signal = wait_row_owners(signal,PHASE_OA,CORES,generation,0,oa_tasks,1,GROUPS,0,1)
    quant,scales = o_projection_group_quant_stage(o_r,quant,scales,tokens,worker,VECTORS,RB)
    signal = publish_phase(signal,PHASE_QUANT,core,CORES,generation)
    # NZ O_B按8组*8个D512列块分工；各owner内部遍历实际T，不按行块猜owner。
    signal = wait_row_owners(signal,PHASE_OB,CORES,generation,0,64,1,GROUPS,0,1)
    output = o_dequant_hc_post_stage(partials,scales,weight_scale,residual,post,comb,output,worker,VECTORS)
    signal = publish_phase(signal,PHASE_OUT,core,CORES,generation)
    return quant_gm,scales_gm,output_gm,signal_gm


@pl.jit.inline(auto_scope=False)
def group(
    packed: pl.Tensor[[PACK,4096],pl.BF16],
    wo_a: pl.Tensor[[8,4096,1024],pl.BF16,pl.NZ],
    o_r: pl.InOut[pl.Tensor[[MM,8192],pl.FP32]],
    quant: pl.InOut[pl.Tensor[[MM,8192],pl.INT8]],
    scales: pl.InOut[pl.Tensor[[8,SCALE],pl.FP32]],
    wo_b: pl.Tensor[[8192,4096],pl.INT8,pl.NZ],
    partials: pl.InOut[pl.Tensor[[MM,32768],pl.INT32]],
    weight_scale: pl.Tensor[[4096],pl.FP32],
    residual: pl.Tensor[[T,16384],pl.BF16],
    post: pl.Tensor[[T,4],pl.FP32],
    comb: pl.Tensor[[T,16],pl.FP32],
    output: pl.InOut[pl.Tensor[[T,16384],pl.BF16]],
    signal: pl.InOut[pl.Tensor[[WORDS],pl.INT32]],
):
    with pl.cluster(name_hint="dynamic_output_pipeline"):
        o_r,partials,signal = cube(packed,wo_a,o_r,quant,scales,wo_b,partials,weight_scale,residual,post,comb,output,signal)
        quant,scales,output,signal = vector(packed,wo_a,o_r,quant,scales,wo_b,partials,weight_scale,residual,post,comb,output,signal)
    return o_r,quant,scales,partials,output,signal


@pl.jit
def entry(
    packed: pl.Tensor[[PACK,4096],pl.BF16],
    wo_a: pl.Tensor[[8,4096,1024],pl.BF16,pl.NZ],
    o_r: pl.InOut[pl.Tensor[[MM,8192],pl.FP32]],
    quant: pl.InOut[pl.Tensor[[MM,8192],pl.INT8]],
    scales: pl.InOut[pl.Tensor[[8,SCALE],pl.FP32]],
    wo_b: pl.Tensor[[8192,4096],pl.INT8,pl.NZ],
    partials: pl.InOut[pl.Tensor[[MM,32768],pl.INT32]],
    weight_scale: pl.Tensor[[4096],pl.FP32],
    residual: pl.Tensor[[T,16384],pl.BF16],
    post: pl.Tensor[[T,4],pl.FP32],
    comb: pl.Tensor[[T,16],pl.FP32],
    output: pl.InOut[pl.Tensor[[T,16384],pl.BF16]],
    signal: pl.InOut[pl.Tensor[[WORDS],pl.INT32]],
):
    with pl.spmd(GROUPS,sync_start=True):
        o_r,quant,scales,partials,output,signal = group(packed,wo_a,o_r,quant,scales,wo_b,partials,weight_scale,residual,post,comb,output,signal)
    return o_r,quant,scales,partials,output,signal


