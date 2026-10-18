# SPDX-License-Identifier: Apache-2.0
"""动态CSA投影分支的核组分工：让独立矩阵链在同一SPMD内并行推进。

整体思路与阅读顺序
    外层已完成HC、QA、Indexer压缩和query投影，并发布对应阶段。本文件先
    用24个AIC完成主压缩投影，然后将AIC分为两个互不重叠的集合：前R核负责
    head权重、query/compact Hadamard和KV；剩余24-R核负责主Q_B。它们没有
    相互等待的数学依赖，结束后各核回到原Score链。先读主函数的两分支，再看
    query Hadamard的四token数学块。R是区间策略的constexpr，不是batch白名单。

输入输出、动态范围与工作区
    normed是BF16[T,4096]，主Q_B仍读原pad128 INT8 QR和Native NZ权重；
    query Hadamard读取已有BF16[pad16(T)*64,128]视图。T由设备Tensor维度取得，
    compact行数与请求数量仍读本轮device描述符。所有输出使用原有GM缓冲，
    不增加Host回读、静态任务表或新的工作区。每个helper仍遍历本核负责的
    全部动态行/列块；核组分工固定不等于每核实际工作量固定。

分工与同步合同
    辅助分支的逻辑worker等于物理AIC编号；Q_B后部连续24-R个物理编号对
    24-R取模，恰好得到完整逻辑worker集合的一个循环置换。
    每阶段仍由全部24个AIC各发布一次：没有该阶段业务的核先发布空完成，
    真正producer必须写回全部业务后发布。原AIV的等待集合是新producer集合
    的超集，因此本候选保留其较保守的等待和缓存获取，不伪造数据就绪。
    head权重必须在等待Indexer RoPE/compact norm之前发布，否则AIV在权重
    归约处等待、AIC又等待后面的norm，会形成环。Q_B分支在计算前先发布其
    四个空阶段，避免辅助分支的AIV消费者被无工作核延迟。

数值与验证边界
    主压缩、权重、KV和Q_B复用原数学helper，只变执行位置和worker条带。
    query Hadamard每四token合为M256，保持每行完整K128和FP32输出；最后
    不足四token时使用实际有效行，不读取容量尾部的未初始化RoPE。四token
    对应64个head/每token及64KiB Left容量，不是某个测试shape。性能尚待
    完整同轮比较，默认策略不启用；不能把分支重叠直接当作整层性能收益。
"""

import pypto.language as pl

from .compressor_projection_stages import compressor_project_stage
from .indexer_compressor_output_stages import indexer_compressor_hadamard_stage
from .indexer_query_stages import indexer_weights_project_stage
from .kv_stages import kv_project_stage
from .phase_sync import publish_phase, wait_producer, wait_row_owners
from .query_projection_stages import QUANT_WEIGHT_LAYOUT, q_b_stage
from .runtime_abi import (
    B,
    CORES,
    GROUPS,
    ICPAD,
    IDX_HEAD_ROWS,
    IDX_ROPE_HEAD_ROWS,
    KV_ROWS,
    MPROJ,
    PHASE_IDX_HADAMARD,
    PHASE_IDX_NORM,
    PHASE_IDX_QUERY_HAD,
    PHASE_IDX_QUERY_ROPE,
    PHASE_KV_PROJECT,
    PHASE_MAIN_PROJECT,
    PHASE_META,
    PHASE_QB,
    PHASE_WEIGHTS_PROJECT,
    QPROJ_ROWS,
    QR_ROWS,
    T,
    VECTORS,
    WEIGHTS_PARTIAL_ROWS,
    WORDS,
)


@pl.jit.inline(auto_scope=False)
def _query_hadamard_four_tokens(
    query_rope: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """M256只合并独立行；每行的K128乘加、输出舍入和head顺序保持不变。"""
    rows = tokens * 64
    weight = hadamard[0:128, 0:128]
    # 每块至多四个token；末块仍以真实T裁剪GM读取和写回，不用pad容量当有效行。
    for block in pl.range(worker, (rows + 255) // 256, WORKERS):
        first = block * 256
        valid = pl.min(256, rows - first)
        source = pl.slice(query_rope, [256, 128], [first, 0], valid_shape=[valid, 128])
        result = pl.matmul(source, weight, out_dtype=pl.FP32)
        # 此helper使用Tensor级数学，assemble由lowering生成保留有效行的Tile store。
        query_had = pl.assemble(query_had, result, [first, 0])
    return query_had


@pl.jit.inline(auto_scope=False)
def projection_branches_stage(
    normed: pl.Tensor[[T, 4096], pl.BF16],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    main_value: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    main_score: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    weights_proj: pl.Tensor[[4096, 64], pl.BF16],
    weights_partial: pl.InOut[pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]],
    query_rope: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    qr_matmul: pl.Tensor[[QR_ROWS, 1024], pl.INT8],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
    q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_count: pl.Tensor[[1], pl.INT32],
    idx_normed: pl.Tensor[[ICPAD, 128], pl.BF16],
    idx_had: pl.InOut[pl.Tensor[[ICPAD, 128], pl.FP32]],
    wkv: pl.Tensor[[4096, 512], pl.BF16],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
    worker: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    BRANCH_GROUPS: pl.constexpr,
):
    """0<R<24；Host只选择R，动态行数及所有输入依赖仍留在核内。"""
    tokens = pl.tensor.dim(normed, 0)
    # 主压缩不依赖Indexer的RoPE、Hadamard或head权重。前移后，后续AIV
    # 主压缩链不会再等Q_B后的投影；此处仍保持原24核owner和所有K累加顺序。
    main_value, main_score = compressor_project_stage(
        normed, main_wkv, main_wgate, main_value, main_score, worker, GROUPS
    )
    signal = publish_phase(signal, PHASE_MAIN_PROJECT, worker, CORES, generation)
    if worker < BRANCH_GROUPS:
        # 本核从不写Q_B，不参与其真实producer集合；空发布不能代替后20核的发布。
        signal = publish_phase(signal, PHASE_QB, worker, CORES, generation)
        weights_partial = indexer_weights_project_stage(
            normed, weights_proj, weights_partial, worker, BRANCH_GROUPS
        )
        signal = publish_phase(signal, PHASE_WEIGHTS_PROJECT, worker, CORES, generation)
        # 必须先发weights再等RoPE/norm；AIV完成RoPE后可继续weights→pool→norm。
        query_rope_tasks = ((tokens + 7) // 8) * 16
        signal = wait_row_owners(
            signal, PHASE_IDX_QUERY_ROPE, CORES, generation,
            0, query_rope_tasks, 1, VECTORS, GROUPS, 1,
        )
        query_had = _query_hadamard_four_tokens(
            query_rope, hadamard, query_had, tokens, worker, BRANCH_GROUPS
        )
        signal = publish_phase(signal, PHASE_IDX_QUERY_HAD, worker, CORES, generation)
        planner = pl.min((tokens + 7) // 8, VECTORS - 1)
        signal = wait_producer(signal, PHASE_META, GROUPS + planner, CORES, generation)
        requests = pl.tensor.dim(idx_lengths, 0)
        signal = wait_row_owners(
            signal, PHASE_IDX_NORM, CORES, generation, 0, requests, 1, VECTORS, GROUPS, 1
        )
        compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
        idx_had = indexer_compressor_hadamard_stage(
            idx_normed, hadamard, idx_had, compact_count, worker, BRANCH_GROUPS
        )
        signal = publish_phase(signal, PHASE_IDX_HADAMARD, worker, CORES, generation)
        # KV只依赖normed，放在此处与另一核集的Q_B重叠；其AIV后处理不在
        # compact norm之前，因此不会反过来阻断上面的等待。
        kv_fp32 = kv_project_stage(normed, wkv, kv_fp32, worker, BRANCH_GROUPS)
        signal = publish_phase(signal, PHASE_KV_PROJECT, worker, CORES, generation)
    else:
        # 四阶段在本核都没有业务写入。先完成它们的空参与合同，让辅助分支
        # 的消费者只受到实际producer制约；不用等待本核长Q_B结束。
        signal = publish_phase(signal, PHASE_WEIGHTS_PROJECT, worker, CORES, generation)
        signal = publish_phase(signal, PHASE_IDX_QUERY_HAD, worker, CORES, generation)
        signal = publish_phase(signal, PHASE_IDX_HADAMARD, worker, CORES, generation)
        signal = publish_phase(signal, PHASE_KV_PROJECT, worker, CORES, generation)
        # 后部连续N个物理核对N取模是双射，既覆盖全部列块，也让NZ地址从
        # 非负表达式构成。消费者等待全部Q_B实际生产者，不依赖置换前的编号。
        qb_worker = worker % (GROUPS - BRANCH_GROUPS)
        q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS)
        signal = publish_phase(signal, PHASE_QB, worker, CORES, generation)
    return main_value, main_score, weights_partial, query_had, q_proj, idx_had, kv_fp32, signal
