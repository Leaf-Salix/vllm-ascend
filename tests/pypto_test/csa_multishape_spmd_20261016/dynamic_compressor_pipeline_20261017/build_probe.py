# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  # 源字符串是探针排版模板，实际probe.py独立格式化。
"""只生成探针重复的显式参数签名，不生成或冻结任何shape/math循环。"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = [
    ("main_x", "MT,4096", "BF16", False),
    ("main_wkv", "1024,4096", "BF16", False),
    ("main_wgate", "1024,4096", "BF16", False),
    ("main_value", "MPROJ,1024", "FP32", True),
    ("main_score", "MPROJ,1024", "FP32", True),
    ("main_state", "MSTATE_PAGES,MSTATE_ELEMS", "FP32", True),
    ("main_table", "MB,MTABLE_COLS", "INT32", False),
    ("main_ape", "4,1024", "FP32", False),
    ("main_positions", "MT", "INT64", False),
    ("main_bounds", "MQ", "INT32", False),
    ("main_lengths", "MB", "INT32", False),
    ("main_state_slots", "MT,2", "INT32", False),
    ("main_pooled", "MPOOL,512", "FP32", True),
    ("main_gamma", "512", "BF16", False),
    ("main_cos", "MC,64", "FP32", False),
    ("main_sin", "MC,64", "FP32", False),
    ("main_offsets", "MB", "INT32", True),
    ("main_count", "1", "INT32", True),
    ("main_slots", "MSLOTS,2", "INT32", False),
    ("main_normed", "MNORM,512", "FP32", True),
    ("main_debug", "MT,512", "FP32", True),
    ("main_cache", "MCACHE_ROWS,512", "BF16", True),
    ("idx_x", "IT,4096", "BF16", False),
    ("idx_wkv", "256,4096", "BF16", False),
    ("idx_wgate", "256,4096", "BF16", False),
    ("idx_value", "IPROJ,256", "FP32", True),
    ("idx_score", "IPROJ,256", "FP32", True),
    ("idx_state", "ISTATE_PAGES,ISTATE_ELEMS", "FP32", True),
    ("idx_table", "IB,ITABLE_COLS", "INT32", False),
    ("idx_ape", "4,256", "FP32", False),
    ("idx_positions", "IT", "INT64", False),
    ("idx_bounds", "IQ", "INT32", False),
    ("idx_lengths", "IB", "INT32", False),
    ("idx_state_slots", "IT,2", "INT32", False),
    ("idx_pooled", "IPOOL,128", "FP32", True),
    ("idx_gamma", "128", "BF16", False),
    ("idx_cos", "IC,64", "FP32", False),
    ("idx_sin", "IC,64", "FP32", False),
    ("idx_offsets", "IB", "INT32", True),
    ("idx_count", "1", "INT32", True),
    ("idx_slots", "ISLOTS,2", "INT32", False),
    ("idx_normed", "ICPAD,128", "BF16", True),
    ("hadamard", "128,128", "BF16", False),
    ("idx_had", "ICPAD,128", "FP32", True),
    ("idx_scales", "ISCALES,1", "FP32", True),
    ("idx_debug", "IT,128", "FP32", True),
    ("idx_cache", "ICACHE_PAGES,ICACHE_BYTES", "INT8", True),
    ("signal", "WORDS", "INT32", True),
]

CUBE_NAMES = [
    "main_x",
    "main_wkv",
    "main_wgate",
    "main_value",
    "main_score",
    "idx_x",
    "idx_wkv",
    "idx_wgate",
    "idx_value",
    "idx_score",
    "idx_lengths",
    "idx_count",
    "idx_normed",
    "hadamard",
    "idx_had",
    "signal",
]
VECTOR_NAMES = [
    name
    for name, _, _, _ in FIELDS
    if name not in {"main_x", "main_wkv", "main_wgate", "idx_x", "idx_wkv", "idx_wgate", "hadamard"}
]
CUBE_WRITES = ["main_value", "main_score", "idx_value", "idx_score", "idx_had", "signal"]
VECTOR_WRITES = [
    "main_state",
    "main_pooled",
    "main_offsets",
    "main_count",
    "main_normed",
    "main_debug",
    "main_cache",
    "idx_state",
    "idx_pooled",
    "idx_offsets",
    "idx_count",
    "idx_normed",
    "idx_scales",
    "idx_debug",
    "idx_cache",
    "signal",
]
ALL_WRITES = CUBE_WRITES[:-1] + VECTOR_WRITES

HEADER = '''# SPDX-License-Identifier: Apache-2.0
"""两路压缩链的源级动态单MIX组合探针，不是完整CSA。

范围与阅读顺序
    复用正式投影→请求池化→state提交→主RMS/RoPE/cache，以及Indexer紧凑
    RMS/RoPE→Hadamard→量化/key→串行scale。先读阶段常量，再读cube和vector，
    最后group/entry。没有HC、QA/QB、Indexer检索、Attention、输出投影或HCpost。
    build_probe.py只排版重复签名；所有数学/请求/分核循环均在正式动态helper内。

输入与工作区
    主/Indexer各自有独立T、Q、L、positions、投影pad、pooled和state/cache容量。
    x均为BF16[T,4096]，投影权重ND转置布局，Native norm权重保持BF16。
    state直接绑定Native二维连续别名，页内逻辑token数2；主cache为BF16
    [page*32,512]别名，Indexer cache为INT8[page,page_bytes>=4160]，不搬运。
    Q/L设备Tensor保持原样；offset/count由同一device扫描得到，两组互不覆盖。
    pool只保证活动请求行已写；norm按请求读取，Indexer norm初始化真实compact
    区间，Hadamard只读C_valid。所有容量轴保留pl.dynamic，不绑定测试样例。
    各组独立缓冲不得非法别名；主pad64、Indexer投影pad16、compact pad16
    容量及Native唯一写slot/只读共享prefix合同由入口保证。

阶段DAG与核分工
    24AIC：主投影并发布→Indexer投影并发布→等Idx元数据和全部请求norm owner
    →device读取实际C→Hadamard并发布。先发出全部投影，避免与AIV等待形成环。
    48AIV：两组metadata(worker0)→等主投影→主pool发布→等所有主pool历史读完
    →主state commit→主norm/cache→等Idx投影→Idxpool发布→等所有Idxpool读完
    →Idxstate commit→Idxnorm发布→等Hadamard→量化/key发布→worker0等全quant
    →单写者scale commit。phase只是通知，不能替代pool完成历史读取这一业务条件。
    初版保守等真实分块/请求owner，不用历史trace等待表，也不宣称性能最优。

信号与写回生命周期
    一个24核组MIX提供24AIC+48AIV。72核包括空核均且仅begin一次；phase0存代际，
    其余13阶段固定含义，signal首次清零且至少14*72*128个INT32，单流顺序复用。
    默认TWAIT来自正式phase_sync，发布排空本核流水并维护业务/信号缓存；等待后
    再读业务数据。只有worker0发布metadata/scale，其余阶段每个角色核发布一次。
    没有ACK，不复用临时槽；改变核数/phase协议须新清零workspace，失败后亦须重置。
    cluster返回原GM参数别名，避免条件分支返回值被误认为新缓冲；另审计生成CPP
    是否真的写回传入GM。CPU代码生成不证明硬件同步、精度、图重放或性能。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.compressor_projection_stages import compressor_project_stage, indexer_compressor_project_stage
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.compressor_pool_stages import compressor_pool_stage, indexer_compressor_pool_stage, compressor_state_commit_stage, indexer_compressor_state_commit_stage
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.compressor_output_stages import compressor_norm_rope_cache_stage
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_compressor_output_stages import indexer_compressor_norm_rope_stage, indexer_compressor_hadamard_stage
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_cache_stages import indexer_compact_quant_cache_stage, indexer_compact_scale_commit_stage
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.metadata_stages import compact_offsets_and_count_stage
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.phase_sync import begin_invocation, publish_phase, wait_producer, wait_row_owners

GROUPS = 24
VECTORS = 48
CORES = 72
PHASE_MAIN_PROJECT = 1
PHASE_IDX_PROJECT = 2
PHASE_MAIN_META = 3
PHASE_IDX_META = 4
PHASE_MAIN_POOL = 5
PHASE_MAIN_COMMIT = 6
PHASE_MAIN_OUTPUT = 7
PHASE_IDX_POOL = 8
PHASE_IDX_COMMIT = 9
PHASE_IDX_NORM = 10
PHASE_IDX_HADAMARD = 11
PHASE_IDX_QUANT = 12
PHASE_IDX_SCALE = 13
'''

CUBE_BODY = """    worker = pl.tile.get_block_idx()
    signal, generation = begin_invocation(signal, worker)
    main_value, main_score = compressor_project_stage(main_x, main_wkv, main_wgate, main_value, main_score, worker, GROUPS)
    signal = publish_phase(signal, PHASE_MAIN_PROJECT, worker, CORES, generation)
    idx_value, idx_score = indexer_compressor_project_stage(idx_x, idx_wkv, idx_wgate, idx_value, idx_score, worker, GROUPS)
    signal = publish_phase(signal, PHASE_IDX_PROJECT, worker, CORES, generation)
    # C来自metadata写入的device值；先等发布再读，不能用compact容量猜C。
    signal = wait_producer(signal, PHASE_IDX_META, GROUPS, CORES, generation)
    signal = wait_row_owners(signal, PHASE_IDX_NORM, CORES, generation, 0, pl.tensor.dim(idx_lengths,0), 1, VECTORS, GROUPS, 1)
    count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
    idx_had = indexer_compressor_hadamard_stage(idx_normed, hadamard, idx_had, count, worker, GROUPS)
    signal = publish_phase(signal, PHASE_IDX_HADAMARD, worker, CORES, generation)
"""

VECTOR_BODY = """    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = GROUPS + worker
    signal, generation = begin_invocation(signal, core)
    main_tokens = pl.tensor.dim(main_positions, 0)
    idx_tokens = pl.tensor.dim(idx_positions, 0)
    main_requests = pl.tensor.dim(main_lengths, 0)
    idx_requests = pl.tensor.dim(idx_lengths, 0)
    main_offsets, main_count = compact_offsets_and_count_stage(main_bounds, main_lengths, main_offsets, main_count, main_tokens, worker)
    idx_offsets, idx_count = compact_offsets_and_count_stage(idx_bounds, idx_lengths, idx_offsets, idx_count, idx_tokens, worker)
    if worker == 0:
        signal = publish_phase(signal, PHASE_MAIN_META, core, CORES, generation)
        signal = publish_phase(signal, PHASE_IDX_META, core, CORES, generation)
    # 主投影M64*N64展平后为ceil(T/64)*16个task，每个task按%24归属AIC。
    main_tasks = ((main_tokens + 63) // 64) * 16
    signal = wait_row_owners(signal, PHASE_MAIN_PROJECT, CORES, generation, 0, main_tasks, 1, GROUPS, 0, 1)
    main_pooled = compressor_pool_stage(main_value, main_score, main_state, main_table, main_ape, main_positions, main_bounds, main_lengths, main_pooled, worker, VECTORS)
    signal = publish_phase(signal, PHASE_MAIN_POOL, core, CORES, generation)
    # 所有请求可能读共享历史页：不是只等本核pool返回，就可以覆盖Native state。
    signal = wait_row_owners(signal, PHASE_MAIN_POOL, CORES, generation, 0, main_requests, 1, VECTORS, GROUPS, 1)
    main_state = compressor_state_commit_stage(main_value, main_score, main_ape, main_positions, main_bounds, main_lengths, main_state_slots, main_state, worker, VECTORS)
    signal = publish_phase(signal, PHASE_MAIN_COMMIT, core, CORES, generation)
    signal = wait_producer(signal, PHASE_MAIN_META, GROUPS, CORES, generation)
    main_normed, main_debug, main_cache = compressor_norm_rope_cache_stage(main_pooled, main_gamma, main_cos, main_sin, main_bounds, main_lengths, main_positions, main_offsets, main_slots, main_normed, main_debug, main_cache, worker, VECTORS)
    signal = publish_phase(signal, PHASE_MAIN_OUTPUT, core, CORES, generation)
    # Indexer投影为M16*N32，展平后每M16八个列块；与主投影owner公式不同。
    idx_tasks = ((idx_tokens + 15) // 16) * 8
    signal = wait_row_owners(signal, PHASE_IDX_PROJECT, CORES, generation, 0, idx_tasks, 1, GROUPS, 0, 1)
    idx_pooled = indexer_compressor_pool_stage(idx_value, idx_score, idx_state, idx_table, idx_ape, idx_positions, idx_bounds, idx_lengths, idx_pooled, worker, VECTORS)
    signal = publish_phase(signal, PHASE_IDX_POOL, core, CORES, generation)
    signal = wait_row_owners(signal, PHASE_IDX_POOL, CORES, generation, 0, idx_requests, 1, VECTORS, GROUPS, 1)
    idx_state = indexer_compressor_state_commit_stage(idx_value, idx_score, idx_ape, idx_positions, idx_bounds, idx_lengths, idx_state_slots, idx_state, worker, VECTORS)
    signal = publish_phase(signal, PHASE_IDX_COMMIT, core, CORES, generation)
    signal = wait_producer(signal, PHASE_IDX_META, GROUPS, CORES, generation)
    idx_normed = indexer_compressor_norm_rope_stage(idx_pooled, idx_gamma, idx_cos, idx_sin, idx_positions, idx_bounds, idx_lengths, idx_offsets, idx_normed, worker, VECTORS)
    signal = publish_phase(signal, PHASE_IDX_NORM, core, CORES, generation)
    count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
    had_tasks = ((count + 15) // 16) * 2
    signal = wait_row_owners(signal, PHASE_IDX_HADAMARD, CORES, generation, 0, had_tasks, 1, GROUPS, 0, 1)
    idx_scales, idx_debug, idx_cache = indexer_compact_quant_cache_stage(idx_had, idx_positions, idx_bounds, idx_lengths, idx_offsets, idx_slots, idx_scales, idx_debug, idx_cache, worker, VECTORS)
    signal = publish_phase(signal, PHASE_IDX_QUANT, core, CORES, generation)
    if worker == 0:
        # 即使Native slot各不相同，32个FP16 scale仍共享64B读改写区，必须单写者。
        signal = wait_row_owners(signal, PHASE_IDX_QUANT, CORES, generation, 0, idx_requests, 1, VECTORS, GROUPS, 1)
        idx_cache = indexer_compact_scale_commit_stage(idx_scales, idx_positions, idx_bounds, idx_lengths, idx_offsets, idx_slots, idx_cache, worker)
        signal = publish_phase(signal, PHASE_IDX_SCALE, core, CORES, generation)
"""


def declaration(names, *, writes):
    result = []
    for name, shape, dtype, _ in FIELDS:
        if name in names:
            annotation = f"pl.Tensor[[{shape}], pl.{dtype}]"
            if name in writes:
                annotation = f"pl.InOut[{annotation}]"
            result.append(f"    {name}: {annotation},")
    return "\n".join(result)


def ordered(names):
    return [name for name, _, _, _ in FIELDS if name in names]


def build():
    names = [field[0] for field in FIELDS]
    dimensions = sorted({axis for _, shape, _, _ in FIELDS for axis in shape.split(",") if not axis.isdecimal()})
    text = HEADER + "\n" + "\n".join(f'{name} = pl.dynamic("COMP_PIPE_{name}")' for name in dimensions)
    for role, role_names, writes, body in (
        ("cube", CUBE_NAMES, CUBE_WRITES, CUBE_BODY),
        ("vector", VECTOR_NAMES, VECTOR_WRITES, VECTOR_BODY),
    ):
        aliases = "".join(f"    {name}_gm = {name}\n" for name in writes)
        returns = ", ".join(name + "_gm" for name in writes)
        prologue = '    pl.func_attr({"split_aiv": True})\n' if role == "vector" else ""
        text += f"\n\n@pl.jit.incore\ndef {role}(\n{declaration(role_names, writes=writes)}\n):\n{prologue}{aliases}{body}    return ({returns})\n"
    text += f'\n\n@pl.jit.inline\ndef group(\n{declaration(names, writes=ALL_WRITES)}\n):\n    with pl.cluster(name_hint="dynamic_compressor_pipeline"):\n'
    text += f"        {', '.join(CUBE_WRITES)} = cube({', '.join(ordered(CUBE_NAMES))})\n"
    text += f"        {', '.join(VECTOR_WRITES)} = vector({', '.join(ordered(VECTOR_NAMES))})\n"
    text += f"    return ({', '.join(ALL_WRITES)})\n"
    text += f"\n\n@pl.jit\ndef entry(\n{declaration(names, writes=ALL_WRITES)}\n):\n    with pl.spmd(GROUPS, sync_start=True):\n"
    text += f"        ({', '.join(ALL_WRITES)}) = group({', '.join(names)})\n    return ({', '.join(ALL_WRITES)})\n"
    (HERE / "probe.py").write_text(text)


if __name__ == "__main__":
    build()
