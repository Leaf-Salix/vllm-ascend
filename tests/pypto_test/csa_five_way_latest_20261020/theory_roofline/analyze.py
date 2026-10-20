# SPDX-License-Identifier: Apache-2.0
"""CSA 整层计算量、主数据流与理论下界的离线模型。

只读已实测的形状、策略及性能；不运行 NPU。一个乘加按两次运算计。
“有用数学”和“当前分块执行”分别统计，不能用补齐后的运算冒充有效工作。
HBM 项是明确指定冷/热条件的模型，不把 GM 地址访问一律当作 HBM 事务。
特殊 Vector 指令、标量、通知等待和指令启动不伪装成普通 FLOP。
"""

import configparser
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIVE = HERE.parent
REPO = HERE.parents[3]
PLATFORM = Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/data/platform_config/Ascend910_9392.ini")
PHASES = REPO / "tests/pypto_test/csa_critical_path_20261020/CURRENT_SCORE_PRESSURE_ANALYSIS.json"

# 读本机 CANN 的可编程核数和频率，不使用双 die 卡或 profiler 中含保留核的总数。
PLATFORM_CONFIG = configparser.ConfigParser()
PLATFORM_CONFIG.read(PLATFORM)
CORES = PLATFORM_CONFIG.getint("SoCInfo", "cube_core_cnt")
FREQUENCY = PLATFORM_CONFIG.getfloat("AICoreSpec", "cube_freq") * 1e6
FP16_PEAK = CORES * FREQUENCY * 2 * 16 * 16 * 16
INT8_PEAK = FP16_PEAK * 2
# 官方A3/2201 Mmad标准FP32 cube_k=4，BF16 cube_k=16。未开启HF32时为1/4；
# 仍保留模式敏感性，不能仅凭Python dtype断言运行时全局HF32控制寄存器状态。
FP32_MODEL_PEAK = FP16_PEAK / 4
HBM_PEAK = 1.6e12  # 厂商论文的每 die 上限；不是本轮实测持续带宽。


def read(path):
    return json.loads(Path(path).read_text())


def ceil_div(x, y):
    return (x + y - 1) // y


def pad16(x):
    return ceil_div(x, 16) * 16


def score_geometry(history, batch, sequence, policy):
    """按源级 group/leaf/item 公式重建数学工作数；输入是既有等长测试场景。"""
    group_size, score_tile = policy["QUERY_GROUP"], policy["SCORE_TILE"]
    visible_queries = [(history + q + 1) // 4 for q in range(sequence)] * batch
    group_visible = [
        max((history + q + 1) // 4 for q in range(first, min(first + group_size, sequence)))
        for _ in range(batch) for first in range(0, sequence, group_size)
    ]
    groups, length = len(group_visible), (history + sequence) // 4
    leaves, leaf_tiles, extra = ceil_div(length, 8192), 8, 0
    if policy["BALANCED"]:
        multiple = 3 if 3 * groups in (24, 48) else 6 if 6 * groups == 24 else 1
        if multiple > 1:
            leaves = ceil_div(leaves, multiple) * multiple
            leaf_tiles, extra = divmod(ceil_div(length, 1024), leaves)
            leaf_tiles = max(leaf_tiles, 1)
    blocks = [0] * CORES
    leaf_jobs = []
    for item in range(groups * leaves):
        group, leaf = (item % groups, item // groups) if groups < CORES else divmod(item, leaves)
        begin = (leaf * leaf_tiles + min(leaf, extra)) * 1024
        capacity = (leaf_tiles + int(leaf < extra)) * 1024
        valid = max(min(group_visible[group] - begin, capacity), 0)
        if valid:
            steps = ceil_div(valid, score_tile)
            blocks[item % CORES] += steps
            leaf_jobs.append(dict(group=group, leaf=leaf, owner=item % CORES,
                                  valid=valid, steps=steps))
    # QK 物理 M=G*64；head 加权归约是 M16×K(G*64)，包含零系数和空输出行。
    qk_per_block = 2 * group_size * 64 * score_tile * 128
    ws_per_block = 2 * 16 * group_size * 64 * score_tile
    per_core_us = [(n * qk_per_block / (INT8_PEAK / CORES)
                    + n * ws_per_block / (FP16_PEAK / CORES)) * 1e6 for n in blocks]
    return dict(
        groups=groups, group_size=group_size, compressed_rows=length,
        visible_query_row_sum=sum(visible_queries), group_visible=group_visible,
        leaves=leaves, leaf_tiles=leaf_tiles, extra_leaves=extra,
        core_score_blocks=blocks, total_score_blocks=sum(blocks), score_tile=score_tile,
        qk_ops=sum(blocks) * qk_per_block, ws_ops=sum(blocks) * ws_per_block,
        leaf_jobs=leaf_jobs, core_cube_peak_us=per_core_us,
        balanced_cube_peak_us=sum(per_core_us) / CORES,
        slowest_core_cube_peak_us=max(per_core_us),
        # L1→L0B 的 key INT8 和 QK→WS FP16 矩阵，不含其它端口/延迟。
        right_bytes_per_block=score_tile * (128 + group_size * 64 * 2),
    )


def analyze_case(row):
    h, b, s = row["history"], row["batch"], row["query_tokens"]
    t = b * s
    result_path = FIVE / f"results/h{h}_b{b}/l1_v1/result.json"
    measured = read(result_path)
    policy, shapes = measured["policy"], measured["shapes"]
    assert measured["passed"] and measured["nz_weight_formats"] == {
        "wq_a": 29, "wq_b": 29, "wo_a": 29, "wo_b": 29}
    assert shapes["wq_b"] == [1024, 32768] and shapes["wo_a"] == [8, 4096, 1024]
    tm = pad16(t)
    closed = b * sum((h + q + 1) % 4 == 0 for q in range(s))
    geom = score_geometry(h, b, s, policy)
    stages = []

    def stage(name, dtype, useful, issued, source):
        peak = {"fp16_bf16": FP16_PEAK, "int8": INT8_PEAK, "fp32": FP32_MODEL_PEAK}[dtype]
        stages.append(dict(name=name, dtype=dtype, useful_ops=useful, issued_ops=issued,
                           useful_peak_us=useful / peak * 1e6, issued_peak_us=issued / peak * 1e6,
                           source=source))

    stage("HC_pre 投影", "fp32", 2*t*16384*24, 2*tm*16384*32, "hc_stages.py::hc_project_range")
    stage("Q_A", "fp16_bf16", 2*t*4096*1024, 2*tm*4096*1024, "qkv_stages.py::q_a_stage")
    stage("Q_B", "int8", 2*t*1024*32768, 2*tm*1024*32768, "query_projection_stages.py::_q_b_nz_stage")
    stage("KV 投影", "fp16_bf16", 2*t*4096*512, 2*tm*4096*512, "kv_stages.py::kv_project_stage")
    stage("主压缩 value/gate", "fp16_bf16", 4*t*4096*1024, 4*tm*4096*1024, "compressor_projection_stages.py")
    stage("Indexer 压缩 value/gate", "fp16_bf16", 4*t*4096*256, 4*tm*4096*256, "compressor_projection_stages.py")
    stage("Indexer query 投影", "int8", 2*t*1024*8192, 2*tm*1024*8192, "indexer_query_stages.py")
    stage("Indexer query Hadamard", "fp16_bf16", 2*t*64*128*128, 2*t*64*128*128, "projection_overlap_stages.py")
    stage("Indexer compact Hadamard", "fp16_bf16", 2*closed*128*128, 2*pad16(closed)*128*128,
          "indexer_compressor_output_stages.py")
    stage("Indexer head weights", "fp16_bf16", 2*t*4096*64, 2*tm*4096*64, "indexer_query_stages.py")
    stage("Indexer QK", "int8", 2*geom["visible_query_row_sum"]*64*128, geom["qk_ops"], "indexer_score_stages.py")
    stage("Indexer WS", "fp16_bf16", 2*geom["visible_query_row_sum"]*64, geom["ws_ops"], "indexer_score_stages.py")
    stage("Sparse Attention QK+PV", "fp16_bf16", 4*t*64*640*512, 4*t*64*640*512, "attention_stream_stages.py")
    stage("O_A", "fp16_bf16", 2*t*8*4096*1024, 2*tm*8*4096*1024, "output_projection_a_stages.py")
    # 小中 O_B 仍计算整 M32/M96；大 M128 已按实际行截尾，再受硬件 M16 粒度约束。
    ob_rows = ceil_div(t, policy["OB_ROWS"]) * policy["OB_ROWS"] if policy["OB_ROWS"] <= 96 else tm
    stage("O_B", "int8", 2*t*8192*4096, 2*ob_rows*8192*4096, "output_projection_b_stages.py")

    weight_sizes = {
        "hc_fn": 24*16384*4, "wq_a": 1024*4096*2, "wq_b": 1024*32768,
        "wkv": 4096*512*2, "main_value_gate": 2*1024*4096*2,
        "indexer_value_gate": 2*256*4096*2, "indexer_query": 1024*8192,
        "indexer_head": 4096*64*2, "hadamard": 128*128*2,
        "wo_a": 8*4096*1024*2, "wo_b": 8192*4096,
    }
    # 外部主数据的冷缓存账本。页界对齐、norm/scale、state 写回等小项刻意不加，
    # 因而是偏乐观的主数据下界，不能声称精确计算了所有 HBM 字节。
    external = dict(
        main_matrix_weights=sum(weight_sizes.values()),
        indexer_key_and_scale=b*geom["compressed_rows"]*(128+2),
        sparse_kv_max_reuse=b*(512+128+s-1)*512*2,
        sparse_kv_per_query=t*640*512*2,
        hidden_input_output=t*16384*2*2,
    )
    common = external["main_matrix_weights"] + external["indexer_key_and_scale"] + external["hidden_input_output"]
    cold_min = common + external["sparse_kv_max_reuse"]
    cold_query = common + external["sparse_kv_per_query"]

    # Attention 程序明确存在的主要 GM 数据流。每个 query 5 个 K128 块。
    # 每项只计真正读/写；共享 KV 同时用于 QK/PV，在 AIC 端只载入一次。
    attention = dict(
        native_cache_to_aiv=t*640*512*2,
        kv_ring_write=t*640*512*2,
        kv_ring_read=t*640*512*2,
        qk_score_write_read=2*t*64*640*4,
        probability_write_read=2*t*64*640*2,
        pv_write_read=2*t*5*64*512*4,
        query_read=t*64*512*2,
        packed_output_write=t*64*512*2,
    )
    attention_bytes = sum(attention.values())
    total_issued_us = sum(x["issued_peak_us"] for x in stages)
    # 两种精度共享 Cube 资源，等效占用时间相加，而不是取两者最大值。
    total_useful_us = sum(x["useful_peak_us"] for x in stages)
    hc = stages[0]
    issued_without_hc = total_issued_us - hc["issued_peak_us"]
    return dict(
        history=h, batch=b, query_tokens=s, tokens=t, padded_cube_rows=tm, ob_rows=ob_rows,
        policy=policy, stages=stages, score_geometry=geom,
        useful_cube_us=total_useful_us, issued_cube_us=total_issued_us,
        hc_fp32_peak_sensitivity_us={
            str(divisor): issued_without_hc + hc["issued_ops"] / (FP16_PEAK/divisor)*1e6
            for divisor in (1, 4, 8)},
        main_matrix_weight_bytes=weight_sizes, external_payload_bytes=external,
        cold_main_bytes_max_reuse=cold_min, cold_main_bytes_per_query=cold_query,
        cold_hbm_max_reuse_us=cold_min/HBM_PEAK*1e6,
        cold_hbm_per_query_us=cold_query/HBM_PEAK*1e6,
        cold_roofline_max_reuse_us=max(total_issued_us, cold_min/HBM_PEAK*1e6),
        cold_roofline_per_query_us=max(total_issued_us, cold_query/HBM_PEAK*1e6),
        attention_gm_bytes=attention, attention_gm_total_bytes=attention_bytes,
        attention_gm_bandwidth_sensitivity_us={str(tb): attention_bytes/(tb*1e12)*1e6 for tb in (2, 4, 6)},
        actual_dynamic_l1_us=row["dynamic_l1_us"],
        original_multitask_l1_us=row["multitask_l1_us"],
        thirty_percent_target_us=row["multitask_l1_us"]*0.7,
        remaining_to_target_us=row["dynamic_l1_us"]-row["multitask_l1_us"]*0.7,
        issued_cube_peak_occupancy_model=total_issued_us/row["dynamic_l1_us"],
        measurement=str(result_path),
    )


def main():
    rows = [analyze_case(row) for row in read(FIVE/"PERFORMANCE.json")["rows"]]
    phases = read(PHASES)
    for row in rows:
        if row["history"] == 131072 and str(row["batch"]) in phases:
            old = phases[str(row["batch"])]
            # 检查 CPU 重建的 24 核 Score 块数与此前真机诊断逐核相同。
            observed = [item["blocks"] for item in old["score_groups"]]
            assert observed == row["score_geometry"]["core_score_blocks"]
            row["observed_score_blocks_match"] = True
            row["stage_probe"] = {
                name: old["phases"][name]
                for name in ("QB", "SCORE_CUBE", "SCORE_VECTOR", "ATTN_CUBE", "ATTN_VECTOR", "OB", "OUT")}
            row["stage_probe_source"] = str(PHASES)
    report = dict(
        assessment_status="cube_component_see_FULL_ROOFLINE_for_vector_and_memory",
        vector_model_in_this_file=False,
        full_resource_report="FULL_ROOFLINE.json",
        complete_csa_theoretical_latency_available=False,
        scope="TP1、正式W8A8、第4层、S6；HC_pre+输入RMS+完整CSA+HCpost；无MoE/整网",
        source_snapshot=str(FIVE/"source_dynamic"), measured_dynamic_commit="13259f92",
        hardware=dict(soc="Ascend910_9392", logical_device_scope="一个die",
                      cube_cores=CORES, vector_cores=48, cube_frequency_hz=FREQUENCY,
                      fp16_bf16_peak_ops_per_second=FP16_PEAK, int8_peak_ops_per_second=INT8_PEAK,
                      fp32_peak_model_assumption=FP32_MODEL_PEAK, hbm_peak_bytes_per_second=HBM_PEAK,
                      l2_bytes=PLATFORM_CONFIG.getint("SoCInfo", "l2_size"), platform_source=str(PLATFORM)),
        hardware_sources=[
            "https://arxiv.org/html/2506.12708v2#S3.SS3.SSS1",
            "https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html",
            "https://www-file.huawei.com/admin/asset/v1/pro/view/6ca306adab0a4852bacffce25f5573ec.pdf",
        ],
        limits=[
            "峰值上限模型，不是实测可持续算力或带宽。",
            "Vector排序/归约/exp、Scalar寻址、同步、启动、片上端口延迟未折成普通FLOP。",
            "冷缓存账本不适用于声称ACLGraph热重放的精确HBM流量；GM请求不等于HBM事务。",
            "实际Sparse Top-K跨query重合率未知，显式给出最大复用与逐query读取情形。",
            "当前缓存BYPASS声明无有效地址offset，生成普通TLOAD，不能假设其每次必访HBM。",
            "当前分块计算估计保留M16、WS M16和历史尾块；不是逐条硬件指令动态计数。",
        ],
        rows=rows,
    )
    (HERE/"ANALYSIS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    for row in rows:
        print(row["history"], row["batch"],
              *(round(row[key], 2) for key in (
                  "useful_cube_us", "issued_cube_us", "cold_hbm_max_reuse_us",
                  "cold_hbm_per_query_us", "actual_dynamic_l1_us")))


if __name__ == "__main__":
    main()
