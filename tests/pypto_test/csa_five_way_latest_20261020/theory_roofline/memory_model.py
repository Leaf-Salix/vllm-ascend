# SPDX-License-Identifier: Apache-2.0
"""按当前分块重建24核Cube/片上输入端口，及有缓存条件的HBM账本。

L1→L0A/B采用官方A3 LoadData带宽256/128 B/cycle，不把平台ini的调优参数
当成指令带宽。FIX这里只对有明确公开吞吐的Score INT32→FP16→L1计数，
其余GM搬运不借用A5吞吐。GM逻辑请求、L2命中和HBM事务是三种不同口径。
"""

from collections import Counter
import json

from analyze import HERE, CORES, FREQUENCY, FP16_PEAK, INT8_PEAK, FP32_MODEL_PEAK, HBM_PEAK, ceil_div, pad16

CUBE_SOURCE = "https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html"
L2_BYTES = 201326592


class CubePorts:
    def __init__(self):
        self.stages = {}

    def mm(self, name, worker, m, n, k, dtype, *, left_factor=1, right_factor=1, fix_l1=False):
        assert 0 <= worker < 24 and m % 16 == 0 and n % 16 == 0
        cores = self.stages.setdefault(name, [Counter() for _ in range(24)])
        size, peak = {"fp32": (4, FP32_MODEL_PEAK), "bf16": (2, FP16_PEAK), "int8": (1, INT8_PEAK)}[dtype]
        ops = 2*m*n*k
        cores[worker].update(
            ops=ops, cube_cycles=ops/(peak/CORES/FREQUENCY),
            left_bytes=m*k*size*left_factor, right_bytes=k*n*size*right_factor,
            fix_to_l1_bytes=m*n*2 if fix_l1 else 0,
        )

    def export(self):
        combined = [Counter() for _ in range(24)]
        stages = []
        for name, cores in self.stages.items():
            for i in range(24):
                combined[i].update(cores[i])
            stages.append(self.summarize(name, cores))
        result = self.summarize("整层合计", combined)
        result["stages"] = stages
        return result

    @staticmethod
    def summarize(name, cores):
        cycles = {}
        for metric, rate in (("cube_cycles", 1), ("left_bytes", 256),
                             ("right_bytes", 128), ("fix_to_l1_bytes", 128)):
            cycles[metric] = [c[metric]/rate for c in cores]
        times = {key: [c/(FREQUENCY/1e6) for c in value] for key, value in cycles.items()}
        return dict(name=name, counts_per_aic=[dict(c) for c in cores],
                    sum_counts=dict(sum(cores, Counter())),
                    service_us_per_aic=times,
                    balanced_us={key: sum(values)/24 for key, values in times.items()},
                    current_assignment_us={key: max(values) for key, values in times.items()})


def qa_or_kv(log, t, name, nblocks, m16_parallel=False, physical_shift=0):
    dense = 32 if t < 128 else 64
    groups = min(3, max(1, t//dense))
    if m16_parallel:
        for item in range(nblocks*ceil_div(t, 16)):
            log.mm(name, item % 24, 16, 128, 4096, "bf16")
        return
    full = t//dense*dense
    for col in range(nblocks):
        for group in range(groups):
            owner = (col*groups+group+physical_shift) % 24
            for _ in range(group*dense, full, groups*dense):
                log.mm(name, owner, dense, 128, 4096, "bf16")
            for _ in range(full+group*16, pad16(t), groups*16):
                log.mm(name, owner, 16, 128, 4096, "bf16")


def cube_ports(row):
    t, b, p, geom = row["tokens"], row["batch"], row["policy"], row["score_geometry"]
    log = CubePorts()
    for item in range(ceil_div(t, 16)*4):
        log.mm("HC_pre 投影", item % 24, 16, 32, 4096, "fp32")
    qa_or_kv(log, t, "Q_A", 8, p["QA_M16_PARALLEL"])
    qa_or_kv(log, t, "KV 投影", 4, physical_shift=4)
    for col in range(32768//256):
        for start in range(0, t, 128):
            # 正式代码qb_worker=physical_worker%20，物理核集合4..23。
            logical = col % 20
            physical = logical if logical >= 4 else logical+20
            log.mm("Q_B", physical, pad16(min(t-start, 128)), 256, 1024, "int8")
    for name, mtile, ntile, nfull in (
        ("主压缩 value/gate", 64, 64, 1024),
        ("Indexer 压缩 value/gate", 16, 32, 256),
    ):
        for item in range(ceil_div(t, mtile)*(nfull//ntile)):
            block = item//(nfull//ntile)
            rows = pad16(min(mtile, t-block*mtile))
            # value/gate共用输入Tile；输入端口取可复用一次的乐观下界。
            log.mm(name, item % 24, rows, ntile, 4096, "bf16")
            log.mm(name, item % 24, rows, ntile, 4096, "bf16", left_factor=0)
    for col in range(8192//256):
        for start in range(0, t, 64):
            log.mm("Indexer query 投影", col % 24, pad16(min(64, t-start)), 256, 1024, "int8")
    for item in range(ceil_div(t*64, 256)):
        log.mm("Indexer query Hadamard", item % 4, min(256, t*64-item*256), 128, 128, "bf16")
    for item in range(2*ceil_div(b, 16)):
        log.mm("Indexer compact Hadamard", item % 4, 16, 64, 128, "bf16")
    for item in range(4*ceil_div(t, 16)):
        log.mm("Indexer head weights", item % 24, 16, 64, 1024, "bf16")
    for job in geom["leaf_jobs"]:
        g, width = geom["group_size"], geom["score_tile"]
        for _ in range(job["steps"]):
            log.mm("Indexer QK", job["owner"], g*64, width, 128, "int8", fix_l1=True)
            log.mm("Indexer WS", job["owner"], 16, width, g*64, "bf16")
    for token in range(t):
        for _ in range(5):
            log.mm("Sparse Attention QK+PV", token % 24, 64, 128, 512, "bf16")
            # PV显式把probability一次放入L0A，再复用到四个N128；按一次计。
            log.mm("Sparse Attention QK+PV", token % 24, 64, 512, 128, "bf16")
    nblocks = 1024//p["OA_COLS"]
    for item in range(8*ceil_div(t, 128)*nblocks):
        block = (item//nblocks) % ceil_div(t, 128)
        log.mm("O_A", item % 24, pad16(min(128, t-block*128)), p["OA_COLS"], 4096, "bf16")
    for item in range(8*8):
        for _ in range(2):
            for start in range(0, t, p["OB_ROWS"]):
                m = p["OB_ROWS"] if p["OB_ROWS"] <= 96 else pad16(min(p["OB_ROWS"], t-start))
                log.mm("O_B", item % 24, m, 256, 1024, "int8")
    result = log.export()
    expected = {stage["name"]: stage["issued_ops"] for stage in row["stages"]}
    # 两套独立循环公式核对所有15个矩阵阶段，避免端口模型漏掉矩阵或pad。
    assert set(expected) == set(log.stages)
    for stage in result["stages"]:
        assert stage["sum_counts"]["ops"] == expected[stage["name"]], (
            stage["name"], stage["sum_counts"]["ops"], expected[stage["name"]])
    assert abs(result["balanced_us"]["cube_cycles"]-row["issued_cube_us"]) < 1e-8
    result["independent_cube_counts_match_all_15_stages"] = True
    result["limitations"] = [
        "左右端口分别给下界，不假定它们带宽可相加，也不计搬运启动。",
        "每逻辑矩阵的最少L0输入脚印；后端再分块导致的重复加载只会增加请求。",
        "双压缩矩阵的共同左Tile采用复用一次假设；不借它声称后端已做到复用。",
        "FIX仅核算公开A3速率明确的Score QK→FP16 L1通路，不冒用A5的GM/UB数据。",
    ]
    return result


def memory(row, vector):
    t, b, s, h, geom = row["tokens"], row["batch"], row["query_tokens"], row["history"], row["score_geometry"]
    # 逻辑有效字节；不按整个cache/state allocation容量收费。
    aux = {
        "norm_and_gamma_bf16": (4096+1024+512+512+128)*2,
        "projection_scales_fp32": (32768+8192+4096)*4,
        "hc_scale_base_sink_fp32": (3+24+64)*4,
        "compressor_ape_fp32": 4*(1024+256)*4,
    }
    meta = {
        "positions_bounds_lengths_slots": t*(8+8+2*8)+(b+1)*3*4+b*(3*4+2*8),
        "indexer_and_compressed_page_tables": 2*b*ceil_div(geom["compressed_rows"], 32)*4,
        "original_and_state_selected_pages": b*(5+2*4)*4,
        "rope_main_and_compact": t*64*2*4 + 2*b*64*2*4,
    }
    reads = {
        "main_matrix_weights": sum(row["main_matrix_weight_bytes"].values()),
        "auxiliary_weights": sum(aux.values()),
        # 本轮新写的压缩key不是外部冷读。旧历史key只读一份，G6/G2复用另算GM请求。
        "old_indexer_keys_and_scales": b*(h//4)*(128+2),
        "hidden_input": t*16384*2,
        "old_state_values_scores": b*4*2*(512+128)*4,
        "metadata_and_rope": sum(meta.values()),
    }
    # q=0要看127条旧SWA及512条旧compact；其余query最大复用这些旧compact。
    sparse_min = b*(127+512)*512*2
    sparse_no_reuse = b*(s*512 + sum(127-q for q in range(s)))*512*2
    writes = {
        "hidden_output": t*16384*2,
        "state_values_scores": t*2*(1024+256)*4,
        "original_kv": t*512*2,
        "main_compact_kv": b*512*2,
        "indexer_compact_key_scale": b*130,
    }
    cold_read_min = sum(reads.values())+sparse_min
    cold_read_noreuse = sum(reads.values())+sparse_no_reuse
    # 热重放容量下界只扣除起始时最多可缓存的只读字节，不假定所有GM都命中。
    # 这是容量约束，不是实际命中率或LRU缓存模拟。
    hot_capacity_bytes = max(0, cold_read_min-L2_BYTES)

    # 语义级GM流量下界：物化一次的中间量至少写一次、每个直接消费者读一次。
    # 只用于显示L2带宽敏感性；不拿它冒充完整动态访存计数或HBM计数。
    gm = {
        "hc_widen_materialize_and_two_consumers": 3*t*16384*4,
        "hc_partials_reduce_raw_gates": 2*4*pad16(t)*32*4 + 3*t*32*4 + 2*t*(8+4+16)*4,
        "mixed_write_norm_read": 2*t*4096*2,
        "normed_write_five_branches": 6*t*4096*2,
        "qa_write_read": 2*t*1024*4,
        "qr_two_roots_write_read": 4*t*1024 + 4*t*4,
        "main_query_acc_write_read": 2*t*32768*4,
        "main_query_bf16_write": t*32768*2,
        "kv_acc_write_read_and_normed_write_read": 2*t*512*(4+2),
        "compressor_projections_write_commit_read": 4*t*(1024+256)*4,
        "compressor_pool_zero_close_read": (2*t+b)*(512+128)*4,
        "main_normed_and_debug": t*512*4+3*b*512*4,
        "indexer_compact_hadamard_quant": b*128*(2+2+4+4+4),
        "indexer_query_acc_write_read": 2*t*8192*4,
        "indexer_rope_write_read": 2*t*8192*2,
        "indexer_hadamard_write_read": 2*t*8192*4,
        "indexer_quant_write_and_score_read": t*8192 + len(geom["leaf_jobs"])*geom["group_size"]*8192,
        "head_weights_and_coefficients": 2*4*pad16(t)*64*4+2*t*64*4
            +(geom["groups"]+len(geom["leaf_jobs"]))*16*geom["group_size"]*64*2,
        "score_transfer_write_read": 2*geom["total_score_blocks"]*geom["group_size"]*geom["score_tile"]*4,
        "leaf_roots_write_read": 2*sum(vector["sorting"]["active_roots_per_query"])*1024*4,
        "topk_output_and_indices_read": 3*t*512*4,
        "attention_plan_and_signed_rope": 2*t*(512+640+16+64)*4,
        "attention_payload": row["attention_gm_total_bytes"],
        "oa_result_write_twice_read": 3*t*8192*4,
        "oa_quant_write_ob_read": 2*ceil_div(t, row["policy"]["OB_ROWS"])*row["policy"]["OB_ROWS"]*8192,
        "ob_partial_write_read": 2*t*32768*4,
    }
    if not row["policy"]["BALANCED"]:
        gm["short_score_arena_write_read"] = 2*t*geom["compressed_rows"]*4
    traffic_min = sum(gm.values()) + sum(reads.values()) + sparse_min + sum(writes.values())
    # attention_payload已经含Native KV读取，故加外部数据时不能再次加sparse_min。
    traffic_min -= sparse_min
    return dict(
        cold_read_ledger_bytes=reads, auxiliary_weight_bytes=aux, metadata_bytes=meta,
        sparse_old_kv_max_reuse_bytes=sparse_min,
        sparse_old_kv_each_query_bytes=sparse_no_reuse, output_dirty_bytes=writes,
        cold_read_max_reuse_us=cold_read_min/HBM_PEAK*1e6,
        cold_read_per_query_us=cold_read_noreuse/HBM_PEAK*1e6,
        cold_read_writeback_per_query_us=(cold_read_noreuse+sum(writes.values()))/HBM_PEAK*1e6,
        hot_start_capacity_floor_bytes=hot_capacity_bytes,
        hot_start_capacity_floor_us=hot_capacity_bytes/HBM_PEAK*1e6,
        hbm_peak_bytes_per_second=HBM_PEAK, l2_capacity_bytes=L2_BYTES,
        semantic_gm_materialization_bytes=gm, semantic_gm_ledger_total_bytes=traffic_min,
        gm_aggregate_rate_sensitivity_us={str(tb): traffic_min/(tb*1e12)*1e6 for tb in (2, 4, 6, 7)},
        cold_bandwidth_sensitivity_us={str(tb): cold_read_min/(tb*1e12)*1e6 for tb in (0.8, 1.2, 1.6)},
        limits=[
            "cold是假定这些外部只读输入首次在本die L2之外，不代表ACLGraph热重放。",
            "输出可先留在L2，只有显式writeback场景才将全部脏输出算入同窗口HBM。",
            "最强复用和逐query供给两端描述KV访问条件，不是实际命中率置信区间。",
            "语义GM账本不含所有重复Tile搬入、Scalar信号、cacheline放大和临时溢出。",
            "L2聚合带宽未实测且未找到本机公开保证值；2/4/6/7TB/s只是敏感性参数。",
            "7TB/s来自官方192MB L2案例中的假设示例，不能充当Ascend910_9392实测规格。",
        ],
    )


if __name__ == "__main__":
    base = json.loads((HERE/"ANALYSIS.json").read_text())
    vectors = json.loads((HERE/"VECTOR_MODEL.json").read_text())["rows"]
    rows = []
    for row, vector in zip(base["rows"], vectors):
        assert (row["history"], row["batch"]) == (vector["history"], vector["batch"])
        ports, mem = cube_ports(row), memory(row, vector)
        rows.append(dict(history=row["history"], batch=row["batch"], ports=ports, memory=mem))
        print(row["history"], row["batch"],
              {k: round(v, 2) for k, v in ports["current_assignment_us"].items()},
              "cold", round(mem["cold_read_max_reuse_us"], 2))
    (HERE/"MEMORY_MODEL.json").write_text(json.dumps(
        dict(source=CUBE_SOURCE, rows=rows), ensure_ascii=False, indent=2)+"\n")
