# SPDX-License-Identifier: Apache-2.0
"""完整 CSA 的 Vector 工作量账本：只在 CPU 复算，不启动 NPU。

总览
----
1. 沿已测源码逐阶段统计元素操作，覆盖 HC、Q/KV、两套压缩、Indexer、
   Attention、输出量化和 HCpost；不能只用矩阵 FLOP 代表整个算子。
2. 普通算术、特殊函数、转换、Gather、排序各用 A3 自己的理论吞吐。
   Sort/MrgSort 的单位为 (score,index) 项，不是底层 FP32 暂存宽度。
3. 按实际 worker 映射分别累加48个 AIV，报告理想均衡和最忙核两种服务量。
   这仍是峰值工作量模型，不能代替含 Scalar、事件等待和搬运的流水模拟。
4. 归约同时给两种口径：理想元素归约；当前通用 Tile 归约的折叠+vcadd/
   vcmax 模型。特定静态 Tile 可触发更快路径，不能把后一口径称为严格下界。
   UB 中的额外复制、广播展开、指令头、mask 设置未作逐指令动态计数。

所有七档均 S6、无 dummy；公式只描述这组实测，不代替动态算子的输入合同。
源码位置和作用写在 STAGE_SOURCES，输出 JSON 保留逐阶段、逐核和逐类别账本。
"""

from collections import Counter
import json

from analyze import HERE, ceil_div, pad16
from vector_sort_bound import analyze as sort_work

FREQ_MHZ = 1850
VECTORS = 48
VECTOR_SOURCE = "https://asc.gitcode.com/api/appendix/vector_instruction_theoretical_perf_summary.html"
# 每个 AIV、每 cycle 的元素吞吐。arith32包含Add/Sub/Mul/Min/Max/Abs/Neg，
# 不将Exp/Div、排序或转换混进FP32 FLOP峰值。
RATES = {
    "arith32": 64, "arith16": 128,
    "exp32": 32, "sqrt32": 32, "div32": 32, "recip32": 64,
    "cast32": 64, "cast16_to_int8": 128,
    "fill32": 64, "fill16": 128, "index32": 64,
    "gather32": 5, "gatherb16": 42, "gathermask32": 32,
    "sort_items": 2, "merge_items_min": 2, "merge_items_extra": 2,
    "reduce_repeat32": 64 / 7,
}
STAGE_SOURCES = {
    "HC加宽与RMS": ["hc_stages.py::hc_widen_rms_range"],
    "HC投影归约": ["hc_stages.py::hc_reduce_range"],
    "HC门控": ["hc_stages.py::hc_gates_range"],
    "HC混合与输入RMS": ["hc_stages.py::hc_mix_norm_range", "../deepseek_v4_flash_csa/rmsnorm.py"],
    "HC_Sinkhorn": ["hc_stages.py::hc_comb_range"],
    "QR归一化量化": ["qkv_stages.py::q_r_stage"],
    "公共RoPE准备": ["query_projection_stages.py::q_rope_prepare_stage",
                    "attention_plan_stages.py::attention_inverse_rope_sign_stage"],
    "主Query反量化RMS_RoPE": ["query_projection_stages.py::q_dequant_rope_stage"],
    "原始KV归一化RoPE": ["kv_stages.py::kv_norm_rope_stage"],
    "主压缩pool_state": ["compressor_pool_stages.py::compressor_pool_stage",
                        "compressor_pool_stages.py::compressor_state_commit_stage"],
    "主压缩norm_RoPE_cache": ["compressor_output_stages.py::compressor_norm_rope_cache_stage"],
    "Indexer压缩pool_state": ["compressor_pool_stages.py::indexer_compressor_pool_stage",
                             "compressor_pool_stages.py::indexer_compressor_state_commit_stage"],
    "Indexer压缩norm_RoPE": ["indexer_compressor_output_stages.py::indexer_compressor_norm_rope_stage"],
    "Indexer压缩量化": ["indexer_cache_stages.py::indexer_compact_quant_cache_stage"],
    "IndexerQuery反量化RoPE": ["indexer_query_stages.py::indexer_query_dequant_rope_stage"],
    "IndexerQuery量化": ["indexer_query_stages.py::indexer_query_quant_stage"],
    "IndexerHead系数": ["indexer_query_stages.py::indexer_weights_reduce_stage",
                       "indexer_group_stages.py::indexer_head_coefficients_stage"],
    "IndexerScale与排序准备": ["indexer_score_stages.py::indexer_score_vector_lane_stage"],
    "Indexer叶内排序归并": ["indexer_topk_stages.py", "indexer_score_stages.py"],
    "TopK末级归并发布": ["indexer_publish_stages.py::indexer_pair_publish_stage",
                       "indexer_topk_stages.py::merge_top512_roots"],
    "前置metadata与页证明": ["indexer_scale_layout_stages.py::indexer_scale_layout_plan_stage",
                           "attention_plan_stages.py::attention_geometry_prepare_stage"],
    "Attention有效块计划": ["attention_plan_stages.py::attention_slot_plan_stage"],
    "AttentionSoftmax": ["attention_stream_stages.py::attention_qk_pv_vector_lane_stage"],
    "Attention在线归并与逆RoPE": ["attention_stream_stages.py::attention_qk_pv_vector_lane_stage"],
    "O_A输出量化": ["output_projection_a_stages.py::o_projection_groups_quant_stage"],
    "O_B反量化与HCpost": ["output_projection_b_stages.py::o_dequant_hc_post_stage",
                        "output_projection_b_stages.py::hc_post_row_block"],
}


class Ledger:
    def __init__(self):
        self.stages = {name: [Counter() for _ in range(VECTORS)] for name in STAGE_SOURCES}
        self.reductions = {name: [] for name in STAGE_SOURCES}

    def add(self, stage, worker, **counts):
        assert 0 <= worker < VECTORS
        assert all(v >= 0 for v in counts.values())
        assert counts.keys() <= RATES.keys()
        self.stages[stage][worker].update(counts)

    def reduce(self, stage, worker, rows, columns, repeats=1):
        """通用实现折到64列再作整行归约；小于64列仍是一条masked归约。

        部分8×1024等静态Tile存在vcgadd/vcgmax快速实现，见TRowReduceOps。
        因此另保留按有效元素构造的乐观归约口径供敏感性比较。
        """
        folds = max(ceil_div(columns, 64) - 1, 0) * 64
        self.add(stage, worker, arith32=rows * folds * repeats,
                 reduce_repeat32=rows * 64 * repeats)
        self.reductions[stage].append(dict(worker=worker, rows=rows, columns=columns, repeats=repeats,
                                          generic_cycles=rows * (folds / 64 + 7) * repeats,
                                          ideal_cycles=rows * max(columns - 1, 0) / 64 * repeats))

    def quant_casts(self, stage, worker, elements):
        self.add(stage, worker, cast32=2 * elements, cast16_to_int8=elements)

    def high_precision_rsqrt(self, stage, worker, rows):
        # 当前A2/A3带tmp的TRSQRT使用vsqrt+vdiv；不能按单条vrsqrt记账。
        self.add(stage, worker, sqrt32=rows, div32=rows)

    def export(self):
        stages, total_lo, total_hi, ideal_total = [], [0.0] * 48, [0.0] * 48, [0.0] * 48
        for name, cores in self.stages.items():
            lo = [sum(n / RATES[op] for op, n in counts.items() if op != "merge_items_extra")
                  for counts in cores]
            hi = [v + counts["merge_items_extra"] / 2 for v, counts in zip(lo, cores)]
            ideal = lo.copy()
            for reduction in self.reductions[name]:
                ideal[reduction["worker"]] -= reduction["generic_cycles"] - reduction["ideal_cycles"]
            assert min(ideal) >= -1e-8
            total_lo = [a + b for a, b in zip(total_lo, lo)]
            total_hi = [a + b for a, b in zip(total_hi, hi)]
            ideal_total = [a + b for a, b in zip(ideal_total, ideal)]
            counts = sum(cores, Counter())
            stages.append(dict(
                name=name, source=STAGE_SOURCES[name], counts=dict(counts),
                counts_per_aiv=[dict(x) for x in cores],
                cycles_min_per_aiv=lo, cycles_max_per_aiv=hi,
                reductions=self.reductions[name],
                balanced_us=sum(lo) / 48 / FREQ_MHZ,
                current_assignment_min_us=max(lo) / FREQ_MHZ,
                current_assignment_max_us=max(hi) / FREQ_MHZ,
            ))
        return dict(
            stages=stages, ideal_balance_min_us=sum(total_lo) / 48 / FREQ_MHZ,
            ideal_balance_max_us=sum(total_hi) / 48 / FREQ_MHZ,
            current_assignment_min_us=max(total_lo) / FREQ_MHZ,
            current_assignment_max_us=max(total_hi) / FREQ_MHZ,
            ideal_reduction_current_assignment_us=max(ideal_total) / FREQ_MHZ,
            ideal_reduction_balance_us=sum(ideal_total) / 48 / FREQ_MHZ,
            service_min_us_per_aiv=[x / FREQ_MHZ for x in total_lo],
            service_max_us_per_aiv=[x / FREQ_MHZ for x in total_hi],
            counts=dict(sum((Counter(s["counts"]) for s in stages), Counter())),
        )


def add_rope_index(ledger, stage, worker, rows):
    """当前浮点奇偶索引准备的源级工作量；不把Scalar索引当Vector。"""
    n = rows * 64
    ledger.add(stage, worker, index32=64, cast32=64 + 3 * n,
               arith32=9 * n, fill32=n)


def analyze(row):
    t, b, s, h = row["tokens"], row["batch"], row["query_tokens"], row["history"]
    assert s == 6 and t % 8 == 0 and h % 4 == 0
    p, geom = row["policy"], row["score_geometry"]
    log = Ledger()
    tiles8 = t // 8
    for block in range(tiles8):
        w, r = block % 48, 8
        n = r * 16384
        log.add("HC加宽与RMS", w, cast32=n, arith32=n + 32*r + 2*r, fill32=r)
        log.reduce("HC加宽与RMS", w, r, 512, 32)
        log.high_precision_rsqrt("HC加宽与RMS", w, r)
        # Pre/Post物理各8列，Post只写4列，不能只算8个门值。
        n = r * 16
        log.add("HC门控", w, arith32=6*n, exp32=n, recip32=n)
        # 四路pairwise混合+BF16舍入；归一化第二遍重读mixed。
        n = r * 4096
        log.add("HC混合与输入RMS", w, arith32=10*n + 18*r,
                cast32=4*n + 4096, fill32=r)
        log.reduce("HC混合与输入RMS", w, r, 256, 16)
        log.high_precision_rsqrt("HC混合与输入RMS", w, r)
        # 初始四行softmax，再20次列归一化、19次后续行归一化。
        n = r * 16
        log.add("HC_Sinkhorn", w, arith32=5*n + 20*n + 19*4*r,
                exp32=n, div32=40*n)
        log.reduce("HC_Sinkhorn", w, r*4, 4, 21)  # 一次row_max，20次row_sum。
        # QR: 四段平方和及gamma后amax，第二遍乘rstd/gamma/量化尺度。
        n = r * 1024
        log.add("QR归一化量化", w, arith32=6*n + 12*r, cast32=1024,
                div32=r, recip32=r, fill32=4*r)
        log.reduce("QR归一化量化", w, r, 256, 8)
        log.high_precision_rsqrt("QR归一化量化", w, r)
        log.quant_casts("QR归一化量化", w, n)
        add_rope_index(log, "公共RoPE准备", w, r)
        log.add("公共RoPE准备", w, arith32=r*64)
        # Attention逆RoPE符号独立准备，不能与正向signed_sin合并。
        add_rope_index(log, "公共RoPE准备", w, r)
        log.add("公共RoPE准备", w, arith32=r*64)
    for block in range(ceil_div(t, 16)):
        log.add("HC投影归约", block % 48, arith32=3*16*32)

    # 主Query按当前动态head分组映射；各档均无8行尾块。
    query_head_group = next((n for n in (64, 32, 16, 8)
                             if tiles8 >= 48*n//64 and tiles8 % (48*n//64) == 0), 4)
    for task in range(tiles8 * (64 // query_head_group)):
        w = task % 48
        n, r = 8 * query_head_group * 512, 8 * query_head_group
        log.add("主Query反量化RMS_RoPE", w, arith32=4*n + 2*r + 3*r*64 + 3*512,
                cast32=2*n + 5*512, index32=512, gather32=r*64)
        log.reduce("主Query反量化RMS_RoPE", w, r, 512)
        log.high_precision_rsqrt("主Query反量化RMS_RoPE", w, r)
    for block in range(ceil_div(t, 32)):
        w, r = block % 48, min(32, t - block*32)
        n = r * 512
        log.add("原始KV归一化RoPE", w, arith32=4*n + 2*r + 3*r*64,
                sqrt32=r, div32=r, cast32=n + 512, gather32=r*64, fill32=r*64)
        log.reduce("原始KV归一化RoPE", w, r, 64)
        if r < 32:
            add_rope_index(log, "原始KV归一化RoPE", w, 32)

    # 压缩pool按请求条带；本轮每请求恰一次关闭，四条旧state+四条本轮overlay。
    for request in range(b):
        w = request % 48
        for prefix, dim in (("主", 512), ("Indexer", 128)):
            pool = prefix + "压缩pool_state"
            norm = "主压缩norm_RoPE_cache" if prefix == "主" else "Indexer压缩norm_RoPE"
            log.add(pool, w, arith32=61*dim + s*2*dim, exp32=15*dim,
                    div32=dim, fill32=s*dim)
            n = s * dim
            log.add(norm, w, arith32=3*n + 2*s + 4*s*64,
                    sqrt32=s, recip32=s, cast32=dim + (dim if prefix == "主" else n),
                    gather32=s*64, fill32=2*16*64)
            log.reduce(norm, w, s, dim)
            add_rope_index(log, norm, w, 16)
            # Indexer compact Hadamard后只有关闭行量化；含BF16往返舍入。
            if prefix == "Indexer":
                q = "Indexer压缩量化"
                log.add(q, w, arith32=3*128 + 1, cast32=2*128,
                        div32=1, recip32=1, fill32=32)
                log.reduce(q, w, 1, 128)
                log.quant_casts(q, w, 128)

    for task in range(tiles8 * 16):
        w, r, n = task % 48, 8*4, 8*4*128
        log.add("IndexerQuery反量化RoPE", w, arith32=2*n + 3*r*64,
                cast32=2*n, gather32=r*64)
        # 真正TGATHER前的行偏移/奇偶索引；rems按当前整数实现的乐观工作量。
        log.add("IndexerQuery反量化RoPE", w, index32=64+8,
                arith32=6*64 + 512 + 8, fill32=512)
    for w in range(min(48, tiles8*16)):
        add_rope_index(log, "IndexerQuery反量化RoPE", w, 8)
    for token in range(t):
        w, r, n = token % 48, 64, 64*128
        log.add("IndexerQuery量化", w, arith32=4*n + 2*r, div32=r, recip32=r, fill32=2*r)
        log.reduce("IndexerQuery量化", w, r, 64, 2)
        log.quant_casts("IndexerQuery量化", w, n)
    for block in range(ceil_div(t, 16)):
        log.add("IndexerHead系数", block % 48, arith32=4*16*64)
    for group in range(geom["groups"]):
        g = geom["group_size"]
        log.add("IndexerHead系数", group % 48, cast32=2*g*64,
                arith16=g*64, fill16=16*g*64)

    # Score Vector只乘KV scale；QK INT32转FP16和ReLU在FIX完成，不重复算给AIV。
    sorting = sort_work(row)
    for job in geom["leaf_jobs"]:
        for lane in range(2):
            w = job["owner"]*2 + lane
            width = geom["score_tile"] if p["BALANCED"] else geom["score_tile"]//2
            queries = geom["group_size"]//2 if p["BALANCED"] else geom["group_size"]
            log.add("IndexerScale与排序准备", w, cast32=job["steps"]*width,
                    arith32=job["steps"]*width*queries, fill16=job["steps"]*width)
            if p["BALANCED"]:
                log.add("IndexerScale与排序准备", w, fill32=queries*1024+geom["group_size"]*1024)
            if p["SCALE_PAGE_COALESCING"]:
                # 本次实测audit逐请求均direction=-1，确实走逆序页gatherb。
                nidx = 2*(width//32)
                log.add("IndexerScale与排序准备", w, index32=job["steps"]*nidx,
                        arith32=job["steps"]*nidx*9, gatherb16=job["steps"]*width)
    for segment in sorting["segment_jobs"]:
        # 构建全局index、夹紧score和填无效尾；sort吞吐另算，不能重复计。
        log.add("IndexerScale与排序准备", segment["worker"], index32=segment["tile"],
                arith32=2*segment["tile"], fill32=segment["tile"]-segment["valid"])
    for w in range(48):
        local_merge = sorting["local_merge_items_per_aiv"][w]
        log.add("Indexer叶内排序归并", w, sort_items=sorting["sort_items_per_aiv"][w],
                merge_items_min=local_merge)
        low = sorting["merge_items_min_per_aiv"][w] - local_merge
        high = sorting["merge_items_max_per_aiv"][w] - local_merge
        log.add("TopK末级归并发布", w, merge_items_min=low, merge_items_extra=high-low)
    for token in range(t):
        # GatherMask的吞吐按输入1024个FP32位置统计，两次拆score/index。
        log.add("TopK末级归并发布", token % 48, gathermask32=2*1024)

    # metadata没有当免费处理：Vector页证明/地址构建计数；Scalar寻址另列边界。
    planner = min(tiles8, 47)
    pages = ceil_div(geom["compressed_rows"], 32)
    if p["SCALE_PAGE_COALESCING"]:
        for _ in range(b):
            for start in range(0, pages, 1024):
                valid = min(pages-start, 1024)
                log.add("前置metadata与页证明", planner, index32=1024, cast32=1024+valid,
                        arith32=2*1024+2*valid+8*valid, fill32=8*1024)
                log.reduce("前置metadata与页证明", planner, 8, valid)
    for block in range(tiles8):
        w, r, n, v = block % 48, 8, 8*512, 8*128
        log.add("前置metadata与页证明", w, fill32=64 if p["EARLY_ATTENTION_GEOMETRY"] else 0)
        log.add("Attention有效块计划", w, cast32=2*n+2*r+128, index32=128,
                arith32=18*n+2*r+7*v, fill32=2*n+3*r+v)
        if p["PLAN_PHYSICAL_ROWS"]:
            log.add("Attention有效块计划", w, arith32=34*n, cast32=4*n, gather32=n)
        else:
            # 无物理行计划时检查整页范围，每8行块内只缓存当前request。
            requests = len(set(token//s for token in range(block*8, block*8+8)))
            for _ in range(requests):
                for start in range(0, pages, 1024):
                    valid = min(pages-start, 1024)
                    log.add("前置metadata与页证明", w, cast32=valid,
                            arith32=8*valid+8*valid, fill32=8*1024)
                    log.reduce("前置metadata与页证明", w, 8, valid)
            log.add("Attention有效块计划", w, cast32=n, arith32=8*n)
        log.reduce("Attention有效块计划", w, r, 128, 4)

    for w in range(48):
        log.add("Attention在线归并与逆RoPE", w, index32=64+16,
                arith32=6*64+16+1024, fill32=1024+32*512)
    for token in range(t):
        for lane in range(2):
            w, heads, blocks = token % 24 * 2 + lane, 32, 5
            n = heads*128*blocks
            log.add("AttentionSoftmax", w, arith32=3*n, exp32=n, cast32=n)
            log.reduce("AttentionSoftmax", w, heads, 128, 2*blocks)
            n = heads*512
            log.add("Attention在线归并与逆RoPE", w,
                    arith32=blocks*(6*heads+3*n)+4*heads+3*heads*64,
                    exp32=2*heads*blocks+heads, div32=n, cast32=n+heads*64,
                    gather32=heads*64)

    qt_blocks = ceil_div(t, 32)
    for group in range(8):
        for block in range(qt_blocks):
            w, rows = (group*qt_blocks + block) % 48, min(32, t-block*32)
            n = rows * 1024
            log.add("O_A输出量化", w, arith32=2*n+2*rows, div32=rows, fill32=2*rows)
            log.reduce("O_A输出量化", w, rows, 1024)
            log.quant_casts("O_A输出量化", w, n)
            if block == qt_blocks-1:
                z = (ceil_div(t, p["OB_ROWS"])*p["OB_ROWS"] - t)*1024
                log.add("O_A输出量化", w, fill16=z, cast16_to_int8=z)
    for token in range(t):
        log.add("O_B反量化与HCpost", token % 48, arith32=53*4096,
                cast32=18*4096, fill32=4096)
    result = log.export()
    result.update(history=h, batch=b, query_tokens=s, tokens=t,
                  query_head_group=query_head_group,
                  sorting=sorting)
    # 独立守恒检查：矩阵后处理的真实head数、完整Softmax元素、原序HCpost。
    by_name = {stage["name"]: stage for stage in result["stages"]}
    assert by_name["AttentionSoftmax"]["counts"]["exp32"] == t*64*640
    assert by_name["主Query反量化RMS_RoPE"]["counts"]["gather32"] == t*64*64
    assert by_name["IndexerQuery反量化RoPE"]["counts"]["gather32"] == t*64*64
    assert by_name["O_B反量化与HCpost"]["counts"]["arith32"] == t*53*4096
    assert sum(sorting["sort_items_per_aiv"]) == sum(x["tile"] for x in sorting["segment_jobs"])
    # 每query的真实候选应恰覆盖一次；尤其防止G2把两个lane当成两条query。
    covered = [0] * t
    for job in sorting["segment_jobs"]:
        covered[job["query"]] += job["valid"]
    assert covered == [(h+token % s+1)//4 for token in range(t)]
    result["candidate_coverage_verified"] = True
    return result


def main():
    base = json.loads((HERE / "ANALYSIS.json").read_text())
    rows = [analyze(row) for row in base["rows"]]
    report = dict(
        status="all_vector_compute_stages_covered_source_work_model",
        exact_dynamic_instruction_trace=False, source=VECTOR_SOURCE,
        architecture="A2/A3、2201；不使用950/A5表", rates_elements_per_cycle=RATES,
        frequency_mhz=FREQ_MHZ, vectors=48, stages_covered=list(STAGE_SOURCES),
        limits=[
            "源级元素工作量及Tile通用归约模型；非逐条机器指令或精确耗时预测。",
            "当前最忙核服务量按真实worker分配求和；未加入Scalar、通知、DMA和指令头。",
            "额外UB复制、广播展开、mask切换及流水冲突只能使达到峰值更困难。",
            "提前耗尽根归并给处理量范围，不是实测置信区间。",
            "非矩阵数学均计Vector；页表Scalar、MGATHER行DMA不能误算TGATHER。",
            "high_precision rsqrt按当前A3 vsqrt+vdiv计；没有假设单vrsqrt满足精度。",
            "当前Tile归约口径可能被静态专用归约加速，另给理想归约敏感性。",
        ], rows=rows,
    )
    (HERE / "VECTOR_MODEL.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    for r in rows:
        print(r["history"], r["batch"], *(round(r[k], 2) for k in (
            "ideal_balance_min_us", "current_assignment_min_us",
            "current_assignment_max_us", "ideal_reduction_current_assignment_us")))


if __name__ == "__main__":
    main()
