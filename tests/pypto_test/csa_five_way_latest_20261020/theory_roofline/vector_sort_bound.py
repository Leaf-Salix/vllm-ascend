# SPDX-License-Identifier: Apache-2.0
"""长 G6、短 G2 Top-K 的 Vector 排序/归并峰值服务时间。

官方 A2/A3 Sort32、MrgSort 为2个排序项/cycle。按当前源码的512/1024/2048
分段、每叶累积归并及最终根归并计数，映射到真实48个AIV。
提前耗尽的最终3/4路归并给最少/至多处理量，不猜测实际分数排序分布。
这是 Vector 的一个子集，不能标为完整 Vector 或完整 CSA 下界。
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RATE = 2
FREQUENCY_MHZ = 1850
SOURCE = "https://asc.gitcode.com/api/appendix/vector_instruction_theoretical_perf_summary.html"


def analyze(row):
    history, sequence = row["history"], row["query_tokens"]
    geom = row["score_geometry"]
    assert sequence == 6 and geom["group_size"] in (2, 6)
    sort_items, merge_items = [0] * 48, [0] * 48
    active_roots = [0] * row["tokens"]
    segments = {"512": 0, "1024": 0, "2048": 0}
    segment_jobs = []
    for job in geom["leaf_jobs"]:
        if not row["policy"]["BALANCED"]:
            # 短 G2：两个 AIV 分候选域，每个 AIV 都处理组内两条 query。
            # 本轮每叶三次 K768，半叶容量1152；不能套长G6的按query分工。
            assert geom["group_size"] == 2 and geom["leaves"] == 1
            half_span = job["steps"] * geom["score_tile"] // 2
            for lane in range(2):
                worker = job["owner"] * 2 + lane
                for query in range(job["group"] * 2, job["group"] * 2 + 2):
                    visible = (history + query % sequence + 1) // 4
                    valid = max(min(visible - lane * half_span, half_span), 0)
                    assert 0 < valid <= 2048
                    tile = 512 if valid <= 512 else 1024 if valid <= 1024 else 2048
                    active_roots[query] += 1
                    segments[str(tile)] += 1
                    sort_items[worker] += tile
                    merge_items[worker] += (2 if tile == 512 else 3) * tile
                    segment_jobs.append(dict(worker=worker, query=query, valid=valid, tile=tile))
            continue
        first = job["group"] * 6
        begin = (job["leaf"] * geom["leaf_tiles"] + min(job["leaf"], geom["extra_leaves"])) * 1024
        for lane in range(2):
            worker = job["owner"] * 2 + lane
            for query in range(first + lane*3, first + lane*3 + 3):
                visible = (history + query % sequence + 1) // 4
                valid = min(job["valid"], max(visible - begin, 0))
                if valid <= 0:
                    continue
                active_roots[query] += 1
                segment_count = 0
                for offset in range(0, valid, 2048):
                    remaining = min(2048, valid - offset)
                    tile = 512 if remaining <= 512 else 1024 if remaining <= 1024 else 2048
                    segments[str(tile)] += 1
                    segment_jobs.append(dict(worker=worker, query=query, valid=remaining, tile=tile))
                    sort_items[worker] += tile
                    # 512两轮归并；1024两轮+两个512根归并；2048三轮全量归并。
                    merge_items[worker] += (2 if tile == 512 else 3) * tile
                    if segment_count:
                        merge_items[worker] += 1024  # 新512根与旧512根全量归并。
                    segment_count += 1
    low, high = merge_items.copy(), merge_items.copy()
    for query, roots in enumerate(active_roots):
        owner = query % 48
        for child in range(1, roots, 3):
            runs = min(4, roots - child + 1)
            # 两路无exhausted；三/四路在任一512项输入耗尽时停止。
            low[owner] += 1024 if runs == 2 else 512
            high[owner] += runs * 512
    low_us = [(s + m) / RATE / FREQUENCY_MHZ for s, m in zip(sort_items, low)]
    high_us = [(s + m) / RATE / FREQUENCY_MHZ for s, m in zip(sort_items, high)]
    return dict(
        history=history, batch=row["batch"], query_tokens=sequence,
        full_vector_model=False, subset="仅Indexer Sort32/MrgSort，不含其它Vector操作",
        segments=segments, sort_items_per_aiv=sort_items,
        segment_jobs=segment_jobs, active_roots_per_query=active_roots,
        local_merge_items_per_aiv=merge_items,
        merge_items_min_per_aiv=low, merge_items_max_per_aiv=high,
        peak_service_min_us_per_aiv=low_us, peak_service_max_us_per_aiv=high_us,
        ideal_balance_min_us=sum(low_us)/48, ideal_balance_max_us=sum(high_us)/48,
        current_assignment_min_us=max(low_us), current_assignment_max_us=max(high_us),
        score_cube_slowest_core_peak_us=geom["slowest_core_cube_peak_us"],
    )


def main():
    data = json.loads((HERE / "ANALYSIS.json").read_text())
    rows = [analyze(row) for row in data["rows"]]
    report = dict(
        status="partial_vector_bound_not_complete_csa_roofline",
        source=SOURCE, source_table="A2/A3表14；不同于A5节",
        elements_per_cycle=RATE, frequency_mhz=FREQUENCY_MHZ,
        assumptions=[
            "排序项是(score,index)队列元素，不是底层FP32暂存中的单个数。",
            "当前源码排序/归并在同一AIV按依赖顺序执行，按各指令峰值服务量相加。",
            "忽略指令头开销、UB读写/重排、Scalar、DMA、通知等待和其它算术。",
            "exhausted最终多路归并给处理量上下界，不是统计置信区间。",
            "长G6按query分工，短G2按候选半叶分工；分别复建实际循环。",
        ],
        rows=rows,
    )
    (HERE / "VECTOR_SORT_BOUND.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(row["batch"], round(row["current_assignment_min_us"], 2),
              round(row["current_assignment_max_us"], 2))


if __name__ == "__main__":
    main()
