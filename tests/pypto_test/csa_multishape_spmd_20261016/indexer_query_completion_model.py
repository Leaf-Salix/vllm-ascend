# SPDX-License-Identifier: Apache-2.0
"""B16长历史Indexer的query完成粒度模型；不估计NPU时间，不修改核体。

保留item=leaf*B+request、worker=item%24的原所有权，比较每核叶优先与
同一request先完成两种遍历。仅说明何时数据可能就绪；原整阶段通知不改变
时消费者仍不能提前执行。不能把score块数直接换算成微秒。
"""

import json
from pathlib import Path


def items(leaves, batch=16):
    return {worker: [(item % batch, item // batch)
                     for item in range(worker, batch * leaves, 24)]
            for worker in range(24)}


def frontier(order, leaf_cost, batch=16):
    end = {}
    per_request = {}
    for worker, work in order.items():
        cost = 0
        for request, leaf in work:
            cost += leaf_cost[leaf]
            per_request[worker, request] = cost
        end[worker] = cost
    precise = [max(per_request[w, request] for w in order
                   if (w, request) in per_request) for request in range(batch)]
    whole_stage = [max(end[w] for w in order if (w, request) in per_request)
                   for request in range(batch)]
    return {"query_data_ready_score_blocks": precise,
            "with_original_stage_notification_score_blocks": whole_stage,
            "worker_total_score_blocks": list(end.values())}


def analyze():
    covered = 0
    for count in range(1, 33):
        original = items(count)
        grouped = {w: sorted(work) for w, work in original.items()}
        for worker, work in original.items():
            assert sorted(work) == sorted(grouped[worker])
            assert all((leaf * 16 + request) % 24 == worker for request, leaf in work)
        flattened = [item for work in grouped.values() for item in work]
        assert len(flattened) == len(set(flattened)) == 16 * count
        assert set(flattened) == {(r, leaf) for r in range(16) for leaf in range(count)}
        covered += len(flattened)

    # 与原叶规划一致：decode含6个本轮token，最大compressed长度floor(131078/4)。
    cache_length = 131078 // 4
    tiles = (cache_length + 1023) // 1024
    count = ((cache_length + 8191) // 8192 + 2) // 3 * 3
    leaf_cost = [tiles // count + int(leaf < tiles % count) for leaf in range(count)]
    original = items(count)
    grouped = {w: sorted(work) for w, work in original.items()}
    before, after = frontier(original, leaf_cost), frontier(grouped, leaf_cost)
    assert before["worker_total_score_blocks"] == after["worker_total_score_blocks"]
    assert before["with_original_stage_notification_score_blocks"] == after[
        "with_original_stage_notification_score_blocks"]
    return {"scope": "所有权/完成粒度的CPU模型，非性能预测、非编译/设备证明",
            "device_tensor_reads_on_host": False, "math_changed": False,
            "legal_leaf_counts_checked": 32, "leaf_items_checked": covered,
            "sample_cache_length": cache_length, "sample_leaf_count": count,
            "score_blocks_per_leaf": leaf_cost, "original": before, "grouped": after,
            "necessary_followups": [
                "只重排叶循环而保留原阶段通知，消费者不会提前",
                "需拆分或新增按request发布的完成点，并重新证明AIC/AIV配对与FFTS排空",
                "merge每成员跨两个query，QK每成员跨四个query，需相应拆分消费粒度",
                "O-A覆盖整批96行，O-B的M96也要求整块输入；全链条收益不能从Indexer外推",
                "所有信号仍须保持代际、独占缓存行、DCCI/DSB及业务数据完成语义"]}


if __name__ == "__main__":
    result = analyze()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('legal_leaf_counts_checked', 'leaf_items_checked',
                                          'score_blocks_per_leaf', 'original', 'grouped')},
                     ensure_ascii=False, indent=2))
