# SPDX-License-Identifier: Apache-2.0
"""交错流水的设备信号验收；仅在已同步的测试检查点读取，不进入计时窗口。

从本轮实际 group_plan/group_offsets 重建 wave 和真实写者，不用 B*S 或最大
history 猜测。ROOT 按有效叶、原全局 item 和双AIV有效lane确定写者；TOPK包含
dummy行；PLAN按全局M8块确定写者。AIC不能写这些新增阶段。
保留前一个检查点的期望矩阵：本轮使用的槽更新generation，未用槽保持原值。
因此可以检查首次清零、metadata变化后旧代际保留以及实际写者漏写/越界写。
这里只验发布覆盖和代际；缓存可见性、FFTS及数值仍须由设备执行和完整对照证明。
"""

from signal_checks import CORES, LEGACY_PHASES, WAVE_PHASES


def expected_wave_writers(*, tokens, requests, plan, offsets, policy):
    """返回各wave的范围及三个写者集合；输入来自本轮设备metadata快照。"""
    if len(plan) != 8 or len(offsets) < requests + 1:
        raise AssertionError("wave metadata长度不符")
    groups, _history, leaves, leaf_tiles, extra, group_size, balanced, roots = plan
    base = (requests + 16) // 16 * 16
    if not (0 <= groups <= tokens and offsets[requests] == groups
            and len(offsets) >= base + groups * 4 and leaves > 0
            and leaf_tiles > 0 and 0 <= extra < leaves):
        raise AssertionError("wave metadata范围不符")
    wanted = {"QUERY_GROUP": 6, "QK_PANEL": 64, "SCORE_TILE": 1024,
              "PREFETCH": 1, "TO_L0": True, "BALANCED": True, "ROOTS": 1}
    eligible = all(policy.get(k) == v for k, v in wanted.items())
    if not eligible or groups == 0:
        return []
    # plan[6] 来自 pl.cast(BALANCED, INT32)。当前编译器会将 i1 true
    # 符号扩展为 -1；它仍表示开启平衡策略。只接受两种 true 表示，
    # 不能把任意非零垃圾值也当作合法计划，更不能因此跳过逐核写者检查。
    if (group_size, roots) != (6, 1) or balanced not in (-1, 1):
        raise AssertionError("实际设备plan与交错策略不一致")
    descriptors = [offsets[base + 4 * g:base + 4 * g + 4] for g in range(groups)]
    previous_end = 0
    for owner, first, live, visible in descriptors:
        if not (0 <= owner < requests and 0 < live <= 6
                and previous_end <= first < first + live <= tokens and visible >= 0):
            raise AssertionError("group descriptor越界、重叠或无效")
        previous_end = first + live
    ranges = []
    gb = qb = 0
    while gb < groups:
        ge, qe = groups, tokens
        for g in range(gb + 8, groups):
            if descriptors[g][1] % 8 == 0:
                ge, qe = g, descriptors[g][1]
                break
        ranges.append((gb, ge, qb, qe))
        gb, qb = ge, qe
    # 算子仅在至少有两个wave时交错；单wave沿用原60阶段，不发布扩展槽。
    if len(ranges) < 2:
        return []
    result = []
    for gb, ge, qb, qe in ranges:
        root_owners = set()
        for group in range(gb, ge):
            _owner, _first, live, visible = descriptors[group]
            for leaf in range(leaves):
                begin = (leaf * leaf_tiles + min(leaf, extra)) * 1024
                capacity = (leaf_tiles + int(leaf < extra)) * 1024
                if min(visible - begin, capacity) <= 0:
                    continue
                item = leaf * groups + group if groups < 24 else group * leaves + leaf
                for lane in range(2):
                    if live > 3 * lane:
                        root_owners.add(24 + 2 * (item % 24) + lane)
        result.append(dict(
            group_begin=gb, group_end=ge, query_begin=qb, query_end=qe,
            writers=[sorted(root_owners), sorted({24 + q % 48 for q in range(qb, qe)}),
                     sorted({24 + tile % 48 for tile in range(qb // 8, (qe + 7) // 8)})],
        ))
    return result


class WaveSignalVerifier:
    """每个独立PreparedDynamicCSA一个实例；从全零初态递推扩展信号。"""

    def __init__(self, prepared):
        self.prepared = prepared
        self.phase_count = getattr(prepared, "signal_phase_count", LEGACY_PHASES)
        self.expected = [[0] * CORES for _ in range(self.phase_count - LEGACY_PHASES)]
        self.last_generation = 0

    def check(self, signal, generation):
        if self.phase_count == LEGACY_PHASES:
            return dict(passed=True, applicable=False)
        if type(generation) is not int or generation <= self.last_generation:
            raise AssertionError("wave检查点的代际必须单调前进")
        prepared = self.prepared
        requests = prepared.buffers["idx_lengths"].numel()
        plan = prepared.buffers["group_plan"].cpu().tolist()
        groups = plan[0]
        base = (requests + 16) // 16 * 16
        if not 0 <= groups <= prepared.tokens:
            raise AssertionError("实际group数超出T上界")
        offsets = prepared.buffers["group_offsets"][:base + 4 * groups].cpu().tolist()
        waves = expected_wave_writers(tokens=prepared.tokens, requests=requests, plan=plan,
                                      offsets=offsets, policy=prepared.policy.constexprs())
        if len(waves) * WAVE_PHASES > len(self.expected):
            raise AssertionError("wave数超出预留槽")
        expected = [row[:] for row in self.expected]
        for wave, info in enumerate(waves):
            for relative, owners in enumerate(info["writers"]):
                for owner in owners:
                    expected[wave * WAVE_PHASES + relative][owner] = generation
        actual = signal[LEGACY_PHASES:, :, 0].tolist()
        mismatches = [(LEGACY_PHASES + p, core, value, actual[p][core])
                      for p, row in enumerate(expected) for core, value in enumerate(row)
                      if value != actual[p][core]]
        report = dict(passed=not mismatches, applicable=True, generation=generation,
                      previous_checked_generation=self.last_generation, wave_count=len(waves),
                      waves=waves, mismatches=mismatches, tokens=prepared.tokens,
                      requests=requests, plan=plan, offsets=offsets,
                      scope="实际设备metadata及全部扩展word0；无host数学，不进入性能窗口")
        if not mismatches:
            self.expected = expected
            self.last_generation = generation
        return report
