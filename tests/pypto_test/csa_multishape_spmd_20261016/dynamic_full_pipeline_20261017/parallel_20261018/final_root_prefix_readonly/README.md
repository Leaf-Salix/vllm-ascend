# 最终 Top512 根前缀剪枝：只读可行性结论（2026-10-18）

**本轮不建议实施。** 现接口能表达“短前缀合并、严格认证、失败回退”，但没有直接
按输出 512 项停止的接口；当前最终阶段通常只有 1–2 次 merge。尚无认证命中率、
当前无插桩阶段成本或足以覆盖认证/重做开销的收益证据。本次未改源、编译或上卡。

## mrgsort 的真实能力

以当前隔离依赖为准：PyPTO 位于
`R/.worktrees/pypto-csa-static-fixes-20261016`；实际 PTO-ISA 位于
`R/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa`，
`R=/data/pyptouser/qinchuanyu/pto-eager`。不是 A5 接口推断。

- PyPTO `python/pypto/language/op/tile_ops.py:3410` 只提供输入、
  `tmp/exhausted` 或单输入的 `block_len`，没有输出项数参数。
  `src/ir/op/tile_ops/sort.cpp:190` 的返回 shape 等于 tmp。
- A3 `include/pto/npu/a2a3/TMrgSort.hpp:51–59` 只把 exhausted 编到
  “任一输入耗尽暂停”位；各输入实际长度分别来自 `GetValidCol()`（224–227行）。
  `vmrgsort4` 先写 tmp，126–127行读取状态、再复制 tmp→dst；
  dst 有效宽度仅限制这次复制，不限制硬件归并。缩小 tmp 不能充当 TopK 截断。
- 不同输入长度可表达：PyPTO 只要求类型一致；PTOAS
  `lib/PTO/IR/PTOTile/PTOMergeSortMultiplyShift.cpp:144–176`
  逐输入累计有效列数，不要求各输入等长，且要求 tmp 容纳全部输入有效列。
  因而“累计根512项＋新根128项”不是接口禁区。
- 当前 `indexer_topk_stages.py:78–105` 的三/四路已经 exhausted=True，
  每输入512项，耗尽时保证至少已输出512项；二路仍普通完整 merge。
  换成128项输入后，耗尽只保证128项，不能仍取512：
  前缀尝试须普通完整 merge。PyPTO只返回结果Tile，executed计数由后端临时生成，
  未暴露给调用者（`src/backend/common/pto_ops_datamove.cpp:881–943`）。

## 认证可在设备直接完成，但并非无成本

`pl.tile.read` 返回同dtype Scalar（`tile_ops.py:360`），经
`pto_ops_memory.cpp:637–663` 生成 `pto.tgetval`；
实际 ISA `include/pto/common/pto_tile.hpp:1514–1518` 的
`Tile<float>::GetValue` 直接解引用 UB。无需把候选根先写 GM，
也不需要为认证新增 GM store/fence/cache 维护。

现成完整生成物已存在 FP32 UB→Scalar→比较：
`../query_dequant_group_20261018/cpu_v1/kernel/ptoas/dynamic_csa_full.cpp:56390–56400`，
其 GetValue 前有 V→S set/wait。认证的候选第512 score 在交错pair偏移1022；
每个原根首未读的第129 score 在偏移256，须额外载入该分数或对齐小块，
再做 MTE2/V→S 获取、R+1次Scalar读、R次严格比较和设备分支。
原 arena 已发布，但这些新增本地流水依赖不能省略。

## 精确性条件与实际根数

先以相同三/四/二路分支及 incoming 顺序合并各根前128项，保留候选Top512。
只有所有原根第129项 score **严格小于** 候选第512项时才接受。
有序根的后缀不会再有更高分；删除严格低于阈值的后缀不影响Top512，
保留原merge树、参数次序及根内次序可保留入选同分顺序。相等、padding同分、
阈值/哨兵无序比较都须回退；不得用 ≤ 或重新按index排序。
认证前须只读原 arena，失败才执行原合并，不能先覆盖根0后回退读取。
这属于接口/数学可行性，未做新设备精度证明。

当前长G6、ROOTS=1的实际计划见主 PAR
`indexer_query_group_audit/README.md:88–100`，
公式为 `indexer_group_stages.py:85–107`、`indexer_publish_stages.py:81–118`：

| 当前S6工作量 | 本query真实根数 R | 原最终merge次数 | 128前缀的限制 |
| --- | ---: | ---: | --- |
| 128K B4/B8/B16 | 6 | 2 | 最终Top512不能从任何根取超过128项；命中率未知 |
| 128K B24/B28，visible=32768 | 4 | 1 | 每根必须恰贡献128项，且截断边界不能同分 |
| 同档，visible=32769 | 5 | 2 | 第五叶仅1个有效候选，不能当五个同等大小根 |
| 当前8K G2 | 2 | 1 | 总前缀仅256项，不足512，不能走此128尝试 |

R5若尾叶那1项入选，前四根合计贡献511项且各自≤128；
若它未入选，则前四根仍须各128项。R6没有这个恰好条件，但没有实测通过率。
Mixed Q 必须用每query的 visible/root_count，不可用统一最大根数替代。

原最终根载荷只有每query 16/20/24 KiB（R4/5/6），不是重排32768候选。
成功尝试可把输入降到约4/5/6 KiB加哨兵，但仍要1–2次普通归并、截根和认证；
失败会额外做一遍前缀路径再做原合并。S1/S5、集中分数、全相等及短尾不能假定高命中率。

## 历史排重与价值判断

VALIDATION §318–319、§324 已有四路耗尽暂停和UB累计根，原报告为
主仓 `tests/pypto_test/results/csa_topk_fourway_20260928/README.md` 与
`csa_topk_ub4_20260928/README.md`。后者旧分task长B16根阶段
13.072→11.539 μs，而完整CSA只1109.682对1111.950 μs；不是当前动态消融。
检索 MASTER、VALIDATION 和这些候选报告，未找到同样的短前缀严格认证方案。

已有593长B16插桩 `PAR/current_clock_h131072_b16_v1/phase_summary.json`
的 PHASE36 去GM等待 other 中位为9.60/9.00/9.80 μs/核，包含两query的
几何、根合并、拆分发布与插桩；不是纯merge，也不是最新正式阶段耗时。
所以**当前收益上界未知，不能从这些数据承诺少量μs收益**。
在已有耗尽暂停/UB根、仅1–2次最终merge的前提下，新增认证和失败重做
尚无足够收益依据；本轮收口，不扩展候选或测试。

