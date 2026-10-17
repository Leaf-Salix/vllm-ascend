# Attention 物理行计划：有界只读方案审计

2026-10-18。未改源码、未编译、未测。结论是：**否决把现有 Scalar 地址循环直接搬进 plan；只保留“plan 内按 query 向量化解析物理行”的单一候选供后续评估。** 该候选与历史 UB 缓存/几何缓存并不相同，但没有证据承诺完整性能收益，本轮不追加设备任务。

## 当前真实工作及必须保留的外部输出

canonical `attention_plan_stages.py:133-307` 的 `attention_slot_plan_stage`：

1. `topk: INT32[T,512]` 是外部逻辑索引，保持只读；先按 Q/L、position 验证真实 owner 和 causal 关系。
2. `c_page_indices` 是遵循原 FP32 clamp、INT32 cast 边界后的最终逻辑候选。不能直接拿原始 topk 当页号，因为在极大整数边界两者可能不同。
3. `compressed_pages_all_valid`（同文件91行）先以1024列 Tile 验证本请求全部可能页列；同一 M8 tile 内相邻相同请求复用结果。全合法时每行候选页 mask 是向量运算，已经不做512次 Scalar 页表查询；页空洞或容量不足才逐候选回退。
4. 私有 `attention_indices[PLAN_CAP,512]` 保存逻辑候选或-1；bias、五个块的 mask 由原逻辑有效性产生。

`attention_stream_stages.py:309-322` 的压缩 gather：两 AIV 每块各64候选，共四块。每个有效候选仍 Scalar 读 index、除32求列、读 compressed_table、检查物理页上界、乘32加余数，最后 `gather_row`。两个 lane 总共处理每 query 的512项，并非每个 lane 都处理512项。

外部 `topk_indices`、`topk_scores`、Top-K 顺序、tie 顺序必须完全不变。仅私有 `attention_indices` 可以更换含义；bias 必须继续由逻辑 mask 生成，绝不能把物理行代入原 `min(c_out,0)*1e20` 或 causal 上界公式。

## 为什么不能直接平移 Scalar 循环

当前 plan 按 M8 分给48 AIV，但七档实际仅3～24个 AIV有 plan 计算；Attention 则由24组、48 AIV处理 query，每个 lane只处理256压缩候选。把512项 Scalar映射加到原 fast-plan 后，会重新引入已消除的逐候选页查找，并把它移到更少 owner 上，且所有 Attention 要等 plan 完成。

| 场景S6 | T | 实际plan owner | 每plan owner最多映射项 | 原每消费AIV最多地址项 | Scalar工作最大集中倍数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K B4 | 24 | 3 | 4096 | 256 | 16 |
| 128K B8 | 48 | 6 | 4096 | 512 | 8 |
| 128K B16 | 96 | 12 | 4096 | 1024 | 4 |
| 128K B24 | 144 | 18 | 4096 | 1536 | 2.67 |
| 128K B28 | 168 | 21 | 4096 | 1792 | 2.29 |
| 8K B16 | 96 | 12 | 4096 | 1024 | 4 |
| 8K B32 | 192 | 24 | 4096 | 2048 | 2 |

这是工作分配计数，不是时间预测。0有效候选的行实际可跳过，短可见历史也会少于512项；不应据表把所有输入都算成满512。总映射项最多512T，并没有因移动位置减少。只有向量化取代 Scalar 热循环，才有继续讨论的依据。

## 与历史三个方案及直接L1实验的区别

| 历史方案 | 实际做法与结论 | 本建议的区别 |
| --- | --- | --- |
| `dynamic_full_pipeline_20261017/qk_address_cache_v1` | 消费AIV每个64候选块载入页表到UB，用INT32位移/gather地址；256<C≤2048、cache行数≤2^31才开。边界及整链PASS，长B16 962.66μs，相对同轮原task874.08μs慢10.13%，无净收益，恢复原实现。 | 不在Attention的高UB占用/三槽循环内加Tile或重复载页表；将向量地址准备置于已有plan阶段，私有GM indices直接承载结果。 |
| `QK_QUERY_METADATA_EXPERIMENT.md` | 在每query第一tick一次算512地址，两AIV各自都算512；两个1×512 INT32 Tile共4096B跨tick携带；设备PASS但无明确收益，未采用。 | 每query仅一个plan owner算512，不重复两份；无跨Attention循环长期存活Tile，无新增GM数组。 |
| `dynamic_full_pipeline_20261017/query_metadata_v1` | plan仅保存request/position到mask第5/6列；消费者仍逐候选计算页地址。长B16慢1.66%，短B32快1.33%，默认关闭。 | 发布每候选物理行，不是两个几何标量；消除的是页推导/查页工作，不能拿几何复用成绩作为本方案收益。 |
| `QK_DIRECT_L1_EXPERIMENT.md` | 两AIV在Attention内把私有indices原地转物理行，另加FFTS4；搬运改由Cube逐行完成，并改变三槽ACK，完整1100.10μs，失败淘汰。 | 只在plan发布前解析；保持双AIV原64行采集、KV GM交接、AIC三槽L1及全部FFTS。历史直接L1整体失败不能单独归因于物理索引，但也不能忽略它已经做过“私有indices可改物理行”的接口验证。 |

所有时间均为旧版本/旧实验原始记录，仅供排重，不是当前候选或新基线测量。

## 唯一具体候选：有限容量的向量物理行计划

新增默认关闭的源码策略 `PLAN_PHYSICAL_ROWS`，不新增根Tensor、工作区、phase。生产者和消费者必须使用相同策略和同一个运行时适用谓词；建议首版谓词仅依赖已有descriptor：

```text
PLAN_PHYSICAL_ROWS
and 0 < compressed_table_columns <= 2048
and 0 < compressed_cache_rows <= 2**29
and compressed_cache_rows % 32 == 0
```

最后两个条件使物理页数≤2^24，页号范围检查可复用FP32的精确整数界限，合法物理行仍由INT32位移/加法产生；**禁止物理行经过FP32**。这不是要求实际只分配这些上限，而是UB页表容量和地址表示的保守能力范围。正式七档的页宽与cache行数必须依据实际descriptor确认，不由128K/8K标签猜测。超出该谓词时，生产和消费同时走完全原有逻辑索引路径；不得一个输出物理行而另一个再查页。

在适用分支中，每个query：

1. 保留原owner/causal、`c_page_indices` 数学。按实际 `C` 加载一份至多2048项的INT32 compressed页表，全部载入都带真实valid_shape；最后不足容量的未写UB不得被gather。
2. 对该行512个 `c_page_indices`，Vector INT32计算逻辑列与余数；为UB gather单独夹紧到 `[0,C-1]`。夹紧仅保障访存，原负索引、upper、列边界、物理页边界都继续参与有效mask。
3. `page = gather(table, safe_column)`；Vector检查 `0<=page<physical_pages`。非法page也要先夹到安全范围再位移，避免极端INT32负页/大页的溢出参与地址。用INT32计算 `safe_page*32+remainder`，再把无效候选设为-1。
4. 原logical mask与页mask形成bias和五块标记；物理结果写到原 `indices` 的对应行。Top-K外部张量完全不写。
5. 当前整页合法性预扫描可以由“本次选中512项页的向量合法性检查”替代，不应两套检查重复相加：未被Top-K选择的坏页原先只会触发慢回退，本身不会使合法候选失效。选择页逐项判定与当前fallback的输出等价，但仍须源码/地址oracle验证这一点。

为避免在plan内新增另一份长期 `[8,512]` Tile，首版可在每行向量完成后立即写原indices对应行。相对当前一次M8写出变成至多8次M1写出，GM总字节不增加，但DMA发射次数增加，这是明确成本。也可研究复用原 `c_page_indices` 的行存储后一次写出，但不能未检查编译alias就预称原地复用成功。本轮不实现两种变体，只以逐行发布作为保守可实施表达。

额外暂存为页表Tile至多8192B，以及512项Vector地址/mask/临时gather，局限在单个plan行；**不跨Attention tick存活**。完整UB能否容纳仍未编译验证，不能用逻辑容量直接判PASS。plan原M8分核保持，没有同时改48核分配来混淆收益来源。

## 消费端与同步合同

适用分支的压缩块只做：Scalar读取已发布物理行，检查非负，按物理行 `gather_row`；可保留一次物理行 `< cache_rows` 防御断言/guard，但不再查逻辑页表或重建request。每个plan行的owner/causal已经验证，失活/陈旧/dummy行必写全-1、mask0，不能沿用上一次物理行。

原始SWA块仍用原Q/L/positions和原页表，语义与五个run完全不变。若把压缩块的几何查询移除，应只在压缩分支中移除；SWA仍须原 `attention_query_geometry`。这会顺带去掉压缩四块中每query两lane共8次几何扫描，但不把它与已失败query-metadata开关混开。

`PHASE_ATTN_PLAN` 发布前必须完成物理indices的Vector GM store；现有发布fence/消费者cache维护保持。A3 Scalar读取物理行仍依赖这条发布/获取协议，不能因为“plan已算好”就去掉缓存维护。indices原本就由Vector写、其他AIV Scalar读，这里不新增通信类型。

mask前五列、KV_READY/SCORE_READY/PROB_READY/PV_READY、三槽覆盖约束、尾排空均不变；只换gather的地址来源。调用期间Q/L、positions、页表与cache descriptor不可改变；下一调用必须重建全部实际T行，T后容量不写不读。图重放里的页表交换和A→B→A必须重新计算物理行，不能按Host shape缓存地址。

## 收益范围、否决条件与后续门槛

可确定消除：适用分支每个有效候选的Scalar逻辑除/余数、Scalar页表读取、页上下界分支和地址乘加；最多512T次，但512T项地址仍需在plan内以Vector运算计算。GM indices仍写/读各一次，不减少其传输字节；新增/变化的是页表Vector加载、UB gather、逐行结果store，以及plan阶段集中负载。

因此该候选有区别于旧尝试的结构依据，但尚无证据优于当前fast-plan。若实现只能退化为plan的512次Scalar映射，或保留整页预扫描再重复加载/验证且没有实际Vector替代，则应直接否决，不上卡。若UB规划导致当前失败mi/li类似膨胀，也应有界收口，不改编译器强塞。

本轮只留设计。将来若执行，必须先核对真实生成ISA及地址oracle，覆盖负索引/负页/大页/窄页表、C=1/255/1025/2048及超界回退、mixed/odd/dummy/stale、T尾、INT32/FP32边界和页表A→B→A。原plan probe的indices期望值必须按内部策略明确区分逻辑/物理；不能为通过测试跳过它。随后完整精度/Top-K/cache-state/graph/保护/代际通过才可测同进程完整L1。当前没有新结果，不据局部工作量计数估算完整节省百分比。
