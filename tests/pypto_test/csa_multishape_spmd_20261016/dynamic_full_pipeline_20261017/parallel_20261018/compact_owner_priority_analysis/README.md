# 仅真实 compact Hadamard owner 在主 QB 前执行：只读审计

2026-10-18。结论：**在下述合同下可以无环实现，但当前证据不支持继续制作设备候选，
本轮否决推进。**原因不是数学依赖禁止前移，而是 2～4 个真实 owner 已承担较重的
QB/主压缩任务；已有长 B16 端点显示，它们在 QB 原起点还拿不到 IDX_NORM。
即使提前 Had，compact 后续仍需等 COEFF，Score 又受全 scale 发布及双槽背压限制。
因此不能把 Had 提前的时间直接解释为整层缩短的时间。

只读起点为 canonical 正式 `deepseek_v4_flash_csa_spmd/{cube,vector}_schedule.py`
已经采用的 compact 优先版本。没有改正式源码、冻结候选或依赖，没有编译/上卡，
没有提交新性能任务。本文唯一新增模型是源级有限偏序及工作量枚举，不是性能模拟器。

## 1. 和已做方案的区别

- 历史 `csa_multishape_spmd_20261016/deferred_main_q_schedule_analysis.json` 把主
  Q 投影/反量化整体降低优先级。两种固定旧成本模型均为 799.38 μs，对照 773.43 μs；
  当时没有生成设备候选。该数字属于历史成本模型，不是设备性能。
- `ATTEMPTS.md:1896` 的固定分核整组延后主 Q 后处理也无收益：旧共同 MIX 开始
  模型 773.41→788.94 μs；Indexer 最早开始虽提前，最后结束没有提前，Attention
  反而后移。这否定了“主 Q 无 Indexer 数学依赖，所以全面后移必有利”的推论。
- 已实测并迁回正式的 `parallel_20261018/compact_priority_evidence/README.md`
  调整为 AIC `QB→compact Had→MAIN→Score`、AIV `COEFF→compact quant/scale
  →QUERY→MAIN→Score`。本审计以这个版本为起点，不重复其性能归因。
- 此次仅评估：真实有 compact Had 工作的 AIC 改为 `Had→QB→MAIN→Score`；
  其他 AIC 保留 `QB→原 Had 等待/空发布→MAIN→Score`。不减少通知，不移动 AIV，
  不重分 Had/QB/MAIN/Score 的 owner，不改变数学块或参数。

在 MASTER/ATTEMPTS 及上述专项记录中未找到这个精确的 owner 条件变体已实测记录；
这不意味着值得实现。已查历史只能排除重复，不能提供新收益。

## 2. 真实行数、owner 和七档工作量

`metadata_stages.py:65` 的设备扫描使用 Native Q/L：对活动请求 r，
`S_r=Q[r+1]-Q[r]`，`H_r=L[r]-S_r`，
`C=sum(floor(L_r/4)-floor(H_r/4))`。dummy 不累加；陈旧 position 不改变分配的
compact 行号，后续 norm/cache 再按 causal/slot 校验。**C 不是 T/4。**

七档 fixture 的定义来自 `tests/pypto_test/dsv4_csa_single_layer.py:85`：所有
请求 S=6，`L=history+6`，position 从 history 开始，history 都整除 4；故每请求
恰好关闭一行，`C=B`。H mod4 为 0/1 时 S6 每请求产出 1 行，2/3 时产出 2 行；
mixed/odd 请求必须继续读设备 Q/L 计算。任何实现都不能把本文七个 C 固化进源码。

`indexer_compressor_output_stages.py:177`：`R=ceil(C/16)`，共有 `2R` 个
M16/N64/K128 Had tile，`owner=logical%24`，`Mblock=logical%R`，
`Nblock=logical//R`。有工作集合恰为 `[0,min(2R,24))`，但 C 大时每核可以多 tile。
C=0 时没有数学 owner，不能伪造等待对象；容量仍须覆盖 pad16(C)。

当前 Native NZ2 的 QB 使用 M128/N256/K256，32768 输出列共 128 列块：
`Nblock%24` 分核，每核处理全部实际 M128 块。AIC0～7 有 6 个列块，8～23 有 5 个；
下面表中的 QB 数量已乘 `ceil(T/128)`。MAIN 是 M64/N64，1024 输出列共 16 块，
`logical%24` 分核；每 tile 各做 wkv/wgate 两条独立 K4096 归约。

| 场景 | T / C | Had owner（每核 1 tile） | QB M128 tile：0～7 / 8～23 | MAIN tile/core | Score steps/core |
| --- | ---: | --- | ---: | --- | --- |
| 128K B4 | 24 / 4 | 0、1 | 6 / 5 | 0～15:1；16～23:0 | 0～11:6；12～23:5 |
| 128K B8 | 48 / 8 | 0、1 | 6 / 5 | 0～15:1；16～23:0 | 全部 11 |
| 128K B16 | 96 / 16 | 0、1 | 6 / 5 | 0～7:2；8～23:1 | 全部 22 |
| 128K B24 | 144 / 24 | 0～3 | 12 / 10 | 全部 2 | 全部 33 |
| 128K B28 | 168 / 28 | 0～3 | 12 / 10 | 全部 2 | 0～3 为 41；全范围 33～41 |
| 8K B16 | 96 / 16 | 0、1 | 6 / 5 | 0～7:2；8～23:1 | 全部 6 |
| 8K B32 | 192 / 32 | 0～3 | 12 / 10 | 全部 2 | 全部 12 |

Score step 是一次 FIFO 生产/消费，不是时长。长档 G6/balanced：B4/8/16 采用
6 leaves，B24/28 采用 5；短档 G2/nonbalanced 各 query group 一个 leaf。
由于本轮最后 query 已见 32769/2049 个压缩候选，而不是恰好 32768/2048，尾 step
必须保留。具体 `(group,leaf)`、每核 QB 列号、MAIN 有效 M 行及 Score steps 全列于
`evidence.json`，不是从执行顺序猜 owner。

低编号 Had owner 同时属于 QB 的 6 列块重负载区。B16 两名 Had owner 还承担两份
MAIN tile，而 16 名 AIC 只有一份；B24/B28 虽然每核都是两份，0～15 的有效 M
行合计 128，16～23 分别为 80/104。B28 Had owner 0～3 又恰好承担最多的 41 个
Score step。没有“少数核只做轻活、可免费插入 Had”的分工依据。

## 3. 既有长 B16 原始时钟给出的反证性约束

只读取 `current_clock_h131072_b16_v1/clock_round_{0,1,2}_spmd_reference/
phase_clock.json`。该作业使用 593c5d25 原顺序，Had 仍在 MAIN 后；**不是本轮
已迁回 compact 优先版的阶段测量，也不是所评估 owner 提前版的测量。**
数字是已插桩 notify 的端点，包含调度/等待/流水/插桩影响；不视为独立算子时长。

| 轮次 | Had owner0/1 的 QB 前一通知结束 | 实际 IDX_NORM owner 最晚结束 | 两 owner 此处尚缺就绪时间 | 全 QB 最晚结束 | 全 COEFF 最晚结束 |
| --- | --- | ---: | --- | ---: | ---: |
| A0 | 175.52 / 177.44 | 190.44 | 14.92 / 13.00 | 247.82 | 252.88 |
| B1 | 174.42 / 174.22 | 191.24 | 16.82 / 17.02 | 246.20 | 256.40 |
| A2 | 171.96 / 172.02 | 192.12 | 20.16 / 20.10 | 242.60 | 255.84 |

单位 μs。这里取的是 B16 真正 NORM 请求 owner AIV0～15，不拿无数学工作者的
发布中位数替代；Had 数学 owner 是 AIC0/1。旧 QB 区间约 68.26～70.38 μs，
旧 MAIN→Had 通知间隔为 3.36～6.16 μs，但后者包含 META/NORM 获取及发布，不是
纯矩阵运算时间。没有把它从端点剪下作为新版本的固定服务时间。

这些数据至少排除了“IDX_NORM 在 QB 开始前已经就绪、提前等待免费”的说法。
与此同时，COEFF 最晚结束比全 QB 晚 5.06/10.20/13.24 μs；即使 Had 极早完成，
每名 AIV 的 quant/scale 仍排在自己的 COEFF 后，Score 又等全 48 个 scale 发布。
因此早 Had 存在撞上另一道门的明确风险。本文不将旧端点差当新方案时延预测或
收益上界；尤其不能据此声称“新方案一定慢 13～20 μs”。

## 4. 局部代数：为什么重排本身不会提前 owner 的 MAIN

仅为说明串行顺序，设单核到 QB 前位置时间为 t，IDX_NORM 可读时间为 n，QB
服务时间为 q，Had 为 h，随后 MAIN 为 m。忽略资源竞争和新增分支开销，当前
compact 优先版该核 MAIN 结束为 `max(t+q,n)+h+m`；owner 提前版为
`max(t,n)+h+q+m`。

两式差为 `max(t,n)+q-max(t+q,n)`，即
`min(q,max(n-t,0))`，非负。n≤t 时只是交换同核两段，MAIN 不提前；n>t 时先等
norm 吞掉了原来可跑 QB 的时间。其余核的 Had 原本无数学工作、QB 顺序也未变，
不凭空缩短自己的 MAIN。真实执行还可能有 L1/Left/Right 分配和带宽变化，故这
只是固定服务时间下的依赖结论，不是硬件时长定理。

可能的收益只剩：Had 早发布 → 部分 AIV compact quant/scale 更早 → 全 scale
门更早 → 已完成 MAIN 的 AIC 提早生产 Score。若 COEFF 已经卡住 compact，或
全 QB/MAIN 门又卡住两个 AIV，这条提前量就无法完整传至 Score/Attention。

## 5. 无环必须同时满足的条件

1. 先完成当前 QB 前的所有 `IDX_PROJECT / IDX_QUERY_PROJECT / WEIGHTS_PROJECT /
   KV_PROJECT / IDX_QUERY_HAD` 生产与发布，再进入 owner 的 IDX_NORM 等待。
   NORM 的生产链不可以再等待 QB、MAIN 或 Score。否则只移动“看似独立”的 Had
   代码可能把尚未发出的前驱一起堵住。
2. owner 判定来自已获取 PHASE_META 后的实际 device `idx_count`，按上文循环
   算法。全 24 核要先知道该值才能分支，可能新增/前移 META 等待和 Scalar 分支；
   不把该开销隐含为零。不用 Host `.item()`，不按 B 或 T 推算 C。
3. 每个工作核发布一次真实 Had 完成，之后做 QB 并发布 QB；其他核保留 QB 原位。
   AIV 继续只等真实 `2*ceil(C/16)` 对应 owner；若改成全 24 核，会增加无关等待。
   阶段槽是 `(phase,core,generation)` 独立槽，允许 phase30 比 phase22 先发布，
   不能改成“阶段号代表已完成此前所有阶段”的单调计数。
4. 保留 `publish_phase` 的流水完成和缓存发布、`wait_*` 的获取/cache 维护。
   不因旧时钟显示生产者先完成就删同步。Native state/cache 仍遵守正式入口的
   不重叠/明确只读别名及单流工作区合同；Had 的 GM 输出不得与 QB/MAIN 暂存复用。
5. AIC 在进入 Score 之前仍完成自己的 QB 和 MAIN 并发布，AIV 仍完成自己的
   compact scale→主 Query→主压缩链。AIV 绝不能先等 `PHASE_SCORE_CUBE`。
   即便其他 AIC 很早进入 Score 并填满两槽，被推迟的 owner 仍能不依赖 ACK 地
   完成 QB/MAIN；AIV 前置门随后可打开，才有 ACK，故不存在环。
6. Score 两槽 READY、两 lane ACK、第 3 步前的复用等待及每 leaf 最后
   `min(steps,2)` 次 drain 全保留。空 lane 也必须先读/ACK 再做有效行分支。
   两槽只能缓冲最多两次未消费的生产，不能把它当作允许整个 Score 无界提前。

局部前移不新增数学写地址；不过当前分析不检查新分支后的后端内存分配、真实
FFTS/缓存可见性或数值。无环只是其中一个必要条件，不等于可发布实现已验证。

## 6. 唯一有限模型及复现

`analyze.py` 仅标准库，读取正式源码记录函数范围，按上述源公式列出 owner，
构建当前/owner 提前两种偏序。保留 24 AIC、48 AIV、全 scale 门、全 QB 门、
主压缩投影 owner、活动请求 pool 集合，以及逐 leaf 两槽两 lane ACK/drain。

七档加 6 个几何边界，共 13 个案例、26 种排序无环。边界包括 mixed odd/零关闭、
H mod4 改变 C，以及仅为核分工公式检查的 C=192/224（24 owner、多 wave）。
较大 T 的边界只检查这一个偏序抽象，不表示正式整层入口支持/精度通过。
负例主动在 AIV Score 前加入等待 CUBE_DONE，模型正确检出环；不能称模型已验证
所有隐藏依赖。模型抽象掉数值、缓存、tile liveness、MTE/Scalar 竞争及硬件事件。

复现（只输出 JSON，不启动编译或设备）：

```bash
python analyze.py --repo /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
```

`evidence.json` 保存每核列表、所有偏序结果、源函数范围及三轮原始端点摘录。
本次没有 hash、二进制或新设备产物；没有更改旧验证结果。当前停止于此审计，
不建议仅为该重排再编译/上卡；若未来有当前 compact 优先版的实测端点证明
Had/scale 仍独占关键门且 owner 的等待可被真正隐藏，再重新评估，而不是直接
沿用此处旧版本端点作证明。
