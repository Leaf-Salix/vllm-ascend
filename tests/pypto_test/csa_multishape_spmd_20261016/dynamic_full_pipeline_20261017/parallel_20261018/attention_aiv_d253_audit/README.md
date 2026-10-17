# d253 Attention AIV 五块推进与搬运：有限只读复核

2026-10-18。**本轮没有找到尚未被既有试验覆盖、且证据足以推进的最小候选。**
不重做两 AIC orphan，不改源码、不编译、不上卡、不建新模型或复制大生成物。

当前正式 33 份 Python 与主 PAR 的
`hc_active_notifications_evidence/source_candidate` 逐字相同。核对实际已执行生成物为
`hc_active_notifications_h131072_b16_v1/build_output/decode_csa_spmd_3kuwaqat/ptoas/`
下的 `dynamic_csa_full.cpp/pto`；同次 `4ai0h2s2` 是 d56 对照，未混用。
绝对路径和大小见 `evidence.json`。主 PAR 的既有
`attention_readonly_audit/README.md` 已对原分 task 与融合版本作数学、事件和体积对照，
本轮重点复核现 d253 搬运结构及历史方向的真实边界。

## 当前五块之间哪些是往返，哪些是复用

源码均位于正式 `attention_stream_stages.py`：

| 路径 | 现实现与可删除性 |
| --- | --- |
| KV，:267–323、AIC :151–155 | 两 AIV 各采 64 候选到 UB，写各半 GM ring；AIC 等两 lane 的 KV_READY 后一次读入 L1。SWA 与四个压缩块是不同候选，没有保证它们的 KV 行相同，不能直接跨块复用。 |
| QK/PV 的 KV，:154–157、:177–196 | QK 用 L1 转置视图，PV 用同一 L1 槽；已经没有第二次 KV GM→L1 搬运。q_flat 在 :145–146 每 query 只加载一次。 |
| Score、probability、PV，:361–379、:390–403 | 在 AIC 与 AIV 间交换真实数学结果；在原角色分工下不是同核冗余。不能删除传输而保留原消费者。 |
| mi/li，:377–378、:392–393 | 同一 AIV 写后两拍回读，语义上可本地保存；三槽 UB 候选已完整 CPU 超容量，不重开。 |
| running m/l/oi，:254–261、:394–406 | 已经保留在本核 Tile，五块按原序在线更新；末块直接归一化/RoPE/packed，没有整 query 中间结果写 GM 再读。 |
| 几何，:274–276 | 默认 query-metadata=false，每个有效块、每 lane 重扫 Q/L 并验证 position，最多每 query 十次。此重复已有动态几何复用的同进程反证。 |
| 压缩 index/page，:309–321 | 每 lane 每块读取自己的 64 项。四块合计 512 项/两 lane，不是两 lane 各 512；plan 已验证有效性，但消费仍作页表与物理容量保护。地址缓存及物理 row plan 已覆盖这一方向。 |
| mask、最终 cos/sin/sink | mask 被采集/softmax/归并各自读取，服务不同在飞行 work；旧 mask UB 缓存无屏障减少证据。cos/sin 在末块两个 16-head 发布段重复加载，每 lane 仅重复 512 B，不是五块中的主要 KV 往返；没有证据把这点提升为关键路径候选。 |

五块全有效时，每 query 的**逻辑**量：KV 原 cache 读取、写 ring、AIC 读 ring
各 640 KiB；Score 交换每方向 160 KiB，probability 每方向 80 KiB，PV 每方向
640 KiB；mi/li 同核写与读各 2.5 KiB。它们不是实际 HBM 流量或可节省时间。
绕过 KV ring 理论上少的是两个方向共 1.25 MiB/query，但不能忽略替代路径的
执行核、地址准备、ND2NZ 粒度与三槽流控成本。

## 现成物理代码给出的约束

- 当前 C++ `:58190–58288` 的压缩行循环含 index/page 保护和单行 TLOAD，
  无循环内显式 `wait_flag`/`pipe_barrier`；`:58287` 回到循环头。`:58291–58321`
  在块末汇合 SWA/压缩分支，等待 MTE2→MTE3 后一次写 64×512 BF16 到 ring。
  不能再用“同目的 Tile 所以每行必等上一行”作为独立行流水的依据。
- 物理 PTO `:12623`：KV 半块使用 UB `[65536,131072)`；softmax reduce
  (`:13034`) 与两次 PV 半宽加载 (`:13155/:13166`) 也从 65536 起复用。
  running oi 两半位于 0 和 32768，各 32768 B。把 gather 与后续数学强行
  同时存活需要重新分配这些冲突区；不能只换语句次序或删编译器事件。
- AIC PTO `:3685–3686`：query L1 从 0 起 65536 B，KV 三槽从 65536 起
  393216 B，transpose view 指向同一槽。已有复用不是源码注释上的假设。
- 每 tick 的 gather、softmax(tick−1)、merge(tick−3) 共用同一 AIV，不能假定
  一个核同时执行两个 Scalar 路径。T<144 的 KV 发布仍在上一块 softmax 后；
  T≥144 先消费上一块 SCORE 再发 KV。它们同时限制三槽超前，不能任意提前。

## 历史排重：已确认失败点与尚不能归因的部分

以下 MASTER 指 `vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`。
旧静态结果没有同进程现选配对时，不用跨作业中位差解释精确回退。

| 方向、证据 | 真正已知的结论 |
| --- | --- |
| direct-L1；MASTER :9991–10061、`QK_DIRECT_L1_EXPERIMENT.md` | 全链 PASS，候选1100.10μs，同轮原 task892.66μs；没有同轮现选SPMD消融。已把双AIV各64行改为单AIC 128行 Scalar地址读取+单行ND2NZ，并增加物理地址交接/更换槽ACK。其行循环没有新增逐行屏障。搬运集中化/粒度是合理疑点，尚无证据把总回退定量分配给它们；不能据“去GM往返”重开。A3已有TPUSH/POP Vec→Mat实现仍走GM FIFO。 |
| MGATHER；MASTER :6637–6674、`qk_mgather_result.json` | 首版因跨分支零Tile存活导致UB200960>188416 B；修正后全链PASS，821.62μs，无同进程现选配对且未获采用。ISA仍逐行DMA，QK静态PIPE_ALL由4到5；没有证据保证它减少有效搬运或实际屏障耗时。 |
| 独立行Tile stage2；MASTER :11718–11792、`qk_row_load_pipeline_v1/lowering_audit.json` | 完整CPU中两临时行共用同一UB地址，实际顺序LOAD→wait→TMOV→LOAD→wait→TMOV；if边界使期望预取未形成，每有效行另加1KiB UB拷贝。基线本来就直接写不同子行且无逐行显式wait，因此停止上卡。 |
| mi/li三槽；PAR `attention_local_ring_evidence/README.md` | 768 B逻辑状态的两种表达分别导致整体Vec规划255232/255104 B，均超过188416 B。环本身没有被扩成255KB；具体是整体长期生命周期布局失败，不能据逻辑尺寸小宣称可用。 |
| 物理row plan；MASTER :16260–16285、PAR `physical_plan_evidence/README.md` | 页表/整型/重放及整链PASS。同进程旧904.78→897.44μs，仅0.811%，未采用。它已消除消费侧逐候选页推导，但还把工作集中到plan及其发布之前；没有阶段细分数据可把弱收益精确归因给plan或DMA。 |
| query metadata；MASTER :15987–16031、D `query_metadata_v1/result_summary.json` | plan保存request/position，长B16同进程896.20→911.06μs，短B32 1036.34→1022.52μs，方向相反，默认关闭。不能断言是plan写、寄存器或同步导致。旧静态每query地址缓存还让每lane算512项而原来仅256项，并加4096 B跨tick状态，未有明确收益。 |
| 其它重包装边界 | 动态每半块UB页地址缓存已全链PASS但无净收益（MASTER :15560–15585）；32行KV分块已PASS但未采用（:6975–7002）；提前KV阈值96、双AIV搬运/数学职责拆分均已有试验（历史文档076/079）。本轮不再提出这些同义变体。 |

## 收口

没有发现能够直接删除且尚未试过的大额 GM 往返。对 KV ring 的替代已经触及
核分配与槽流控；单行加载流水的原“串行等待”前提被实际代码否定；剩余重复
几何/地址与本核统计回读均有现成试验覆盖。因此本轮不提出实现或设备任务。

还缺的是**同一当前版本 Attention AIV 内部的成本区分**：源KV采集与ring写出、
softmax/prob发布、PV归并/最终发布分别占多少关键推进时间，以及哪些等待来自
UB复用。现有phase_clock只说明AIC在等AIV推进，不能判定瓶颈是Native gather、
Scalar地址、向量计算还是本核依赖。没有这条观测，不足以选择“只改SWA直达L1”
等旧方向子集；本轮不据此追加插桩或新测。
