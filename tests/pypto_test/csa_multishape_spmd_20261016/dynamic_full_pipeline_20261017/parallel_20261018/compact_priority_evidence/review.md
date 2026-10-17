# Compact 优先编排：独立只读依赖与 FFTS 协议审阅

审阅日期：2026-10-18。对象为本 worktree 相对 `593c5d25` 的
`cube_schedule.py`、`vector_schedule.py` 修改。仅阅读源码和 diff；本次审阅没有
编译、设备提交、性能测试或生产代码修改。下列位置均指本 worktree 的
`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`。

**结论：在既有入口缓冲不重叠、合法写 slot 唯一、单流代际及 FFTS 事件合同下，
未发现这次阶段前移新增的数据依赖缺失、等待环或环槽过早复用。可继续由主会话
验证编译及设备行为；本结论不是候选精度、可见性或性能 PASS。**

## 1. 实际改动与缓存依赖

| 改动 | 保留的获取、生产和发布 |
| --- | --- |
| AIC 的 compact Hadamard 移至 QB 后、主压缩投影前 | `cube_schedule.py:312` 已发布 QB；`316` 起仍等待本轮 META 和全部实际 IDX_NORM 请求 owner，随后读取 device `idx_count`、执行同一 Hadamard，并发布 IDX_HADAMARD。主压缩投影原体及 MAIN_PROJECT 发布仍在其后。 |
| AIV 的 compact key 量化与 scale 提交移至 COEFF 后、主 QUERY 前 | `vector_schedule.py:712` 起仍按 `ceil(compact_count/16)*2` 等待 Hadamard 的 AIC owner；量化后发布 IDX_QUANT，等待所有实际请求量化 owner，执行同一页所有权 scale 提交，然后发布 IDX_SCALE。 |
| Score AIV 不前移 | `vector_schedule.py:815` 起仍在主 QUERY、主 pool/commit/output 之后；仅等待输入 IDX_SCALE，不等待整个 PHASE_SCORE_CUBE。 |

紧凑 Hadamard 的原 owner 是列块优先的 M16/N64 网格
（`indexer_compressor_output_stages.py:190` 起），所以 AIV 的 `had_tasks` 公式仍匹配
真实生产者。C=0 不读取 Hadamard 输入；空量化参与者/scale writer 仍按原合同发布，
没有因前移增加一个等待永远不发布的空 owner。

实际可见性来自保留下来的协议，而不是“另一段计算应该已经花了足够时间”。
`phase_sync.py:84` 的发布保留 fence/cacheinvalid/fence 与 notify；
`phase_sync.py:121` 在全部实际 owner 信号到达后统一获取业务缓存。
AIC 进入 Score 前仍等待 query INT8、COEFF 及全部 48 个 IDX_SCALE writer
（`cube_schedule.py:348`）；AIV 进入 Score 前同样获取全部 48 个 IDX_SCALE。
因此 `idx_key0/idx_key64` 只读别名是在 key/scale 写回发布并获取后消费。
本次未修改这些 helper 或底层 signal 实现。

## 2. 两槽 Score 的三角色协议

直接审查 `indexer_score_stages.py`，不是用阶段模型替代 FIFO 协议：

- AIC 在 `174` 起按 step 顺序生产，槽号为 `worker*2*G+(step%2)*G`。
  step 0/1 写两个不同槽；step≥2 在覆盖该槽前等待 CONSUMED。末尾 `375`
  通过 FIX 发布 READY；每个 leaf 在 `377` 起排空 `min(steps,2)` 个剩余 ACK。
- balanced AIV 在 `469` 等 READY、`470` load 自己的 G/2 行，再在 `473`
  通过 MTE2 发 ACK。非 balanced 路径在 `641/642/647` 做同样的等待、load、ACK，
  两 lane 读取各自半列。ACK 保护的是 GM 槽已被 MTE2 读走，而不是 Top-K 已算完。
- 有效 query 分支位于 READY/load/ACK 之后。尾组中没有有效 query 的 lane 仍消费
  协议，只跳过 query 数学/根写回。空 work item 由三角色同一 device plan/geometry
  判定，一同不发事件。
- Score 使用事件 0/1；Attention 使用基址 2 的事件 2..5。前移的 compact、
  主压缩、QUERY helper 没有使用这些跨核 sync_set/sync_wait。没有新增事件复用。

因此，允许 AIC 提前放入两份分数、随后等待较晚到达的 AIV，没有新增第三个未释放
槽，也没有把阶段 notify 误当作逐槽 ACK。两槽提前生产、双 AIV ACK 汇合和 leaf
排空本来就是该 helper 的既有协议，本次没有更改事件计数/模式。硬件事件可靠性及
编译器生成的 load/ACK 次序仍须由现有协议证据与本候选设备验证共同支撑；仅凭本次
源码审阅不独立宣布机器码或真实 FFTS 可见性已验证。

## 3. 为什么较晚的 Score AIV 不形成等待环

等待链按数据而非阶段编号分析：

1. AIC 等 IDX_NORM 前已发布 Indexer 投影、query 投影、head 权重投影及 QB。
   AIV 的 IDX_POOL→全请求读完→IDX_COMMIT→IDX_NORM 只需要这些此前输入和 META，
   不依赖主压缩、Score 或其后的阶段。
2. AIC 发布 compact Hadamard 后，不等量化/scale 才运行主压缩投影；所有 AIC 都先
   发布自己的 MAIN_PROJECT，才可能进入 Score 或等待 Score ACK。
3. AIV 发布 IDX_SCALE 后，先取得 QB，完成主 QUERY；再取得 MAIN_PROJECT，完成
   主 pool→全请求读完→commit/output，最后到达 Score。它没有等待 SCORE_CUBE 完成。
4. 即便某个 AIC 已因第三份 Score 等 ACK，其他 AIV 等待的 QB/MAIN_PROJECT 也都在
   该 AIC 的 Score 之前。AIV 的主 pool 互等仅等待各请求读完，不依赖 Score 消费。

`runtime_abi.py:9` 明确 phase 是独立的“阶段×核”槽，不是数值递增的全局进度。
将编号 30/32 的发布移到编号 22/11 前，不改变信号判定。主会话报告的顶层语句
多重集合与 144 例 phase 模型是补充证据，本审阅未重复运行；上述第 2 节另补其未
模拟的 FFTS 读写顺序检查。

## 4. state、cache、别名和所有权

- `vector_schedule.py:616/617` 的 Indexer pool 全读完后才 commit，以及 `780/781`
  的主 pool 全读完后才 commit，均未改变。共享 prefix 的只读历史保护仍各自在自己的
  state 平面内成立。Indexer compact 量化/scale 只写 idx_cache，不写主 state/cache。
- scale 提交继续使用 `indexer_scale_plan_stages.py:89` 的 `page%48` 唯一 writer，
  `102` 起沿同页链顺序执行 64B RMW；在此前等全部 key/scale_values 量化 owner。
  同页不同 slot 没有因为前移拆给不同 writer。重复写同一 slot 仍属于原输入合同错误。
- `entry.py:36` 明确 idx_key0/idx_key64 是 idx_cache 的受控只读别名；`46` 明确其余
  缓冲不得互相覆盖。`binding.py:76/89` 分别绑定主/Indexer Native state；默认工作区
  在 `228` 起分别分配，indexer_transfer、score_arena、两路投影及压缩输出没有新增
  生命周期复用。group_offsets 的组描述符与 scale 链表分区由原容量公式隔开，并由
  META 一次发布，之后均只读。
- **限制：绑定层不逐一检测所有外部 Native Tensor/自定义 allocator 的存储交叠。**
  因此不能据本审阅声称任意 main/idx state 重叠、不同工作区重叠也安全。这里使用
  已声明的互不覆盖合同；同一平面的合法共享 prefix、以及明确列出的只读视图是另一回事。

## 5. 仍待当前候选结果回答的事

本审阅只确认源级排序与原同步合同未发现新增阻塞。主会话正在进行的 CPU 后端编译
须确认重新编排后的内存规划和实际生成可用；随后完整设备精度、图重放、guard、
代际及完整 L1 对照才能判断候选是否保留。尤其前移 IDX_NORM/Hadamard 等待可能
推迟主 QUERY/压缩，不保证性能改善；本文件没有把两槽提前启动折算成收益。

## 6. 补充：现成 CPU 产物与旧长档的 Left/Right 槽核对

主会话随后完成本候选 CPU lowering；本审阅只读其
`cpu_v1/result.json`，确认 `full_codegen_passed=true`、`single_mix=true`、
58 个符号轴、130 根参数，`device_execution=false`。没有重新编译。

比较的现成 C++ 文件：

- 候选：本证据目录 `cpu_v1/kernel/kernels/aic/cube_schedule.cpp`。
- 旧长档：主仓 `tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/current_clock_h131072_b16_v1/build_output/decode_csa_spmd_jqlk5eey/kernels/aic/cube_schedule.cpp`。
- 交叉定位：候选 `cpu_v1/kernel/passes_dump/38_after_AllocateMemoryAddr.py`；
  例如 Score 的 `buf_query_left` 在 2281 行，`buf_coefficient_pair` 在 2284 行，
  两个预取 Right 地址在 2487、2605 行。旧机器码没有在本次重新反汇编。

核对方法为先限定生成 C++ 中的 `static __aicore__ void cube_schedule(...)`
函数范围，解析该函数自己的整数常量、Tile 类型与 TASSIGN 地址，再按数学阶段
和 Tile 维度对应。**不能跨整个 C++ 合并 `vN` 常量表**：同一文件还含
`vector_schedule`，局部变量编号重新使用，会把另一个函数的 UB 地址错配给 L0。
这里表中均为各独立 Left/Right 地址空间的字节偏移，不是 GM 地址。

| 阶段、物理 Tile | 旧长档 | 候选 | C++ 首个代表位置：旧 / 新 |
| --- | --- | --- | --- |
| HC FP32 Left[16,256] | 0、16384 | 0、16384 | 6114、6153 / 6031、6070 |
| HC FP32 Right[256,32] | 0 | 0 | 6124 / 6041 |
| Indexer 压缩投影 BF16 Left[16,512] | 0、16384 | 0、16384 | 8060、8127 / 7977、8044 |
| Indexer 压缩投影 BF16 Right[512,32] | 0 | 0 | 8070 / 7987 |
| 主 QB INT8 Left[128,256]、Right[256,256] | 均为 0 | 均为 0 | 10594、10605 / 10511、10522 |
| Compact Hadamard BF16 Left[16,128]、Right[128,64] | 均为 0 | 均为 0 | 11824、11834 / 11029、11039 |
| 主压缩投影 BF16 Left[64,256]、Right[256,64] | 两侧均为 0、32768 | 两侧均为 0、32768 | 11220、11230、11241、11251 / 11388、11398、11409、11419 |
| Score query INT8 Left[384,128] | 0，49152B | 0，49152B | 12470 / 12387 |
| Score coefficients FP16 Left[16,384] | 49152，12288B | 49152，12288B | 12479 / 12396 |
| Score INT8 key Right[128,64] 循环预取槽 | 49152、57344，各8192B | 49152、57344，各8192B | 12947、13204 / 12859、13116 |
| Score 归约 FP16 Right[384,64] | 0，49152B | 0，49152B | 13230 / 13142 |

Score 初始 key 的 Right 地址 0 在两版也相同；这是首块 key 与后继归约 operand
之间原有的生命周期复用，不能把它算作第三个永久预取槽。两版 key Right 地址为
0 的 TASSIGN 文本均 4 处、49152 均 24 处、57344 均 21 处；归约 Right[384,64]
均 16 处且地址 0。这里只比较生成的槽分配，不把文本次数当作运行时迭代次数。

**限定结论：上述 HC、两路压缩/compact、主 QB、Score 未见 allocator 降槽或地址
变化。** 不是整文件分配等价证明：候选这次通用 CPU 配置为 OA_COLS=256、
OB_ROWS=128；旧长档 `result.json` 的 `policies.spmd_reference` 为 OA_COLS=128、
OB_ROWS=96，输出投影物理 Tile 和文本数量本就不同。因此不能将两文件所有 Tile
的多重集合直接作为本次编排变更的单因素比较，也没有据此推断性能。
