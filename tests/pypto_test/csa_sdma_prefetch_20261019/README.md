# CSA 单 SPMD：kernel 模式 SDMA 迁移与 L2 预取优化

## 本轮目标与边界

用户于 2026-10-19 授权先迁移 Simpler PR #2583、PyPTO PR #3001，
随后将 AIV 发起的 SDMA L2 预取作为主要优化方向。未经用户再次明确允许，
所有仓库均不得 push。本目录记录迁移、构建、验证和后续候选结果；未完成的
检查不得预记为通过。

- 先处理 128K/B16、B28，最终覆盖 128K/B4、B8、B16、B24、B28；
  目标是相对同轮原多 task PTO 的完整 L1 incore 首尾耗时降低至少 30%。
- 8K/B16、B32 兼顾，避免明显回退。正式权重始终使用
  `/data/model/DeepSeek-V4-Flash-0731-w8a8`。
- 最终保持完整单 SPMD，只允许区间策略，不增加单点特化。
- 先获得关键路径证据，再约定候选收益门槛；每个候选说明作用区间、
  潜在收益依据和停止条件。数值非逐 bit 的数学改动采用前另行确认。
- 允许短程独立 metadata 诊断对照，最终分离需另行讨论。
- 允许一次有实质差异的旧方案系统重评。模拟只辅助解释局部流水，
  不把受计时退避影响的模拟等待当真机性能。
- 非算子编写问题按既有要求提交对应主仓 issue，并附建议修复；不提修改 PR。
  Simpler 非 bug 改进继续留在本地。

## 迁移来源

| 项目 | 来源 | 本轮接入前基线 |
| --- | --- | --- |
| Simpler | [PR #2583](https://github.com/hw-native-sys/simpler/pull/2583)，`0b34c256ff9c027f1afaf83c9c71eadda8ec2daf` | `57e499238fd4ee3548947ad860be456aaeeb16b5`，保留 256 Tensor + 16 scalar 等改动 |
| PyPTO | [PR #3001](https://github.com/hw-native-sys/pypto/pull/3001)，`ed612fbdb7e321013fc4324bd2a6febd86df2234` | `2dfa7f120ba5e577566046d920845aa241bfa7bb`，包含已发布的 CSA 编译修复 |
| PTOAS | 本轮保持既有已验证的配套工具链 | `score_key_l1_region_sync_20261019/toolchain_v3`；对应复现源码 `e5c1c432af8249bac2f905c7648ebafefec3c20a` |

新建独立 `simpler-csa-sdma-20261019`、`pypto-csa-sdma-20261019` worktree。
不覆盖共享安装或既有复现分支。PyPTO runtime pin / ABI revision 必须指向
合入 SDMA 且保留参数扩容的新 Simpler 提交，而非回退到 PR 原始基线。
运行时、binding、PyPTO adapter 和生成算子一起重建。

## 已核实的机制

Simpler PR 在 kernel 初始化时 provision / warm SDMA，TMR 与 HBG 均将
workspace 地址写入自己的 Runtime 和核上下文；eager / ACLGraph replay
复用该资源。PyPTO PR 增加 `pypto.torch.init(enable_sdma=True)` 并检查
artifact 的 SDMA 需求。SDMA workspace 不成为业务 Tensor 入参。

`pl.prefetch.async_prefetch` 经 `TPREFETCH_ASYNC` 在 AIV 提交 SDMA CMO，
把连续 GM 区域预热到 L2，仍由 MTE 将计算所需数据搬入 UB / L1。
当前前端要求静态长度的连续逻辑一维区域；可选择动态起始地址的固定块，
但必须证明访问范围合法。提交路径包含 Scalar、缓存发布及管线同步成本，
不假定预取免费，也不假定会固定驻留 L2。

## 状态

2026-10-19：已完成两个 PR 的源码审阅及版本核对；本地迁移、重新构建和
真机验证正在进行。已有 PR 的测试结果不作为本轮 CSA 接入通过的证据。


### 本地迁移及运行时验证

- Simpler 本地提交 `09478f27ed0fd2039bd1fbf4015b8dca5b8ca188`：
  完整迁入 #2583 的 44 个文件改动，参数扩容基线保留。
- PyPTO 本地提交 `56cca2c40`：迁入 #3001 的 SDMA 链路、元数据检查和
  相关测试修复；runtime 与 ABI pin 指向上述合成提交。冲突中的新版 JIT
  结构缓存测试不属于当前基线且与 SDMA 无关，保留当前 CSA 的 JIT 测试。
- 首轮 Simpler runtime 构建发现历史 editable finder 抢先绑定旧源码，
  因此不计为迁移构建通过。新增仅本目录进程启用的 `sitecustomize.py`，
  对父子进程固定源码及 binding 位置；重新构建日志显示基线由
  `ef681289` 切换为 `09478f27`，TMR/HBG host、AICPU、AICore 构建完成。
- TMR 原 PR 五场景：`task_20261019_211547_34556672723`，5 passed / 69.04 s。
- HBG 原 PR 五场景：`task_20261019_211548_34557179457`，5 passed / 74.15 s。
  两组覆盖 eager、首次 capture/replay、recreate、disabled、program isolation，
  每场景使用原 PR 的 100 轮验证。原始 JUnit 位于各 runtime 子目录。
- PyPTO core / adapter 的独立构建完成，配置/注册/产物元数据 180 项 UT 通过；尚未声称完整 CSA 通过。

### 第一项候选的证据与停止条件

拟在同一个 MIX 中利用 AIV 等待 Q_A 完成的区间，预取后续主 Q_B 的连续
INT8 权重。优先选择本阶段工作较少的 AIV，用固定大小连续物理块分摊，
实际请求数和长度继续由 device metadata 决定；不新增 Host 数据取值或任务。

预取目标是缩短 Q_B 的冷数据读取及其后继的暴露等待，收益上限受 Q_B
实际访存占比限制。若 SDMA 提交/等待反而延后 QR，或完整 CSA 首尾没有
稳定收益，则不采用；保留记录后调整预取区间与时机。

已核对既有生成代码：PyPTO IR 有 `l2_bypass` 属性，但选定 PTOAS 的
TLOAD lowering 仅在另有 offset 时处理该属性，现有完整 C++ 使用普通
TLOAD。首轮不改这一历史工具链行为，候选与参照保持相同的消费者读取，
避免把缓存策略变化混入 SDMA 收益。暂不据 Python 属性名称判断实际绕过 L2。


## 2026-10-19 22:21 阶段记录：通道故障定位和公开反馈

### 编译与描述符支持

- 重编后的 GM view 9 项 UT 通过，重用已有 issue #3042 的本地原型。
  `gm_view=True` 只允许连续整行 GM 描述符，不隐式搬入 UB。
- NZ 整体物理展平在跨显式 JIT helper 时丢失 TensorMeta。已提交
  [PyPTO #3062](https://github.com/hw-native-sys/pypto/issues/3062)，
  本地仅允许静态元素数相等的 rank-1 整体展平；任意 NZ reshape 仍拒绝。
  正例/反例 2 项 UT 通过；CPU 前后复现在 `nz_metadata_before.json`、
  `nz_metadata_after.json`。本地提交 `7614f4eb0`，未 push。
- `source_qb_v1` 的 InCore reshape/slice 会默认变成 Tile，导致 32 MiB
  NZ 权重被要求搬入 Vec；v2/v3 改由编排传物理展平描述符后，暴露上述
  JIT 元数据问题。补齐后 `cpu_qb_v3_nzfix`、`cpu_qb_v4_single` 通过，
  保留 58 个动态轴、一个 MIX、sync_start 与 requires_sdma。
- v3 为 16 个 AIV 各预取 2 MiB；发现通道冲突后不送完整 CSA。
  v4 为一个 AIV 预取整个 32 MiB，在原 Q_A 等待处与 SDMA 重叠，
  原有数学和各核工作保留。当前只是冻结候选，正式 ops 源未采用。

### 已确认的 SDMA 多 AIV 通道别名

完整 MIX 的 `split_aiv=True` 版本实际启动 24 AIC + 48 AIV。
PTO-ISA 默认用原生 CCE `get_block_idx()` 选择 SDMA channel group；
真机读回显示同组两条 AIV 的该值相同，48 条角色只使用 24 个通道。
原生 `get_subblockid()` 在这些 user kernel 上均为 0，不能直接拼接它；
Simpler 传入的逻辑 block/subblock 正常区分 48 个角色。

`mix_tmr_multi_events_v1` / `task_20261019_220656_145189114696`：
48 个 event/session 都有效，11 个 Wait 返回 false，对应 11 整行未执行。
AIC 输出正确。每核独占 64-byte 诊断行，写后 DCCI/DSB；额外唯一标记确认
读回来自真实执行的探针。未把这些探针用于任何性能结论。

随后新增 `sdma_mix_public_repro.py`：直接传独立连续 4 KiB Tensor，
不依赖本地 GM subview / NZ 展平能力。一个全卡 MIX 内对照如下：

| 配置 | 任务 | 结果 |
| --- | --- | --- |
| 48 AIV 默认 auto channel，eager | `task_20261019_221100_1678155573` | FAIL；有效事件中有 4 个 Wait=false |
| 48 AIV 显式通道 0..47，eager | `task_20261019_221100_16783504279` | 初次及 10 次变输入重复全部 PASS |
| 仅 worker32 发起、其余 AIV 正常计算，默认通道 | `task_20261019_221100_167890724331` | 初次及 10 次变输入 ACLGraph replay 全部 PASS |

显式通道仅为单个全卡 MIX 的因果诊断，不能作为一般多 task runtime 修复。
正式方案应由运行时暴露物理 AIV 唯一的通道，处理跨 task 并发，不能直接使用
逻辑 task 编号。原 PR 的单发起者原生用例通过不代表该并发场景已覆盖。
本地有其它既有 CSA 修复，未声称这是干净 PR head 回归或必定由 PR 新引入。

用户授权 SDMA bug 直接评论原 PR。本次已提交：

- [PyPTO #3001：初始证据](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269202617)
- [PyPTO #3001：完整公开复现与显式通道对照](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269309892)
- [Simpler #2583：运行时侧反馈](https://github.com/hw-native-sys/simpler/pull/2583#issuecomment-6269212292)
- [Simpler #2583：补充成功对照](https://github.com/hw-native-sys/simpler/pull/2583#issuecomment-6269315353)

### 必须剔除的测试/编译混淆

以下失败均保留，但不能用作 SDMA 通道故障证据：

1. smoke v1 未接收队列追加的 `--device`：CLI 失败，未编译执行。
2. smoke v2 将 INDEX 字面值写到 INT32 Tensor：测试常量类型错误，未执行。
3. smoke v3 / 首次诊断漏 `split_aiv=True`，实际 `MixedKernels{0,1,INVALID}`，
   只写偶数行；这是测试角色配置错误。v5/v6 后已核对 `{0,1,1}`。
4. rawids v1 hook 绑错 `simpler_setup.KernelCompiler`，读回 0 不是硬件证据；
   v2 起绑定真实 `pypto.runtime.kernel_compiler.KernelCompiler`，并校验唯一标记。
5. single v1 / one-lane 的 BOOL phi 触发 `eq` 不接收 BOOL 的编译错误，未执行。
6. single v2/v3 用 `pl.cast(wait_result, INT32)`，被降为 `extsi i1 to i32`，
   true 变成 -1，导致 `==1` 分支不走。单发起者 event/session/Wait 实际有效成功。
   v4 改为显式 if/else 写 0/1 后，初次及 10 次 eager 全 PASS。
   此独立标量编译问题已提交
   [PyPTO #3063](https://github.com/hw-native-sys/pypto/issues/3063)。
   `repro_bool_cast.py` 在 CPU 编译执行真实生成算术，得到 `[0,-1]`；未改编译器。
7. 完整 CSA v1 任务已经进入首个候选调用，但随后旧测试按固定目录查找 C++
   失败：新 wrapper 未恢复原 shell 的 PYPTO_PROG_BUILD_DIR，且 JIT 产物含
   `_jit_` 前缀。未取得精度/性能结论，不能记为 CSA 数值失败。已补目录隔离、
   兼容前缀且保留唯一性与真实等待退避源码检查，v2 重提。

### 完整 CSA 首轮待测

- B16 v1 `task_20261019_221257_175779316708`、B28 v1
  `task_20261019_221258_175812820100`：上述产物路径检查失败。
- B16 v2 `task_20261019_222118_220329328819`、B28 v2
  `task_20261019_222118_220349422365`：已提交，待结果。
- 三方均使用本轮新工具链及同一个 SDMA-enabled context；参照算子无 SDMA。
  保留原完整输出、Top-K、cache/state、保护区、metadata/padding、图重放和
  代际检查，通过后才进行三方轮转的 5 个 L1 窗口。只统计 incore 首尾，
  不把 Native 初始化或 runtime warmup 计入性能。
- 本次没有 push，也没有宣称完整 CSA 性能已有改进。


## 2026-10-19 22:42：完整结果、scratch复现与阶段诊断

### 隐藏 UB scratch 冲突：独立于通道碰撞

`sdma_mix_live_tile_repro.py` 先加载一个 1×1024 FP32 Vec Tile，单 AIV
发起预取，再消费旧 Tile。诊断 C++ 在 context 前插入 PIPE_ALL，排除旧
TLOAD 未完成。PTO-ISA 的 context 内部固定把 256-byte scratch 绑定到 UB 0，
与编译器正常分配的存活 Tile 重叠：worker32 前 8 个 FP32 被覆盖，
event/session/Wait 均为 1。改为诊断性独立 UB 8192 后初次及 10 次 eager 通过。

- 默认 UB0：`task_20261019_222954_261045132677`，FAIL。
- 独立 scratch：`task_20261019_222955_26106641469`，PASS。
- [PyPTO #3001 完整复现与建议](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6269721600)
- [Simpler #2583 同步反馈](https://github.com/hw-native-sys/simpler/pull/2583#issuecomment-6269727287)

正式修复需要 scratch 的分配、异步生命周期和 auto-sync 契约，不得在大算子
中硬编码8192。当前 CSA 候选只在阶段间无跨越存活 Tile 的位置调用，未修改
PTO-ISA；完整数值验证通过不代表公共 API 的任意插入位置安全。

另用与Q_B相同的32 MiB输入完成单发起者初次+10次图重放，真实event/session/
Wait均有效成功：`task_20261019_223409_2781142619`，`mix_qb32_single_graph_v1`。

### 完整 CSA 无阶段插桩结果

五窗轮转，同根输入、同一新工具链、同一个SDMA-enabled context，单位μs。
每项973个PASS，max_abs=0；Top-K、完整cache/state、保护区、metadata/
padding、图重放和信号代际通过。这里只代表B16/B28、S6、128K与当前合成历史。

| 预取策略 | B | 多task L1 | 同轮原单SPMD L1 | 候选单SPMD L1 | 相对原单SPMD |
| --- | ---: | ---: | ---: | ---: | --- |
| 一个AIV、32 MiB | 16 | 872.50 | 809.06 | 849.52 | 耗时增加5.001%，不采用 |
| 一个AIV、32 MiB | 28 | 1278.64 | 1224.66 | 1233.68 | 耗时增加0.737%，不采用 |
| 八个核组各一条AIV、各4 MiB | 16 | 868.00 | 812.12 | 838.00 | 耗时增加3.187%，不采用 |
| 八个核组各一条AIV、各4 MiB | 28 | 1285.86 | 1222.44 | 1230.18 | 耗时增加0.633%，不采用 |

八通道不永久保留计算核；仅尾部8个核组的AIV0发起，同组AIV1照常计算。
这规避本单MIX内默认通道的双AIV别名，不宣称修复任意多task并发场景。
逐窗数据见 `performance_summary.json` 和各 `source_matrix/l1_*`。

### 72核阶段时戳指向前段干扰

`diag_qb_reference_h131072_b16_v1` 与 `diag_qb_eight_h131072_b16_v1`
均功能精确通过。任务 `task_20261019_223517_282471032658`、
`task_20261019_223518_28248865613`；每侧一次带阶段时戳的完整调用，
仅用于定位，不能把探针总时长当正式性能或精确HBM带宽测量。

| 观察项 | 原单SPMD | 八通道预取 |
| --- | ---: | ---: |
| 最后INPUT发布 | 27.24 | 49.80 |
| 最后PARTIAL发布 | 35.42 | 75.06 |
| 最后NORM发布 | 56.64 | 97.32 |
| 最后QA发布 | 76.64 | 117.50 |
| 最后QR发布 | 90.10 | 134.06 |
| QB阶段24角色中位区间（含4个空角色） | 66.73 | 38.42 |
| 最后QB发布 | 241.46 | 260.74 |

完整逐核数据在 `phase_clock/phase_clock.json`；实际20个QB执行者的独立统计
另外记录于 `phase_qb_diagnosis.json`。观察显示QB自身缩短，但前段延后，
收益被抵消。预取与HC争用访存资源是当前解释，未采集硬件带宽/命中计数器，
不把它写成已测得HBM饱和。

源级核对：所选尾部角色对当前目标T没有HC M8/M16计算，原位置就会很早
到达Q_A等待；把预取再往函数入口移动并不能创造足够新窗口。下一候选
`source_qb_v6_after_norm` 只让这8个发起者先等真实NORM写者发布，再开始
预取，尝试避开HC输入与投影的关键区间。保留一个SPMD、全部原数学和
角色；依赖集合按device T计算，不新增单点特化或host metadata取值。

B16任务 `task_20261019_224211_3096903345`、B28任务
`task_20261019_224212_309701418809` 在运行；若仍无完整L1稳定收益，就不将
本项扩展到其余档或正式入口。当前正式算子保持E49采用版，所有仓库未push。


## 2026-10-19 22:50：NORM后预取仍无净收益，结束本组调测

两任务均completed/exit0，继续973项PASS、max_abs=0，全状态/保护区/
metadata/padding/ACLGraph与代际通过。完整无阶段插桩L1五窗：

| 场景 | 多task L1 | 原单SPMD | NORM后8通道候选 | 判断 |
| --- | ---: | ---: | ---: | --- |
| 128K/B16/S6 | 873.36 | 804.66 | 841.62 | 增加36.96μs，不采用 |
| 128K/B28/S6 | 1295.38 | 1204.78 | 1222.40 | 增加17.62μs，不采用 |

本轮三种Q_B预取调度都没有完整算子的净收益，到此结束本组调测，不扩七档，
不修改正式入口。NORM后的对照表明，仅避开HC前段仍不足以实现净收益；
不能把之前的带宽争用解释写成已经解决的唯一根因。下一步应评估更长的
Indexer/Attention等待窗口与真正的访存占比，不再反复微调这三个负例。

已向用户提出后续候选扩档门槛：重点B16/B28至少一档稳定净省20μs，另一档
不明显回退；整体30%目标不变。当前待用户答复，不把建议当作已批准门槛。

本目录保存源代码冻结、公开复现、任务结果、原始L1和阶段时戳。编译缓存、
二进制、设备运行日志仅本地保留，清单记录路径和大小，不计算hash。
所有仓库仍未push。
