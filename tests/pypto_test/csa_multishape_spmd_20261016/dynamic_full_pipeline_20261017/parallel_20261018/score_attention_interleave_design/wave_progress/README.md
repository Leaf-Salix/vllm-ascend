# 交错候选的最小wave进度表示：只读设计

2026-10-18，基于formal668。**保留现有signal根Tensor，固定阶段0..59后追加每wave三个阶段即可；不需要新增根入参或把wave编码进代际。**本轮仅阅读并写本报告，没有修改源码、依赖、runner，没有模型、编译、NPU或hash。边界helper虽由另一工作线准备，本文不称其已接入完整算子。

路径缩写：`P`为主仓`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd`；`D`为主仓`tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017`。

## 布局与容量

保持`PHASE_COUNT=60`表示原固定前缀，新增概念常量`WAVE_PHASE_BASE=60`、`WAVE_PHASE_STRIDE=3`。wave w的ROOT/TOPK/PLAN偏移分别为0/1/2：

```text
phase(w, kind) = 60 + 3*w + kind
offset(w, kind, physical_core) = (phase(w,kind)*72 + physical_core)*128
signal[offset] = generation
```

每个槽仍是128个INT32/512B，只有物理核自己的word0可写。AIC编号0..23、AIV编号24..71不变；本设计新增阶段只由AIV写，AIC部分照样保留隔离空间。word1..127继续为padding，不借phase0或padding存wave边界、计数或第二种epoch。`begin_invocation`仍由全部72核每次各执行一次。

与当前边界工作线约定一致：除最后可能不足8组的尾wave外，每wave至少包含8个真实group；从`first_group+8`开始找下一group的`first_query%8==0`切点，找不到则延到末尾。首query_begin=0，末query_end=T，包含dummy/gap，保持global item/query owner。合法groups≤T，故

```text
wave_capacity = ceil(T/8)
signal_phase_capacity = 60 + 3*wave_capacity
signal_words = signal_phase_capacity * 72 * 128
```

Host只用现有hidden.shape[0]的T容量，不读Q/L/positions/group_plan。实际wave_count和所有group/query边界仍从设备描述符计算；groups==0走原不交错路径。容量不是实际wave数：例如T96可预留12wave，但不能据此让设备跑12wave。每新增wave容量为108KiB，T96总signal为884736个INT32/3456KiB，原60phase为552960个INT32/2160KiB；这是GM容量成本，不是性能结论。

现有`runtime_abi.py`的WORDS已是动态轴，`entry.py:264,775`的signal仍为一维INT32 InOut；只扩大该Tensor的shape，无需新增130根入参之外的参数，也不需要新增动态轴。这里只是接口结构判断，没有新的编译ABI PASS。

## 三种完成点与真实owner

保留现有数学分工；下表的ROOT公式限拟定long G6、BALANCED、ROOTS=1分支，其他策略继续原路径。

| 阶段 | 谁发布一次 | 消费者准确等待谁 |
|---|---|---|
| ROOT(w) | 本AIV完成该wave所有由自己写出的leaf根后，原`publish_phase`发布。没有根store的lane无需发布。 | query q的merge只枚举本query真正读取的leaf。设其device group为j、group首query为first、lane=`(q-first)//3`，全局item按原groups<24时`leaf*groups+j`、否则`j*leaves+leaf`；物理根写者=`24+2*(item%24)+lane`。 |
| TOPK(w) | AIV worker=q%48；完成本wave所有归自己的query行后发布，**包括dummy/gap写-1/-inf**。 | 原M8 plan获取该8行内的TopK owner，即`24+q%48`；最后尾tile只等实际T行。 |
| PLAN(w) | AIV worker=`floor(q/8)%48`；完成本wave所有归自己的M8 plan tile及indices/bias/mask写回后发布。 | AIC及双AIV只获取自己本wave query对应的`24+floor(q/8)%48`；两个lane可等同一个非消费型槽。 |

源证据：`indexer_group_stages.py:235–250`、`indexer_score_stages.py:592–602`、`indexer_publish_stages.py:137–175`、`attention_plan_stages.py:155–159`。真实leaf数必须沿`indexer_visible_root_count`和现有visibility合同计算，不能直接用max_leaves等待未生产的根。root_count=0的query不读取pair，不等待ROOT，但仍写本轮空TopK并参加后续plan。

这是每核每wave的聚合完成槽，**不能在每个leaf/query完成时反复写同一generation**。本核该wave全部ROOT发布之前，不得先阻塞于同wave merge所需的其他ROOT；全部TOPK发布之前，不得先进入plan等待。否则一个已提前写过的槽会虚假代表后面的未完成行，或把发布者卡在后继依赖中。发布ROOT的位置必须晚于最后root store及原release，Score MTE2 ACK只能证明FIFO已读，不能代替根ready。root/TopK/plan的业务GM行仍是全局q地址，不通过wave槽确认其可以覆盖；双槽/三槽FFTS及drain保持独立。

## 发布、获取与跨代

`phase_sync.py:84–117`已接受动态phase：`publish_phase`保留业务fence/cacheinvalid/fence以及notify的PIPE_ALL和信号维护；`wait_producer`保留signal wait后业务fence/cacheinvalid/fence。三条链是`root store → ROOT release/acquire → merge store → TOPK release/acquire → plan store → PLAN release/acquire → Attention读取`。无需修改底层signals/wait_codegen。最小实现可以直接复用`wait_producer`，重复owner等待虽正确但有固定成本；ROOT的非连续owner不能误套`wait_row_owners`。M8 plan对连续TopK行可沿现成`wait_row_owners`，因为其owner公式完全相同；Attention跨24行的query列表则显式枚举plan owner。

减少等待集合只保留这条业务链，不自动继承旧全阶段门槛的其它用途。必须继续保留META、Score输入QUERY_QUANT/COEFF/全48 SCALE、Attention QUERY/ORIGINAL_CACHE/MAIN_OUTPUT/ROPE等现有获取；每叶ACK和Attention尾部排空也不能交给wave phase隐式保证。原0..59的槽身份保持，既有全阶段完成通知若保留仍只能每代发布一次，不能拿35/36/37反复充当wave槽。这里不新增全72核barrier。

同一workspace布局固定、单流/图顺序独占且各核代际同步时，上一调用留下的wave槽值小于本代generation，GE等待不会把旧值误认成本轮完成。本轮没有工作或未使用的wave/owner保持旧值，**不逐次清零**；跨调用有效group/leaf集合变化时消费者必须重新按本轮metadata选owner。布局或T容量改变须重新绑定清零workspace，失败/INT32代际上限仍按原合同处理。不能把wave与generation相乘作为新标记，避免引入额外回绕协议。

## 最少修改位置与验收检查

以下是未来实现的修改清单，本轮未执行：

| 位置 | 必要变化或保持项 |
|---|---|
| `P/runtime_abi.py:120–126`、`binding.py:224–248` | 定义固定前缀与wave偏移；binding按上述T上界扩大signal，继续独立512B对齐、首次zero和allocator保护区。可统一为候选预留，旧reference仍只分配60phase。 |
| `P/cube_schedule.py:379–383`、`vector_schedule.py:851–893`及待接入的波次编排 | 接入三种wave发布/获取，保留原输入与最终输出依赖；实际数学step和边界helper的接线另审。`phase_sync.py`/`signals.py`无需为新phase改可执行逻辑；`entry.py:52`及模块概览补容量合同即可。 |
| `D/run_device.py:343,451,500` | 固定`view(PHASE_COUNT,72,128)`改为按该candidate实际numel解析phase容量；先检查整除72×128、满足本包容量合同，再检查所有72个phase0和**全部新旧phase**的padding。所有observed_phase_slots完整保存。 |
| `D/output_policy_matrix.py:92,151,173,249` | 每个candidate/reference各用自己的signal实际容量，不能借正式包的60做reshape。现有phase_count相等断言只可继续核对固定前缀一致，不能表示总容量相等；总words不同不等于根参数ABI不同。保留各自generation/call数、全部padding和业务scratch保护检查。 |
| `D/check_output_phase_signals.py:23` | `len(slots)==60`改为验证固定前缀存在与完整行宽72；原40/41及44..59的精确owner检查原样保留，追加槽不能导致整份检查被跳过。 |
| `D/metadata_replay.py:159,169`、`padding_replay.py:183,193`、`native_query_probe.py:217,225` | 相同固定60 reshape改为本对象实际容量。已有metadata变化、旧signal保留、72核代际、全padding和边界guard强度不下降。 |

`source_loader.py:load_binding`当前返回的是包级PHASE_COUNT，不能把它当实例总容量。最小路径不必改loader：保留其固定前缀信息，runner从各实例signal.shape读取容量并保存schema。`check_cpu.py`现有根Tensor/动态轴审查无需因扩容而改变合同；`source_interface.json`的phase_count=60只能标记旧前缀，若作为报告schema使用需另注明扩展公式。当前无时钟插桩要求；以后若开启phase_clock，新增阶段须补名称映射，不能让`compile_diagnostics.py:104`把未知phase默认记为BEGIN。

已有通用保护只检查phase0全72代际及padding，不要求每个业务phase的全部72槽到当前代。因此追加槽本身不要求放宽门槛。未来还需读回核对wave真实发布owner、未使用槽保持旧值，以及相邻调用有效集合变化；这属于候选验收，本文没有假称已完成。最小进度表示是每wave三类单写者release，不是新的调度模型，也不据通知数量估算收益。
