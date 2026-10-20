# 动态完整 SPMD 与当前 TND 精度版结构独立审查

日期：2026-10-20。只读审查；不修改正式数学、相邻仓库、SDK 或环境，不提交硬件任务。

## 1. 结论与比较对象

当前 fork 为 `48c58ce7c467dfce63d5cb5c41afd225488d6012`，Nal 参考为 `beaefa6f6788a9448d49293238f6818c246f24f5`。

**两边结构确实不同，但不能仅从源码认定某段同步或算子就是主要时间瓶颈，更不能承诺移植后达到截图数值。**

- Nal 当前是一个24组常驻 MIX，从 HC-pre 一直执行到 HC-post；阶段间以生产者完成信号和 owner 依赖衔接。
- 我们当前完整根先执行 HC/Norm/WKV/Indexer权重的一个 MIX 前缀，再运行原精度后缀的 QKV、压缩、Indexer、Sparse、O-proj、HC-post 等任务。父前缀实际MIX并集32描述符；已验收prepack减少至31。不是“有一个SPMD前缀”等于“完整CSA只有一个MIX”。
- Nal 最新源码已经使用 Native Q/L/positions/slots、动态请求边界，**不能再笼统称为只能固定S6的BSH实现**。截图仍是均匀S6测试点，源码支持动态不等于所有混合query/连续轨迹都已设备验收。
- 当前 PyPTO3e87a843 缺少 Nal 用的 signal_notify/wait；当前 Simplera54c0509 的实际MIX32限制已有失败/修复证据。直接 import 原完整入口不能原样复现。
- Nal 当前性能路径仍存在与我们 Native 精度边界不同的数学。应借鉴 owner、布局和流水，不直接替换数学体。

## 2. 本轮源码与证据来源

Nal 已由主线程受控归档至本报告旁 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`；主要读取 entry、runtime_abi、phase_sync、signals、cube_schedule、vector_schedule、attention_dependencies/stream、hc_stages、output_projection_a_stages、policy。没有拉取Tensor dump。

本次开始时 `git show --stat FETCH_HEAD` 在partial clone触发缺失父对象的自动lazy fetch并DNS失败，未显式fetch/clone。随后所有Git读取加 `GIT_NO_LAZY_FETCH=1`，并改读主线程已导出的本地源码。个别缺失Git对象没有继续联网补齐。

我们读取当前checkout的 `deepseek_v4_flash_csa_tnd_precision_spmd/` 及prepack包、full_precision根和相关已验收报告。部分BF16/overlap候选同时存在，实际测试须以 namespace、源码SHA和receipt明确选定，不能把一个候选的结果挪给另一个。

`CAPABILITIES.json` 是主线程在227完成的无设备API检查；本独立审查核对其记录与源码需求，没有重新初始化设备。它证明接口可用性，不证明新Nal根已编译/运行。

## 3. 截图的性能口径

截图数据原样记录在 `USER_BASELINE.json`，尚未独立复现。

| 档位 | Native完整外层跨度 μs | 多任务外层 μs | 多任务首末incore跨度 μs | 动态单SPMD μs |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 791.25 | 626.50 | 492.78 | 476.40 |
| 128K/B24 | 1342.25 | 1197.75 | 1119.08 | 1039.82 |
| 8K/B16 | 803.25 | 705.75 | 691.42 | 644.08 |
| 8K/B32 | 1107.00 | 1038.88 | 986.30 | 933.58 |

必须区分：

- 截图是median；之前我们报告中有mean，不能混作同一统计量。
- 截图“提升”为baseline/dynamic−1，与耗时下降1−dynamic/baseline不同。
- 多任务outer和first-to-last-incore的间隔不是经证明的纯调度成本；不同档位差值也不固定，例如128K/B4约133.72μs，而8K/B16约14.33μs。不能将差值全部归因AICPU或直接外推到我们的根。
- 旧本项目单层经常为SK0/确定性1；截图Native含SuperKernel/static/ACLGraph/DSA多流。先在同输入同环境建立SK1 Native与我们完整根的外层device-stream计时，不能与历史SK0直接算加速比。
- 已读Nal诊断runner设置 `set_deterministic_level(0)`，并在L1窗口启用dep_gen；其参考精度首先对原PTO多任务路径，不等于本项目确定性Native逐bit基线。新诊断应与原截图口径分别记录，不隐去差异。

## 4. 调度架构对照

| 方面 | 当前精度SPMD/预打包根 | Nal完整动态SPMD |
| --- | --- | --- |
| 根/核入口 | Torch根64Tensor；前缀MIX32，prepack31；其余为任务后缀 | 源码根130Tensor+20constexpr；内部core132Tensor+20constexpr，AST计数，不冒充已生成运行ABI |
| 常驻参与者 | 前缀24AIC+48AIV，全体参加各次handoff | 完整24AIC+48AIV，同一个MIX内推进 |
| 前段衔接 | AIC/AIV双方各10次HARD MIX handoff | 每核每代只发布自己的阶段，消费者等真实owner |
| 工作发布 | 阶段级bar_all/cacheinvalid/fence/syncall | GM独占槽signal_notify/wait，业务payload仍fence/cache维护 |
| 元数据 | 前缀之后另CORE_GROUP构造offsets/token_request、另RoPE任务 | 一个AIV做META，与其他角色的HC/QKV链重叠 |
| 主/Indexer投影 | 前缀先WKV/Indexer权重，后缀再Q及压缩等任务 | cube/vector schedule按独立链调整次序，Hadamard与Q_B可分核重叠 |
| Score→TopK→Attention | 外部task依赖与固定精度helper | 按实际group/root/plan owner获取阶段，query归t%24，不把其他query全部当依赖 |
| Attention | Native数值K512/CAST_ROUND，SWA优化只减无效Cube片；真实TND | K128×5块、三槽流水，局部softmax+rint，跨query仅最后drain |
| 输出链 | 原精度WO-A/量化/WO-B/反量化，最后HC-post另任务 | 按输出组发布OA/quant，再OB/HCpost，同MIX内连续 |
| 工作区 | 原根内部scope释放scratch；别名/stream生命周期已有验收限制 | 大量预分配动态workspace，signal代际、每槽单写者、单流独占 |

### 我们前缀的十次同步在等待什么

按父 `hc_prefix_indexer.py`：

1. AIV widen后全体相会；AIC开始square。
2. square结束；AIV取得平方和。
3. AIV RMS与input/weight打包完成；AIC开始HC linear。（prepack只删除固定weight打包。）
4. HC linear结束；AIV开始split归约。
5. split归约结束；AIV门值/Sinkhorn读取归约结果。
6. 门值/comb结束；AIV开始mix。
7. mix结束；AIV按全行做Native精确Norm。
8. Norm结束；AIC做WKV与Indexer权重投影。
9. 两投影完成；AIV做原BF16/FP16舍入。
10. 输出完成并结束前缀。

这是从源码确认的阶段边界。第5/6/7次不是因为AIC有工作才存在，而是当前AIV owner/布局的交接；**不能仅因Cube在等待就直接删barrier**。末尾handoff还可能承担MIX完成可见性，不能凭两角色函数return假定runtime已收齐全部AIV写回。

### Nal阶段协议也有边界

signal为 `(phase,core,128个INT32隔离跨度)`，60phase、72参与者约2.11MiB。每核一次begin推进代际，同一流顺序复用；失败或代际溢出需捕获外重建。所有consumer必须在业务数据发布后获取，不能把普通read/write替换为通知。

`wait_row_owners` 等待的是**相关owner整个阶段完成**，并非逐tile进度。一个owner有多波工作时，第一tile消费者仍可能等到该owner整阶段结束。因此不能把“按owner减少无关依赖”描述成完全无等待的逐行流水。

Attention另外保留FFTS三槽READY/ACK，不用phase代替逐槽生命周期。输出组通知与workspace不能在消费者读完前提前覆盖。完整SPMD减少外部任务，但新增了本核标量轮询、缓存维护、信号容量和独占生命周期，不是免费的调度替换。

## 5. 工具链依赖：什么阻止直接复现

1. 当前 `CAPABILITIES.json` 实测 signal_notify=false、signal_wait=false；bar_all/syncall/cacheinvalid/fence存在。缺少前两项时不能编译Nal `signals.py`。普通pl.read/write轮询不能当作等价替代，A3 Scalar DCache一致性与发布语义会出问题。
2. 当前实际MIX限制32，Nal core至少132个Tensor参数。根130Tensor还超出本项目当前Torch dispatcher64上限。program-mode的包装与kernel-mode Torch接入不同；program测得快不能直接称已接入当前服务根。
3. Nal参考shell明确绑定独立PyPTO worktree、`simpler-csa-core-args-256-20261016` 和独立PTOAS `score_key_l1_region_sync_20261019/toolchain_v3`，使用bootstrap_isolated校验Python/扩展来源。不同阶段文档中的pypto-csa-repro及当前shell的pypto-csa-memory-reuse并非可任意互换名字，需实际源码/二进制SHA锁定。
4. `wait_codegen.py` 的可选100tick退避通过生成C++重写安装：等待前DCCI/volatile读、未就绪再等待syscnt、定期PIPE_ALL。仅import文件不会启用；必须记录真实CCEC输入与替换数量。不能把当前SDK默认等待或普通syncall的性能视为同一策略。
5. Nal policy还依赖特定PTOAS精确L1范围及死状态修复的生成行为。若没有相同工具链，至少要先验证实际L1/UB分配、右矩阵切片offset、死分支消除和事件顺序；不能只看源码有if就认定开销已消失。
6. SDMA-enabled wrapper和普通wrapper是不同环境因素，不能将enable_sdma开启和真正预取收益合在一起。

**结论：在保持当前SDK不变的约束下，原样复现Nal完整单MIX有明确能力阻断。** 这不妨碍对其已导出源码做审查，也不妨碍在现有SDK内做更小MIX/任务融合。扩容ABI/信号接口不是本轮可擅自实施的“小优化”，本review未建议修改或安装框架。

## 6. 不能直接借用的数学

| 部分 | Nal当前代码 | 我们必须保留的Native精度边界 |
| --- | --- | --- |
| HC投影 | 原FP32输入/权重K256 matmul、四split | 现有已验收HF32分解/交织BF16 Cube、Native K顺序及split归约；不得以普通FP32替代 |
| HC mix | `(y0+y1)+(y2+y3)` | 已验收left-fold `((y0+y1)+y2)+y3`及BF16落点 |
| input Norm | mix后16个K256平方和顺序链 | 当前4096列fold树、INT64 sqrt修正、Native scalar division和BF16权重次序 |
| attention | K128局部softmax，概率rint，最终RoPE取未cast FP32 | Native K512/max/归约与概率round；归一化先BF16再widen做inverse RoPE |
| WO-A | 规则K256起点顺序 | 当前按Native标定的cyclic K16起点、各packet及fallback；不能抹掉start表 |
| O-proj量化 | 八个1024列组各自amax/scale | 拼接8192列后的整行动态量化及原RNE边界；不能因组流水改回per-group量化 |

Nal对自身canonical多任务版本的精确回归，不能证明这些差异与我们的Native逐bit基线一致。只移调度也要检验生成器是否改变数值tile/归约，不把AST中算式相同当硬件位相同的充分证明。

## 7. 现有SDK内可验证的单因素候选

以下是可落地审查的方向，不是已经测出收益的结论。先固定所选精度variant，与优化Native同场建立完整边界基线，再选择一项。

### A. 同一完整行owner融合HC mix与精确Norm

当前mix按M8和D分片分工，Norm改为每行一AIV，所以二者之间需要全体handoff及GM中转。可试将同一完整行的mix交给其Norm owner，保持left-fold混合、BF16舍入，然后使用**原4096 fold树/INT64 sqrt/除法/权重乘顺序**。先只改变owner与两阶段编排，仍保留必要BF16中间存储或等价位值，再判断能否移除这一条handoff。

收益来源可能是更少全体相会/GM交接；代价是失去mix列并行、UB压力及短T参与度下降。不能复制Nal的16次K256 RMS来换速度。需T1/T21/T96/T144/T240、poison/±0/极值、当前Native同输入首差停止，实际UB与尾行审查。

### B. 最终O-proj反量化与HC-post做小型AIV融合

两者都在输出尾部，可在不新增完整resident signal的情况下，用同一AIV行owner串接。必须显式保留 `attention_out` 的BF16 rint边界，然后按原HC-post乘加顺序消费；不能直接用FP32反量化结果跳过BF16。保持WO-A/B和整行量化不变。先审Tensor描述符<=32、完整行独占、输出4路和输入hidden/post/comb寿命，再看是否只减少任务/GM往返而没有改变数学。

### C. Attention plan提前解析物理cache行

Nal把本轮已验证TopK逻辑行转为物理cache行，再由gather消费，减少逐行重复除32/查页。我们已经有token_request，不存在完全相同的反复Q/L扫描，不能把Nal的query-metadata优化再算一次收益。

可仅在当前内部plan写入物理行、gather按物理行读，保持对外TopK、bias、valid mask和全部softmax/PV不变；越界/负槽/物理容量检查与plan-gather表示必须同步切换。先用现有gather_row，不假设新MGATHER接口可用。它只影响寻址，数学保持，但真实页布局/短query/dummy/ABA必须逐bit门禁，是否节约Scalar/MTE成本待测。

### D. 按边界逐段扩SPMD，而非一次移入Nal全部根

例如先融合某个参数并集<=32的后缀AIC/AIV阶段，保持已验收数学，替换一段runtime任务往返。每次先证明union ABI、MIX常驻资源、所有producer/consumer和工作区生命周期。只对自有同dtype内部scratch打包；不能绕过alias检查、把Native cache复制当成免费合参，或让O-proj提前采用per-group量化。

### 不优先做的操作

- 直接删HARD handoff或换普通GM自旋：没有可见性与完成语义证明。
- 把另一个metadata任务“移到prefix前”便宣称可重叠：prefix全24AIC+48AIV常驻，外部CORE_GROUP任务未必能共驻；强行交错还可能阻塞syncall参与者。Nal能重叠META，是把角色放在同一MIX内部。
- 只改变pl.spmd数量或把全部源码装进一个大inline：参数上限、scratch生命周期、物理tile、signal接口仍存在。
- 使用S6 padding换资源规整，或用当前batch/截图行号构建白名单：不符合真正TND与动态范围要求。
- 在没有同一工具链与数值门禁时，引入Nal退避C++回调/SDMA开关或N128、K128数学作为已证明提速。

## 8. 对新测量runner的独立意见

新 `tools/probe_csa.py` 从冻结harness派生，Native SK1、实际impl DSA标志为true、确定性1；候选仅选frozen overlap，43源及其source/compile receipt检查保留。FULL实际graph、8边界A/B/A、poison、guard、拒绝fallback及64Tensor根门禁仍在计时前执行；跨bucket旧块在本诊断中明确不重跑，不冒称扩展覆盖。

12×100 raw replay无DFX；之后另用CANN采三次replay。Native源码在辅助流阶段尾部通过main_stream.wait_stream(aux_stream)衔接后续消费者/输出，因此主流事件可以覆盖这些依赖；但**DSA配置为true不等于已经证明实际graph确有多流执行或无串行化**。必须读本次Native trace的stream/事件及全外层跨度，不能以flag一项就声明截图条件完全复现。

建议trace后也再次执行guard与只读packet检查；目前8状态drift检查已有，但保护区/只读范围应独立保留。脚本位于报告根的run-measurement.sh，750秒timeout，任务仍须max780。helper4项hash、AST及shell语法已只读检查，无设备操作。

完整动态Nal因接口阻断未加入时，准确命名为“优化Native vs 当前完整精度TND”两臂基线，不能称三方复现已完成。

## 9. 验收顺序

1. 固定当前63/71等实际variant运行源集合及工具链，不能把旧43数字自动当全部新候选数学源；每次以receipt实际集合为准。
2. 同输入、相同完整HC-pre→HC-post外层范围、SK1 Native/确定性1、无DFX计时；截图的median和speedup定义分别记录。
3. 以CANN trace核实多流和采集边界。没有内部观测就不把整段span细分为纯计算、搬运、等待。
4. 选一个结构因子做CPU/ABI/GCC/CCEC及实际生成物审查，再单卡Native逐bit首差停止。所有任务<=780秒，不安装框架、不覆盖精度基准。
5. 有稳定完整CSA收益后才复测8K/B40、128K/B24真实模型192token/接受事件，不把单层比例直接乘成整模型收益。

本轮没有性能预测值或已测收益承诺；截图仍为用户给定目标，不是当前环境已复现结果。

## 10. 主线程首个SK1诊断的实际阻断更新

主线程报告：首次新诊断在Native graph编译时被旧官方私有SK0 static-cache桥拒绝，合同要求metadata.super_kernel=false且super_kernel_optimize=false；因此没有本轮数值或计时结果。不能删除assert、将SK0包标成SK1，或把失败解释成PTO精度问题。

建议沿用其分进程方案：

1. 现有已资格SK0环境保存同fixture Native/PTO的八边界SHA与输入身份；不改PTO冻结资格。
2. 独立Native-only进程，在Python启动前移除旧whole/sitecustomize路径及LOW_ACCEPTANCE_OFFICIAL_CACHE/STATIC_MANIFEST，使用独立cache/OPP目录，走真正官方SK1编译和reselect。
3. 明确记录新metadata的super_kernel标志和包SHA，不套旧13包SHA。
4. 跨进程先核hidden/positions、初始cache、query bounds/slot/页表和权重SHA，再比较八输出边界；确定性、HCCL/NZ2保持相同。
5. 若Native SK1对SK0本身出现bit首差，先定位Native配置变化，不能先放宽候选精度或归罪PTO。trace仍需核真实多流。

这不会修改SDK或共享安装。缺失signal接口和132描述符的工具链兼容问题仍待用户授权后的独立决定；本review不越权扩展框架。
