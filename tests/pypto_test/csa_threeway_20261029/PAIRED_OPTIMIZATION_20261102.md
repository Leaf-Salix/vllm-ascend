# 保精度优化：同输入归因起点

执行顺序：先同输入分模块定位差距，再做单因素性能试验；有稳定收益后验精度，
最后才复测8K/128K整模型。当前未修改生产kernel、未合入新优化。

当前结论：B32的最大新增差距位于TopK→Attention，约326μs。复用物理plan并保留
Native邻对的MGather候选完整入口初筛快2.79%（30/30更快），对冻结版输出/cache逐bit一致。
但新增T192 Native覆盖发现冻结版此前未验收的精度缺口，因此暂不推广或合入。
已在独立候选修正Q-A/WKV；完整CSA前缀、cache及attention heads均已对齐，
Q-A/WKV/WO-A修正后的T192数值尾部已经对齐Native，MGather候选在同修正基准上
降低延迟3.15%，最终输出及完整物理cache已通过Native逐bit验收。
raw TopK有两个等分位置顺序不同，不能声称全部内部边界逐bit一致。
本轮8K真实模型已完成61440 token对Native一致；目标forward未体现加速，见末尾独立结果。
128K尚未新增验证；不能将单层收益当作整模型收益。

## 完整CSA单卡配对基线

冻结我方65bebf7f和nalinaly1574，公共Native框架41ba392f及共享SDK不变。
真实DeepSeek-V4-Flash layer4权重，共用合成hidden/history与Native builder metadata，
8K/B8/S6、T48，无dummy padding。是单层完整HC-pre至HC-post，不是整模型。
两臂各自按根签名准备Native权重视图；数值边界不同继续保留，不强制共用转换后的权重。

同一张允许的卡、同进程串行交替FULL NPUGraph，5次预热、30个Event样本/臂，
每次cache恢复及同步放在Event外；顺序轮换，双方eager/FULL输出及五组完整cache字节一致。
另检查hidden、readonly metadata及合法写入范围，图外权重/scale/绑定一直保活。
这些是每个实现自身的测量控制，不是nalinaly对Native精度通过。

| 指标（μs） | 我方精度版 | nalinaly |
|---|---:|---:|
| 完整入口Event中位数 | 894.91 | 671.07 |
| Event均值 | 895.43 | 672.63 |
| 独立DFX全核首尾跨度 | 745.78 | 524.60 |
| AIC最后结束，相对本侧首核开始 | 727.70 | 505.74 |
| AIV最后结束，同一时间原点 | 745.78 | 524.60 |

完整入口中位差223.84μs；独立DFX核内跨度差221.18μs，支持优先调查核内流水。
但DFX只采一窗，和Event不是同次调用，不能将两者相减解释成精确AICPU成本。
AIC/AIV驻留都包括等待，不能将AIV更晚结束等同于Vector计算就是瓶颈。
末尾AIV相对AIC收尾间隔都约18μs，也不能凭此排除某个更早阶段。
本轮尚未定位最大差距的具体子模块，不会拿该整体差值直接决定生产改动。

## 准备、运行及失败记录

- 首次CPU任务`task_20261102_100241_9742537193`和`...9743618552`：缺少历史runner模块。
  将同样的隔离导入逻辑直接放入新脚本，无需批量同步历史目录。
- 修正后CPU`task_20261102_100605_10398578905`及`...103990331066`均通过；未初始化NPU。
- 首次设备`task_20261102_101052_11577128175`：遗漏Native自定义算子注册，构造metadata时退出。
  与既有runner对齐，显式调用`enable_custom_op()`，没有更改任何数学。
- 修正后`task_20261102_101224_119377228599`通过，设备已释放，单任务低于15分钟。
  实际使用物理10卡；默认私有白名单仅8–13、15，不使用14或前八卡。

测量脚本为同目录`paired_single_layer.py`；先通过公共bootstrap选择Native冻结源码及共享SDK。
`--cpu ours`和`--cpu nalinaly`分别在不占卡任务预热；设备任务显式设置
`ASCEND_RT_VISIBLE_DEVICES=$TASK_DEVICE`，脚本中使用逻辑npu:0。
设备包装脚本、CPU预热日志、实际测量源码和原始结果均在小证据归档，未来复现需新的输出标签。
CPU预热命令体如下（分别在不占卡、780秒上限的任务执行，arm取ours或nalinaly）：

```bash
r=/data/pyptouser/yejia/vllm-nal-tnd-native-align-20261027-codex
source "$r/src/tests/pypto_test/nal_native_alignment_20261027/env.sh"
s="$r/cohorts/eightcard-source-v5/native/source"
export PYTHONPATH="$r/src/tests/pypto_test:${PYTHONPATH:-}"
python "$s/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
  --shared-toolchain --native-plugin-root "$s" \
  "$r/src/tests/pypto_test/csa_threeway_20261029/paired_single_layer.py" \
  "$r" "$r/results/new-unique-cpu-output" --cpu "$arm"
```

设备命令体沿用上述环境，额外设置允许的物理卡可见性，去掉`--cpu`、改用新输出目录，
并按归档包装脚本准备私有OPP；必须经task-submit分配单卡，不直接在登录进程运行。

`evidence/paired8k-baseline-20261102-evidence.tar.gz`，15,850 byte，
SHA256 `56e91780b25e1b052cff16ee986c8b026623d2a2c2c03c72de50a7e5a0253e48`。
实际测量源码SHA256 `ffcce6bb04bbe1e2ef8d6344c00d191dabb44c942729efe192d968a8fd81abc5`。

## 独立诊断阶段时钟

用户明确批准“允许仅诊断时钟插桩”。生产算子、共享SDK及正常性能路径不改。
诊断副本仅把原内部signal暴露成第56个入口参数，并返回以保活；撤销这三项后，
entry的完整AST与冻结源相同。时钟写在每phase/core独占512B槽的第二64B行，
不改首行业务信号；全部发布receiver追溯到同一signal且偏移为128个INT32的倍数。
MIX统一形参表与Python角色函数的形参位置不同，必须以host编排核验v56，不能猜测。

当前PTOAS实际使用150-tick原生退避。45个等待调用只包装计时，仍执行原来的
`PTOAS__WAIT_GE_BACKOFF<150>`；48个发布点包装写时钟。撤销包装与计数声明后，
生成C++必须与插桩前逐字相同；不使用旧工具中会改写退避行为的TWAIT替换。
首次CPU尝试两任务因旧工具只识别TWAIT而退出，未使用NPU、没有计时结果：
`task_20261102_103709_180810913463`与`...18081484145`。

CPU构建后核对host编排，再将审计绑定到实际缓存的两份核源码和二进制身份；
设备流程必须在首次执行前重复核验，不能把缓存命中当作自动审计通过。
插桩版eager及FULL的output与五组完整cache都必须与各自未插桩版本逐字节相同，
同时通过只读输入和写域检查，才导出阶段结果。这里不是nalinaly对Native精度验收。

阶段时钟仅用于归因，不作为性能收益。发布端点之间的wait只覆盖这些GM等待，
其余跨度还包含搬运、FFTS等待及探针本身；不能命名为纯计算，也不能把跨核并发
阶段简单相加。正常性能继续以未插桩、同卡交替FULL graph为准。

## 时钟验证与归因结果

`task_20261102_111458_25506232942`成功退出，约2分钟内完成，仍只使用允许的单卡。
两臂插桩eager、FULL及五次重复FULL的output、五组完整物理cache均与各自未插桩控制逐字节一致，
readonly与写域保护均通过。每臂每窗1590条有效发布记录，频率50MHz。
这证明探针未改变本组数值，不代表nalinaly已对齐Native。

同任务未插桩Event仍交替保留全部30样本：我方中位850.43μs、均值849.10μs；
nalinaly中位622.62μs、均值632.17μs，后者含一个893.82μs样本，未静默剔除。
普通中位差227.81μs与此前223.84μs相近；不能把跨任务绝对值变化当优化收益。

下表为五窗各自对齐本侧最早BEGIN后的末生产者发布完成时刻，再取中位；单位μs。
是诊断里程碑，非独立模块纯计算时长。

| 里程碑 | 我方精度版 | nalinaly | 累积差 |
|---|---:|---:|---:|
| INPUT：含输入及权重打包 | 52.48 | 19.80 | 32.68 |
| PARTIAL：HC线性与输入平方partial | 76.80 | 29.08 | 47.72 |
| RAW：线性及RMS按段归约 | 112.68 | 31.56 | 81.12 |
| NORM：HC混合与输入Norm完成 | 133.56 | 50.94 | 82.62 |
| TopK完成 | 341.38 | 246.86 | 94.52 |
| Attention Vector完成 | 508.98 | 348.26 | 160.72 |
| OUT：HC-post完成 | 668.74 | 460.76 | 207.98 |

本排序仅限8K/B8/S6、无dummy的这组输入，不直接外推到B32或128K。
主要差距优先级为HC前段、Attention、输出尾段。粗边界差的新增量分别约83、66、47μs；
TopK之前的中间投影/Indexer链新增差约12μs，不应优先纠缠Score的小差。
它们包含重叠及探针扰动，只有完整未插桩单因素A/B才能确认可兑现收益。

进一步观察：到Attention发布的AIC每核间隔中位，我方186.81μs、已覆盖GM等待45.42μs，
nalinaly120.38μs、等待45.62μs。AIV对应178.30/30.27与110.10/28.77μs。
这组证据不支持把该段差距主要归咎于已插桩的GM等待；尚未区分其余部分的MTE、FFTS与计算。
输出末端AIV主要在等AIC，不能靠AIV末端优化来消除O-proj成本。

地址证明测试在227运行4项全部通过，包括拒绝其它allocation、非512B槽和错误host参数映射。
CPU封存曾两次失败：一次环境脚本覆盖PYTHONPATH；一次未考虑SDK原有宏/include预处理。
修正后仍要求审计文本经SDK同一预处理后整段出现在两份实际核源码中，并匹配二进制身份。
无测试被削弱，失败封存均未进入设备执行。

## 第一项单因素候选（尚未验收）

仅改`hc_pack_input_from_bf16`：将32次8列切片重复拼接改成一次Tile gather。
旧的BF16→FP32→BF16、padding清零、HF32模拟、K768/22段归约及所有发布/等待保持原样。
索引将每行每8列重复两遍；标准CPU检查独立构造旧拼接顺序，核验8192个目标索引。
候选只在独立目录，正式精度基准保留。

使用同一配对runner的`--candidate <独立source目录>`，同卡交替测30对；
预设先通过中位≥1%且至少24/30对更快，再比较output及完整cache到冻结精度版。
若通过，再做Native分模块边界验收；整模型8K/128K只在稳定收益且精度验收后考虑。

### HC输入打包：不采用

- CPU初版INT32标量位操作被A3拒绝；INT16版直接转INT32也被现有legalizer拒绝。
  第三版用INT16精确计算索引，再显式经FP32转INT32；0..4095范围无损。SDK未改。
- 第三版CPU构建通过，静态Vec地址上界155,872B；gather src/index/scratch/out不重叠，
  src为FP32连续`[1,4096]`。这些检查只是上卡前资源门槛。
- `task_20261102_113838_296746726286`成功退出，物理8卡；单任务约1分钟。
  未插桩FULL配对：冻结版中位880.11μs、均值880.97μs；候选中位892.29μs、均值892.48μs。
  **候选慢1.38%，仅4/30对更快，不通过性能门槛。**
- 两臂各自eager/FULL输出和全cache自一致、guards通过；跨实现精度资格按约定跳过。
  不合入，不跑该候选的Native八边界或整模型验证。

### 第二项：HC RMS按段块读取（尚未验收）

回到冻结版，不叠加失败的输入打包候选。全22段矩阵视图的尝试被当前SDK的
核内Tensor/Tile提升及specializer元数据推断拒绝，均在CPU结束，未上卡，也未改SDK。
最终最小候选保持原ABI和编排：每段一次`[8,16]`载入，在Tile中取8个原对角线。
**part0再依次加part1..21**的FP32顺序及variance/sqrt/div保留；1.0常量改为等值
Tile常量，写回改成同域8×1 Tile store。真实行、padded行选中的对角线地址都与原实现一致。
新块覆盖同一已完成producer的8行，不提前读取另一producer的数据。
CPU地址枚举覆盖p16=16、32、48、64、128、256和各M8起点。正式精度基准仍未改。

若性能稳定胜出，先比完整输出/cache到冻结版，再用同输入Native FULL核验输出及五组完整
物理allocation，并验证Native warmup/FULL自一致。此轻量检查不观察内部TopK，不称八边界资格；
也不能代替整模型逐token验收。失败即停止后续配置。

阶段证据包：`evidence/paired8k-phase-clock-20261102-evidence.tar.gz`，2,121,028B，
SHA256 `d409cb3dd18eb13227fbab97be2fac6c3402362d96103ec278fb4f55d5180d26`。
包含五窗JSON/Perfetto、CPU审计及启动脚本；该包的clock代码是诊断源码，不用于正常性能。

## B32复核：主要差距转为Attention

为解释此前约1200/1800μs的量级，新增同口径8K/B32/S6、T192、无dummy对照。
沿用同一卡串行交替、5预热+30未插桩Event样本；独立时钟五窗，每臂每窗1685条发布记录。
`task_20261102_123511_40145351719`成功退出，物理8卡，单任务约2分钟。
插桩eager/FULL及五次重复FULL的output和全部物理cache均与各自未插桩版逐字节相同，guards通过。

| 未插桩完整入口（μs） | 我方精度版65bebf7f | nalinaly1574 |
|---|---:|---:|
| Event中位数 | 1637.83 | 1068.40 |
| Event均值（保留全部样本） | 1636.61 | 1066.92 |

中位差569.43μs。诊断完成里程碑如下，仍不能称为纯计算时间或独立模块耗时。

| 完成里程碑（μs） | 我方精度版 | nalinaly | 累积差 |
|---|---:|---:|---:|
| HC/input Norm | 191.34 | 64.22 | 127.12 |
| TopK | 587.74 | 388.74 | 199.00 |
| Attention Vector | 1193.86 | 668.86 | 525.00 |
| HC-post最终输出 | 1459.26 | 881.10 | 578.16 |

从TopK就绪到Attention完成的差距新增326.00μs，约占该诊断末端差距的56%；
HC为127.12μs，中间投影/Indexer新增71.88μs，尾段新增53.16μs。
因此B32应优先处理Attention，不能把B8的HC排序直接外推。
Attention对应AIC间隔中位630.27/304.76μs，已覆盖GM等待59.38/53.21μs；
AIV间隔604.26/277.47μs，GM等待30.58/26.32μs。GM等待本身只多约4～6μs，
其余仍包含标量地址解析、DMA、FFTS、计算和探针成本。

源码发现一个明确可检验的差异：nalinaly复用本轮物理plan并批量读索引、MGather；
我方Native精度路径为保留邻对重排，仍读取原TopK，每行重复页验证、换对、逐行搬KV，
未消费已生成的物理plan。下一项独立候选只替换compressed KV读取，保留SWA与原Scalar回退，
保留Native邻对顺序、512 softmax树、概率CAST_ROUND、PV累加与全部事件协议。
地址和换对比较使用INT32，不将物理地址转FP32；只有position<2**24且producer/consumer
同一physical_plan_supported条件成立时使用已发布plan，防止表示或上界语义不一致。

RMS按段块读取A/B的完整数字：冻结版880.48/878.65μs（中位/均值），候选874.16/874.43μs；
中位小幅下降0.72%，仅17/30对更快，未通过稳定收益门槛，因此未做跨实现精度、未合入。
显式tensor.view单panel版本已完成CPU构建，暂不占卡追小项，先验证B32 Attention大差距。

## Attention物理plan与Native邻对批量读取候选

独立包保留冻结精度版，未修改生产源码。只把compressed KV的逐行寻址/读取替换成
64行物理plan读取与MGather；每个64行块内先按Native条件恢复相邻Key的交换顺序。
位置、页数、布局不满足原plan条件时仍走原Scalar路径；SWA路径保持原样。
物理地址和邻对条件全用INT32；物理行差为正且不超过2**21、两项均有效且不是最后Key时交换。
原QK/PV分段、softmax归约树、概率BF16舍入、inverse RoPE及同步协议保持不变。

`test_physical_pairs.py`在227的CPU任务完成3项测试，包含5,000组随机样本、无效槽、
分页空洞、INT32极值、最后Key及字节跨度边界。逆AST检查证明除了新增读取分支和传入
已存在的physical plan，原attention数学及编排没有变化。
`task_20261102_131724_49339116354`不占卡构建成功；B32产物身份
`0e58c4496a37abb2cdce6d03d0b2a98a27046b6ad85107bc0cf46a4905f46ed8`。
生成PTO的Vec地址上界155,872B，低于192KiB；CPU成功不能作为设备精度或性能结论。
独立生成审查确认pair计算全为INT32、MGather为64×512 BF16/Zero，关键活跃UB不重叠；
读完成后保留MTE2→MTE3同步，再TSTORE，原PIPE_MTE3 READY及下轮复用依赖仍在。
设备A/B仅测未插桩版本，达到中位至少1%且30对中至少24对更快后，
再核验完整输出和五组物理cache对冻结版、对Native逐字节一致。尚不代表整模型token验收。

B32归因、两项HC负面结果及Attention CPU收据归档：
`evidence/b32-attribution-and-candidates-20261102-evidence.tar.gz`，940,885 byte，
SHA256 `dce27c2b4462de1dccfc1722b42da3e9bffb7ea751bd88363ce7e8a8cc9499a7`。

### Attention首次设备结果：有收益，Native资格未通过

`task_20261102_133031_70133924400`在物理8卡运行约1分钟后退出并释放卡。
未插桩配对FULL结果如下，30个样本全部保留。

| 实现 | 中位μs | 均值μs | 候选更快的配对数 |
|---|---:|---:|---:|
| 冻结精度版65bebf7f | 1638.27 | 1640.32 | — |
| physical plan + Native邻对MGather | 1592.53 | 1592.21 | 30/30 |

中位耗时下降2.79%，通过本轮初筛。两臂eager/FULL自身一致；候选输出和五组完整
cache与冻结版逐字节一致，故本次读取改动没有改变该组冻结版数值。
随后Native完整HC-pre至HC-post检查：自身eager/FULL一致、guards通过；
但候选对Native的output和SWA不同，compressed/state/indexer/indexer_state均相同。
任务因此按首差策略失败退出；不再测其它T、128K或整模型，不合入候选。

**这同时暴露冻结版本在本组检查上的资格缺口**，不能归因于MGather引入误差，
因为候选与冻结版完整输出/cache已相同。也不能把该组称为Native精度通过。
只读复核发现T192仍走旧WKV大M路径，现有dispatch只特判T240及T<64；
另新增轻量Native检查使用enforce_eager=True，与此前生产FULL配置不同。
尚未确认根因。下一步复用已有probe_spmd的生产FULL配置和观测门禁，
先逐阶段验证输入Norm、WKV、KV Norm/RoPE到SWA；首差即停，不修改正式基准。

首次A/B全部样本、Native检查和失败日志归档：
`evidence/attention-pairs-ab-20261102-evidence.tar.gz`，11,353 byte，
SHA256 `cad4d711bd47d6153884fcf3832017525349623541cdeb484cf8b1d95de0222b`。

### T192生产FULL首差定位准备

复用已有`nal_native_alignment_20261027/probe_spmd.py`，长度列表为32个6，history8192、
seed1024、真实layer4；其FULL_DECODE_ONLY、static kernel、DSA multistream等配置不改。
增加测试门禁：生产55 eager/FULL自身精确，再对57 FULL的输出及五份完整物理allocation逐字节
比较。通过标记写入production.pt，观测加载必须验证，避免旧57-only参考混入。
85参数观测入口仍需与57生产入口的外部输出、TopK/scores精确，才能信任内部stage。
实际加载30份stage源码与冻结65逐字一致，不使用SOURCE_ROOT来猜测PTO绑定。

三项CPU任务预热55/57/85均通过，分别耗时204.00/205.29/209.36秒，未初始化NPU。
55/57控制`task_20261102_134531_97567114656`完成，用时38.00秒：生产55 eager/FULL、
55 FULL对57 FULL的output和五allocation均逐字节相同；57自身FULL/eager及guards也通过。
后续只跑同组内部首差观测，不扩大测试矩阵。

`task_20261102_134805_101511611421`阶段观测完成，33.73秒，首差后停止：
原输入/缓存/metadata全部匹配production.pt，85入口对57 FULL的八个外界及TopK scores
逐字节一致，eager/FULL与guards通过。HC mixed、post gate、comb、input Norm均对Native
逐bit相同；**首差是Q-A投影BF16**，37个元素不同，max_abs=0.00048828125。
因此不能把WKV先定为根因；尚未执行到WKV检查。确定性torch/level1/HCCL均生效。
下一步只用本次保存的input Norm及Q-A金值，隔离原N128/K256、N32/K256自然顺序和
已有T240列次序假设；原臂必须先重现37个差异，规范F.linear必须先重现实际CV金值。
这些是诊断假设，未证明哪一种适用于T192，不改正式分支算子。

隔离任务`task_20261102_135550_113108332325`完成，21.34秒，物理允许单卡：
规范F.linear先逐bit复现实际CV金值，原N128/K256臂逐bit复现保存FP32 accumulator，
并精确重现37个BF16差异。仅N32细分仍有相同37差；N32并使用已有T240列次序后，
本组T192的Q-A BF16对Native逐bit一致。三臂FULL/eager、输入/权重只读、写域guard均通过。
这支持本组Q-A差异来自累加顺序，不支持仅N32布局就能消除差异。
三项微型CPU编译均未初始化NPU，各约12秒；后续接入独立候选重新从HC开始验证，
不将一次微型结果当作完整CSA或所有输入精度通过。

T192原始控制/首差/微型三臂的小证据包（不含大型tensor快照）：
`evidence/t192-first-difference-20261102-evidence.tar.gz`，47,481 byte，
SHA256 `7342150cbc921b947ddec73c1be9c26b0b7e792a6bc73428cdd314c79d719674`。
输入按上述真实层权重、固定seed和长度生成；完整tensor快照保留在本轮自有远端结果目录。

`qa192_precision_candidate.py`从冻结65创建独立包，只增加T192 Q-A分支与已测helper，
逆AST核验其余原实现不变。common和测试harness随副本冻结，避免SOURCE_ROOT误绑定算子。
候选没有叠加Attention优化；先完成生产55/57/85预热及同组从HC开始的FULL阶段验收，
再考虑修正后基线与Attention候选的组合。当前正式算子仍为冻结65，未合入任何新性能改动。

独立Q-A候选55/57/85 CPU预热全部成功，分别212.76/214.35/204.92秒，未初始化NPU。
候选生产控制`task_20261102_140551_129288929961`完成，40.48秒，55 eager/FULL、
55 FULL对57 FULL的output和五allocation均逐字节一致；57 FULL/eager及guards通过。
实际common与测试harness来源另封存SHA收据，执行候选目录下的probe，不通过改Native路径选包。

候选阶段`task_20261102_140809_134549520173`完成，40.29秒，观测门禁与自控通过；
HC/Input Norm/Q-A BF16/Q-R量化值及scale全部对Native逐bit一致，确认Q-A修正进入完整链路。
首差推进至WKV投影BF16：35元素不同，max_abs=0.00048828125。此处停止，不继续检查后续模块。
下一步只复用该次保存Norm输入、WKV accumulator与CV金值，隔离WKV的原N128/K512、
N32/K256自然顺序和已存在T240列累加顺序；原臂需FP32/BF16均重现，候选不改正式算子。

WKV微型`task_20261102_141504_151969631448`20.33秒后失败退出，设备释放。
原N128/K512精确复现整层FP32和BF16边界，Native规范linear复现实际CV金值；
N32/K256自然顺序仍35差，已有T240列顺序仍32差，max_abs均0.00048828125。
三臂FULL/eager、只读和guards均通过，但不能认为任何候选解决WKV精度；不接入该假设。
按32列统计，原臂差异数为[0,0,3,3,2,5,1,2,2,4,2,1,2,3,2,3]。
下一步只局部观察实际Native WKV kernel/tiling信息，不把带profiler结果当普通性能。

局部Native采集`task_20261102_142219_166635614331`和独立进程tiling日志
`task_20261102_142940_183590122015`均已退出，前/中/后金值及只读检查全部逐bit一致。
实际路径`aclnnMatmulWeightNz_MatMulCommon_MatMulV2`，BF16 ND×FRACTAL_NZ，
确定性level1、high_precision、24核；不是从其它版本MatMul源码推测。
日志最终选择best_idx7、TilingKey230097、152B：L0 M64/N64/K128，
L1 M64/N64/K256，核网格M3×N8；三次launch有效tiling字段一致，split_k_flag=0。
这些字段确认本次N64布局，**没有直接证明K起点或顺逆**。
因此另建MODE3假设：每N64两N32共用K256顺序，起点0/15/4/3/8/7/12/11交替正反，
原失败MODE2保持不变。先原输入原臂FP32/BF16复现，再验MODE3；若通过再测4个独立
CPU随机BF16输入（seed2048/4096/8191/65536，幅度0.05/0.5/2/16），首差即停，
每组保存实际输入及Native/eager/FULL数据。仅为T192、layer4权重的有限资格。

N64顺序假设微型`task_20261102_144407_218298116308`通过，22.45秒：
原路径FP32/BF16重现35差；MODE3原输入及4组独立输入的BF16输出全部对Native逐bit一致，
各组FULL/eager、只读及guard检查通过。不能推广为其它T或其它工具链通用结论。
独立Q-A+WKV组合只新增T192路径，其余分派逆AST保持，原基准和Attention候选均保留。
组合55/57/85 CPU编译通过，用时211.81/208.23/208.12秒，均未初始化NPU，
随后再执行组合完整CSA验收，不直接依据微型结果合入。

组合55/57 FULL控制`task_20261102_145534_247258523167`通过，52.26秒。
完整阶段`task_20261102_145942_26054679187`60.17秒后在raw TopK首差停止：
Q-A/WKV/QK Norm/RoPE、主/Indexer压缩器、全部Native cache、Indexer query、量化及权重
共31个已检查边界全部逐bit一致。TopK只有row183/rank60、61两个位置交换820/1967，
分数均为完全相同的FP32值2.7790469175670296e-05。
当前G2按1152半叶排序，已有G6路径按Native2048分段；不能直接等同为选错Key。

CPU传播分析：该对Key的物理行为63535/64692，两种raw排列经Native邻对规则后
有效顺序相同。正式输出却在另外8行（7/42/69/100/103/137/139/164）有639个BF16元素差，
不在row183。O-proj A中间结果有525个BF16差异，分布在多行。
因此不把最终误差直接归因于此raw TopK排列，也不先改变G2/G6策略。
只补存已经计算的attention heads，保留原raw TopK首差断言，再用CPU区分heads与O-proj A。

WKV修正/Native tiling/候选来源证据包：
`evidence/t192-qa-wkv-diagnostics-20261102-evidence.tar.gz`，89,575 byte，
SHA256 `226e6af990b070f578c89253eb1045a638cbeb607df8f02c0a63f83add3d1ab6`。

补存heads任务`task_20261102_152333_323143631460`63.26秒退出，仍保留raw TopK首差：
全192×64×512个BF16 attention heads对实际Native heads逐bit一致。
因此本组正式误差已收敛到O-proj A，而不是attention或该对TopK排列。
只复用保存的、已对齐Native的真实heads及正式WO-A权重，隔离T192的N子块/K起点；
原臂必须先重现525个BF16差异及完整FP32 accumulator，原T240测试保持不变。

WO-A首次CPU预热因上传未确认完成而读到旧CLI：
`task_20261102_153140_34909278515`、`...349097432047`、`...3491056216`均exit2，
报woa192模式不在choices。没有占卡、没有数值结果。保留v1日志，完成有界rsync并
验证远端三文件及新choices后，以v2新标签重试；不将准备失败计为算子失败。

v2三个CPU编译均通过，各约14秒，未初始化NPU。WO-A微型
`task_20261102_154902_402508732366`32.82秒完成，**JSON pass=false**（任务exit0不等于精度通过）：
相同实际heads逐bit一致，Native transpose-batchmatmul先复现已存Native金值；
原臂完整FP32重现，BF16对Native仍525差；仅N128仍525差，借T240核归属/起点仍517差。
三臂各3次FULL/eager、只读和guard通过，但没有候选解决精度，不接入。
只继续观察当前Native BMM实际tiling，不再直接推广T240映射。

Native WO-A局部采集`task_20261102_155743_3287513423`完成，物理9卡，
Native前/中/后输出与实际保存金值逐bit一致，输入及权重只读。
实际为`aclnnTransposeBatchMatMulWeightNz`，TilingKey32，最终数据376B，24核。
日志明确beforeTune baseM/N/K=64/128/128，afterTune=192/128/80；
不得把beforeTune打印的stepKa8/stepKb4无条件当作最终迭代参数。
下一步先按本实验固定CANN的TCubeTiling定义解码最终数据，再构造单因素候选。
带profiler和debug日志的时间不作为正常性能结果。

### WO-A：按实际Native tiling隔离K分块和累加起点

固定CANN的TCubeTiling字段解码最终376B数据：baseM/N/K=192/128/80，
stepKa=stepKb=4；因此L1窗口为320，4096列共13窗，末窗为256。
Native前/中/后输出与保存金值相同；采集时间不列入普通性能。
公开CANN源码仅用于解释设计，不代表其commit就是本机二进制来源：

- [TransposeBatchMatMul NZ选择K_SHIFT](https://github.com/XtangEver/cann-ops-nn/blob/01a4acd66abcbc1661b92df37e2d0dbda1dbc117/matmul/transpose_batch_mat_mul/op_kernel/transpose_batch_mat_mul.cpp)。
- [KLoop按物理核编号轮转窗口](https://github.com/hicann/asc-devkit/blob/f16994e64df8936ea1e6cbcc47c7944f62fc2264/impl/adv_api/detail/matmul/scheduler/iterator/k_loop/k_loop_mdl_reorder.h)。

64个Native逻辑块按24核分配，前48块owner=logical//3，后16块owner=16+(logical-48)//2。
候选保留原PTO父任务M128/N256归属，只以Native逻辑块计算K起点，不使用PTO worker号代替。
K80自然顺序是独立控制；轮转候选first_panel=(owner%13)*4，然后按52个80列面板环转。
末面板有效K16；不对4096直接取模，以免破坏最后256列窗口。
MODE0/3/4无设备编译通过，约14/14/12秒，未创建Worker或初始化NPU。
独立源码审查未见阻断，硬件资格仍以随后FULL对拍为准。

heads、旧WO-A三臂失败、Native实际tiling及源码解释的证据包：
`evidence/t192-heads-woa-diagnostics-20261102-evidence.tar.gz`，73,292 byte，
SHA256 `fbc4e88f0b123d907defe09ed76be51957fd03e877ccfd23c2d0d8e91da6a184`。

WO-A K80单因素硬件对拍`task_20261102_172339_26563467010`通过：
原臂完整FP32重现、BF16仍525差；K80自然仍525差；K80+Native owner窗口轮转为0差。
三次FULL回放与eager逐bit一致，guard及只读检查通过，任务已退出。
补验`task_20261102_172652_271567314957`25.62秒通过：
同一实际输入及四组独立BF16随机heads（seed2048/4096/8191/65536，
幅度0.05/0.5/2/16）全部Native BF16零差、FULL/eager零差、只读与guard通过。
这说明本组WO-A的实际差异来自K窗口累加顺序，而不是仅改变baseK即可消除。
仍仅覆盖T192、layer4真实权重，不代表整模型或所有T。

生成器`qa192_precision_candidate.py --with-wkv-group64 --with-woa-k80`创建新副本；
原65bebf7f、Q-A/WKV副本和原Attention候选均保留。
完整入口按group_begin/end过滤原64父任务，不重复计算其它组；
仅tokens192、M128/N256分派走新helper，逆AST核验其它分派未变。

组合55/57/85无设备编译分别215.26/222.14/210.59秒通过。
生产FULL控制`task_20261102_173235_282827730967`39.66秒完成并退出；
55→57 FULL输出和五组完整allocation逐bit相同。
Native阶段任务`task_20261102_173417_286302828264`已提交；
修正基准+MGather的无设备预热`task_20261102_173235_282803521269`已提交。
随后SSH连续三次被服务端主动关闭，尚未读取两项最终结果；没有重提或假定通过。
任务均有720秒内层、780秒外层上限；本轮仍只允许物理8–13、15。

下一次性能命令必须同时设置：
`--baseline <qa-wkv-woa192-precision.../source>`、
`--candidate <attention-pair-qa-wkv-woa192.../source>`。
其`candidate_precision_vs_frozen`旧字段在此表示对所选修正baseline，而非原65；
结果`sources`记录实际身份。只有两边共享修正后，才可将收益归于MGather。
目前尚无这次修正基准上的新收益比例、Native整层验收或整模型token结论。

SSH正常重试恢复，未改变登录方式。原阶段任务60.61秒完成并退出：
原raw TopK首差保持（2个等分位置交换），已计算的attention heads、WO-A BF16和最终输出
三项均与Native逐bit一致。该诊断的`pass=false`仍保留；不能声称所有内部边界都逐bit一致。
31项前缀/cache/query等边界保持通过；Native外部完整allocation验收还需独立性能候选门禁。
修正基准及搬运候选的配对命名空间CPU预热均通过，未占NPU。
只针对这两份共同修正的源码执行新的同卡配对，不沿用旧2.79%作为本组收益。

## 修正基准上的单因素验收结果

`task_20261102_174412_305704125153`正常exit0，设备已释放。
仍为真实layer4权重、共用合成输入/history、8K/B32/S6、T192；单卡完整HC-pre至HC-post，
没有整模型token统计。相同数值修正共同放在A/B两边，只改变compressed KV的物理plan邻对MGather。
5次预热后30次交替采样；恢复cache及同步不计入Event，普通性能未加时钟插桩。

| 指标 | 修正基准 | 修正基准+MGather |
|---|---:|---:|
| FULL入口中位数（μs） | 1753.70 | 1698.46 |
| FULL入口均值（μs） | 1753.15 | 1697.75 |
| 中位耗时降低 | — | 3.15% |
| 配对更快样本 | — | 30/30 |
| 相同完整输出/cache | 参考 | 逐bit一致 |
| 候选对Native最终输出+五组完整物理cache | — | 全部逐bit一致 |

Native自身eager/FULL也逐bit一致；只读与合法写入域检查全部通过，outside_slot_bytes均为0。
候选外部Native门禁不观察raw TopK，不能替代上面保留的raw TopK失败记录。
修正基准比原65更慢，包含新增的Native数值顺序修复；不可把1753.70与旧1638.27相减
归因到MGather，也不可把nalinaly旧测值当成本组同卡新基线。
当前尚未新增8K/128K整模型token/性能结论；临时交付基准不改。

T240回归`task_20261102_175514_32341852953`exit0：8K/B40/S6单层FULL，
相同5预热+30交替样本，中位1952.80→1903.26μs（降低2.54%）、均值1950.86→1901.77μs，
30/30配对更快；候选对共同修正基准及Native完整output+五组allocation均逐bit一致。
T192的前/后15对分窗中位收益分别2.90%/3.22%，未见只在一个窗口占优。

用户另行明确批准本轮8–15含14进行八卡真实8K复测，仍禁止前八卡；该许可不是默认规则。
`prepare_candidate_cohort.py`仅复制v5原manifest内文件，逐项核对旧SHA，再替换唯一TND包。
独立审查指出应核对candidate manifest SHA，已补：复制前检查全部changed_files，
复制后校验整个TND包字节清单；非math文件保持原SHA，原快照/金标均不写。
准备v1只做无设备预检，实际整模型使用补齐证据绑定后的v2新快照。
沿用原DP=EP8、每rank40题分5波×8、FULL6/240、4000blocks、192token和旧Native金标。
8K成功前不运行128K；本轮尚未提交或获得新的整模型结果。

已完成T192/WKV/WO-A修正、组合FULL及两档单因素MGather的证据归档：
`evidence/t192-fixes-and-mgather-20261102-evidence.tar.gz`，538,542 byte，176个有限文件，
SHA256 `8e7132d13a74d7f503d383c5e0cc639e440620894f84c48e3789935f32787aef`。
包含原始30样本、完整Native外部cache门禁、失败/控制结果、脚本及隔离数学包；不含大张量。
真实模型测试将复用原v5金标；尚未取得结果前，本报告单层结果不能当作token验收。

### public预热准备失败及修复

public T6/T240 v1预热分别`task_20261102_180058_33313906829`、`...333150611369`
在导入阶段exit1：新副本缺少旧运行生成、但未列入原manifest的`vllm_ascend/_build_info.py`。
旧v5的ours/Native该文件字节相同，仅声明设备类型A3；没有重装环境或探测其它设备。
v1/v2准备快照保留，改用v3，显式复制该文件并记录SHA。
预检新增actual public CSA entry导入及其source路径检查，仍要求未初始化NPU。
此前仅轮询结果文件缺失，误称任务仍编译；已改为同时核对任务状态和首个错误。
本次准备失败未占NPU，不影响已取得的单层性能/数值结果。

## 整模型复测准备及当前任务

v3实际public入口无设备预检通过。public预热v2任务
`task_20261102_180940_34748553715`（T6，188.59秒）和
`task_20261102_180940_34748939579`（T240，220.13秒）均exit0、未初始化NPU。
精确使用服务模块名、相同SDK及task window2048；不以诊断namespace缓存代替。

唯一真实8K任务为`task_20261102_181620_35937654977`。
提交首次SSH中断后，先用`task-submit --find`确认唯一命令无任务，才重试；没有重复作业。
截至18:40仍pending、未占卡；有效私有候选池为8–15，8/9被其它任务占用。
本轮用户单独批准14，0–7仍禁用；未来复现需重新确认14许可。

沿用既有`run_eight_card.sh`与v5 Native金标；唯一新增为CSA数学候选包。
候选cohort=`eightcard-attention-pair-precision-20261102-v3`，arm=`ours`，
输出标签=`attention-pairs-real8k-20261102-v1`，reference=`last8-native-8k-waves8-v5`；
参数为`--inspect-token-sequence --measure-waves`、PTO window2048。
仅一次模型运行，192token首差停止，内层720秒/调度780秒；不重跑Native2。
验收后还需将本轮性能样本与`last8-native-8k-wave-perf-v1`对齐后再报告速度比。
当前没有新增整模型token或速度结果，不以单层字节对拍代替整模型验收。

后续128K原bank已确认可复用：原10月14日128K/B24目录的fixed131072，24题，
但历史Native参考是DP=EP16，不能直接充当新DP=EP8的同配置性能/精度基准。
本轮尚未运行128K；必须8K通过后再确定八卡容量、入场及基线方案。

整模型准备证据：`evidence/real8k-setup-20261102-evidence.tar.gz`，86,524 byte，
SHA256 `06f1bb162896a6669e6dce627d02f66a0c3c9d77de2f2164666d99f68b9cd329`。
包含原失败导入、修复后的public预检/预热、v3全文件SHA及本轮卡号约束；不含模型运行结果。

## 8K真实模型结果与释放设备

原任务`task_20261102_181620_35937654977`已结束并释放全部八卡。
运行在物理8–15，任务日志TASK_RUN_START=18:55:45、TASK_RUN_END=19:06:16，明确释放8–15全部锁；
模型进程630.728秒（约10分31秒）。调度完成列表的49分57秒
包含约39分钟pending，不能当作占卡时间。用户随后要求14留给CI，后续14许可撤回，
恢复默认仅8–13、15；本轮不再提交八卡或128K任务。

| 实现 | 对Native输出token | 目标forward最慢rank均值 | 性能输入门禁 |
|---|---|---:|---|
| Native `41ba392f`，复用 | 61440/61440 | 73.494 ms | 通过 |
| 原精度版`65bebf7f`，复用 | 61440/61440 | 77.079 ms | 通过 |
| 本轮数值修正+MGather候选 | 61440/61440 | 77.652 ms | 通过 |
| nalinaly `1574ed59`，复用旧性能运行 | 不一致 | 69.707 ms（仅描述值） | 未通过，positions不同 |

本轮8rank×40请求×192token，共61,440，输出全部与Native金标一致、零抢占，
21个CSA层覆盖检查无错误；并与旧65的token和接受事件均一致。
但候选及旧65的接受事件均与Native不同，不能称整模型内部执行逐bit一致。
严格postflight继续保留该失败，因此外层任务exit1；子生成进程exit0且完整生成，
没有首个输出token差异或720秒超时。不能只用exit1判断token失败，也不能写成全验收通过。

160个有效目标forward样本与Native的实际B/T/padding/positions门禁通过。
统计口径沿用[原整模型报告](REAL_MODEL_PERFORMANCE_20261031.md)，含HCA/FFN等，
不包含draft，使用八rank均值的最大值，不称同一步EP关键路径。
候选77.652ms相对旧77.079ms观察慢0.74%，单次窗口没有证明整模型收益；
单层T192/T240的稳定改善不能直接推广为真实调度下的整模型加速。
本轮两次模型运行并非同卡交替A/B，背景负载不同，不能据此断定MGather造成回退。
保持原默认精度基准，候选及已验数值方向独立留存，不通用合入。

## 下一项独立候选（尚无设备结果）

只读审查提出：SWA QK/PV只读取KV GM首128行，因此可将CMP part3的两个64行MGather
提前到SWA PROB_READY之后、PV_READY等待之前，写入GM偏移384/448；
CMP阶段跳过已搬part3，最终KV_READY及全部数学/事件不改。
该方案避免跨阶段新增UB暂存；回退条件和原physical fastpath一致，并要求SWA/CMP均有效。
`attention_prefetch_candidate.py`在新目录生成，逆AST核对通过、增量pre-commit通过，
独立源码审查未发现漏写/重写/事件阻断。
两项无设备构建已通过，仍需生成物资源/依赖审查和未来性能先行；本轮没有提交其NPU任务，
不将编译通过当作收益或精度验收。当前真实8K使用的MGather/v3快照保持不变。

预取候选的CPU生成物独立审查：候选和基线Vec最大alloc末端均155872B，
距196608B上限余40736B；预取64×512 BF16 buffer范围[32768,98304)，
I32索引[98336,98592)，与running右[0,32768)、左[123104,155872)不重叠。
已看到MGather→MTE2/MTE3等待→TSTORE，以及后续UB复用同步和最终MTE3的KV_READY。
SSH间歇关闭，最后一段PV-left局部常量/单条load前的细查未再次取全，保留该限制；
不因编译/资源审查通过而授予硬件正确性或性能资格，本轮不启动其NPU任务。

8K完整结果证据：`evidence/real8k-mgather-results-20261102-evidence.tar.gz`，224,956 byte，
SHA256 `e506658df1b136b376e9963d99fcffa1e7beba550a8f26e0a79b6d09c4bac142`。
含全部八rank的61440个token、接受事件、160性能样本、逐输入门禁、源码清单及实际释放锁日志。
不含模型权重或bank大张量；本轮没有新增128K结果。
