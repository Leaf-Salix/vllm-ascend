# 保精度优化：同输入归因起点

执行顺序：先同输入分模块定位差距，再做单因素性能试验；有稳定收益后验精度，
最后才复测8K/128K整模型。当前未修改生产kernel、未合入新优化。

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
