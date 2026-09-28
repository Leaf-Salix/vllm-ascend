# 完整 nalinaly 参考复现：2026-09-28

## 范围与当前结论

本轮先复现完整参考实现及其原始测试，再按完整模块迁移到 Leaf BSH 分支。
此前局部候选的结果不能代表完整参考实现。当前已完成独立环境构建、完整 kernel CPU 编译及原版单层 graph 预检。
单层输出对Native有误差，该档没有提速；七档整模型复现和模块迁移尚未完成。

参考完整性能 kernel 融合 HC pre、input norm、CSA 与 HC post，输入输出包含 mHC 残差流。
现有 Leaf DSA 入口处理已经归一化的 attention 输入，融合边界不同。
后续完整模块迁移需要保留原生 DecoderLayer 的外部签名和返回值，并明确内部融合边界；
不能只移植若干 tile 常量后宣称对齐。

## 固定版本

| 部件 | 本轮参考版本 |
| --- | --- |
| vLLM-Ascend 完整参考源码 | nalinaly `71153bb3be9faa6c79afecd92e229e765cef37c5`，原样归档，逐文件 SHA256 |
| 七档运行方法及报告 | nalinaly `e58ddc94d77c93a0a8a85ab2db4bd773da9dff84` 的 `csa_atomic_matrix_20260928`；运行源码仍为上一行 |
| PyPTO | nalinaly `3e87a843619aca13af39755700513d26b402e924` |
| Simpler | PyPTO 的原始 gitlink `a54c0509552b01e13fb0960e23ce409a01b5024f` |
| PTO-ISA | 官方 `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d` |
| PTOAS | 官方 0.66，CPython 3.12 aarch64 wheel |
| CANN / vLLM / Torch / torch_npu | 9.0.1 / 0.25.1 / 2.10.0+cpu / 2.10.0.post2 |
| Leaf 工作分支 | `dev/pypto-dsv4-csa-v0.25.1rc1-cann9.0.1`，本轮前 HEAD `ee5d9947b`；未修改其 kernel |

早期参考日志的 PyPTO5495749/Simpler166852bf 已过时，不能作为此性能版的依赖。
新建独立 reference-env，构建时启用 `PYPTO_BUILD_TORCH_NPU=ON`、四路并行及已有 ccache。
原有环境保留。预检记录 Python 来源、PyPTO/Simpler/Torch NPU 扩展路径与哈希、Git HEAD/dirty 状态、
ISA HEAD及运行库元数据、PTOAS 版本及 launcher 哈希，并检查 CANN 编译模块 可导入。

复用现有 vLLM-Ascend 原生扩展和 custom OPP，尚未证明其二进制与作者环境完全一致；
不得将当前环境称为逐字节相同的作者环境。

## 验收计划

1. 原版单层脚本：真实第2层权重、合成历史，B4/S6/H8192、performance、NZ2、atomic0、det0、graph。
   检查 Native/PTO 输出与 cache、保护区、A/B/A graph 重放及20次计时。单层通过不代表整模型通过。
2. 原版 prefill 重建真实模型的8K/128K离线 bank，TP4/DP4/EP16；每档四种输入。
   Native 和 PTO decode 使用同一 bank。已有 plan 和75分片检查通过，缓存尚未生成。
3. 原版七档 TP1/DP=EP16、DSpark 出5验6、EPLB关、full_decode_only：128K B4/8/16，8K B16/24/32/40。
   容量40，预算256/400，capture24/48/96/144/192/240；8步warmup、10步独立forward计时、另轮3步profile。
   原 `performance` 入口内部已分开计时与profile，不将profile耗时当正式速度。
4. 核对全部rank配置、位置、token/DSpark、均值/P95/最慢rank及数值精度，再逐模块迁移。
   token一致不替代逐层或共同历史下的数值误差验证。

## 已发生的检查与失败

| 检查/任务 | 结果 | 首因与处理 |
| --- | --- | --- |
| `task_20260928_172049_358065516214` | exit1，未执行算子 | 缺PTO_EAGER_ROOT；补独立原版目录约定 |
| 旧依赖完整CPU编译 | exit1 | assemble不支持pre_quant；改为精确参考工具链，不改kernel |
| 独立依赖构建 | exit0 | PyPTO/Simpler精确提交安装完成 |
| PTOAS启动检查 | 先失败后通过 | 通用tar要求Python3.11；改用官方cp312 wheel，显式使用现有Conda C++库 |
| 完整参考CPU编译 | exit0，COMPILE_PASS | 完整CSA lowering、工具链编译及load；无设备执行 |
| `task_20260928_175830_25582707293` | exit1 | 包装脚本覆盖PYTHONPATH遗漏CANN 编译模块；保留CANN路径并补预检 |
| `task_20260928_180432_275645018241` | exit1 | ASCEND_CACHE_PATH目录未创建；补mkdir |
| `task_20260928_180718_284496627321` | exit1 | 未开启可选Torch NPU扩展；原源码已有模块，补构建开关及预检 |
| `task_20260928_181150_304968913742` | exit1 | 构建进程误用旧editable启动模块，运行库ISA不匹配；修正路径优先级并重建 |
| `task_20260928_181941_208666318027` | exit0 | 原版单层graph/计时完成，具体数值见下节 |

以上启动失败属于环境准备错误，不能计为参考kernel精度失败或性能退化。
曾将未构建扩展误判为未公开文件，已纠正；没有因此新增替代实现。

本地证据目录：工作区 `reports/dsv4-nalinaly-full-repro-20260928/`。
包含原始参考、逐文件manifest、dependency-manifest、环境/构建/运行脚本、checklist及取回的preflight。
远端独立实验目录保留每次失败日志，结果目录不覆盖。后续结果继续追加，不替换失败记录。

## 原版单层预检结果

任务 `task_20260928_181941_208666318027`，原版脚本B4/S6/H8192、正式第2层权重、合成历史，
performance/NZ2/atomic0/det0，四张kernel根权重布局均为NZ。Native与PTO重复运行分别自洽；
graph A/B/A重放与各自eager输出/cache逐位一致，保护区检查通过。

| 指标 | 结果 |
| --- | ---: |
| Native graph中位耗时 | 560.580 μs |
| PTO graph中位耗时 | 568.140 μs |
| PTO相对Native耗时变化 | +1.349% |
| Native/PTO P95 | 576.800 / 581.680 μs |
| 融合层输出相对Native L2 | 0.19502785% |
| 输出不同元素 | 96,880 / 393,216 |
| 输出最大绝对误差 | 0.03125 |
| Top-K集合替换元素数 | 117（24行）；无越界/重复/缺失结构错误 |

输出形状为[24,4,4096]，包含HC post，不能与旧DSA-only输出[24,4096]的误差百分比直接比较。
原比较器的零容差FAIL仅表示有数值差异；进程exit0与graph通过不等于Native数值精度通过。
cache/state相对Native也有差异，全部逐项结果保存在report.json与states.pt；无NaN/Inf。
本档计时20次，compact metadata采用原脚本reuse策略；属于单层预检，不否定或确认七档整模型收益。

单卡原始报告已取回工作区 `single-env6/`；远端保留states.pt和全部原始日志、前后环境指纹。
18:23:08提交 `task_20260928_182308_268934625417`，按原脚本TP4/DP4/EP16依次生成8K/128K真实离线bank。
该任务最终 exit 1；此阶段不测CSA性能。详细原因及后续重测如下。

## 原生 prefill 初始化失败与配置恢复

任务 `task_20260928_182308_268934625417` 在19:06启动、约19:10失败，
日志已备份到本地 `prefill-failed/`。19:48才观察到终态，未做到连续每五分钟监控。
75分片已加载，但首次原生 dummy run 的 MoE hash 路由报 `input_ids=4, rows=2`，
部分TP rank为 `3, rows=2`；真实prefill未开始，两套bank只有plan/token文件，没有KV payload。

根因由源码及独立review确认：

- 历史 `e048502b1` 和 `a28328967` 父提交的离线脚本使用默认auto worker。
- 官方平台只在 `worker_cls == "auto"` 且未启用SP时，将 `all2all_backend` 设置为
  `flashinfer_all2allv`，以关闭框架层 `use_sequence_parallel_moe`。
- `a28328967`（9月26日）新增显式 `OfflineNPUWorker`，绕过该配置分支。
- TP4/DP4/EP16下，模型先拆hidden，MC2随后再次拆hidden；token ID没有第一次拆分。
  8行输入先拆成2行，hidden补6再拆得到2行；ID从8补6再拆得到[4,4,3,3]，吻合错误。
- 官方配套vLLM `752a3a504485790a2e8491cacbb35c137339ad34` 本身也默认
  `allgather_reducescatter`；不能将此错误归因于CSA kernel或仅凭版本字符串归因于依赖漂移。

新增独立 `run_prefill_native_config.py` 副本，仅prefill显式恢复
`all2all_backend="flashinfer_all2allv"`；原始文件、模型、kernel、decode脚本均未修改。
这应称为“恢复原始bank生成的有效配置”，不能称执行脚本完全未修改。
`prefill-native-config.patch` 和manifest保存唯一差异及原/新SHA256。
CPU门禁确认差异恰为该参数、decode kwargs为空、TP4/DP4下的
`use_sequence_parallel_moe` 从True恢复False；这尚不能替代真实NPU验证。

20:00:45提交 `task_20260928_200045_19802682608`，16卡、执行上限7200秒，
仍为原模型、TP4/DP4/EP16、原缓存布局，顺序8K/128K。
输出使用新目录 `results/prefill-native-config-h8192` 与 `...-h131072`，保留失败日志。
状态待核验；尚无完整参考性能结果，也没有将该配置修改用于Leaf生产代码。

## Bank验收与完整decode任务

`task_20260928_200045_19802682608` 已完成，exit0。8K与128K各4组输入、16份TP
manifest、每份191个tensor；原始audit全部PASS，各TP副本有效前缀无差异。
四个DP rank均生成1 token并保存请求结果；prefill耗时含IO，8K约15–18秒、128K约39–41秒，
不能用这些数值声称decode性能。
原始prefill日志及audit已取回本地，远端bank保留完整payload。生产kernel没有改动。

20:12:47提交 `task_20260928_201247_22649656531`，16卡max7200秒，
原始七档矩阵：H131072/B4,8,16和H8192/B16,24,32,40；PTO、Native分开运行，
NZ2/performance/atomic0/det0、DSpark5、EPLB关闭、FULL_DECODE_ONLY。
每档先独立计时再独立profile，不用profile扰动的耗时作主结果；结果尚待核验。
仍使用冻结71153原始decode脚本，prefill配置恢复不传播到decode。

## 七档完整参考 decode 结果

任务 `task_20260928_201247_22649656531` 于20:58查询为 completed(exit=0)，七档均通过原始收集器。
这是冻结 nalinaly `71153bb3` 完整实现的复现，尚不是Leaf模块迁移结果。
每侧每档取预热8步后连续10步，16 rank等权均值；无profiler的 `_model_forward` 设备事件计时。
不含metadata、logits、采样、草稿、步间等待，因此不能称端到端吞吐提高相同比例。

| 上下文 | 单卡BS | Native均值(ms) | PTO均值(ms) | 耗时降低 |
| --- | ---: | ---: | ---: | ---: |
| 128K | 4 | 47.134 | 45.380 | 3.72% |
| 128K | 8 | 56.581 | 55.349 | 2.18% |
| 128K | 16 | 72.278 | 70.647 | 2.26% |
| 8K | 16 | 65.126 | 61.207 | 6.02% |
| 8K | 24 | 79.490 | 75.919 | 4.49% |
| 8K | 32 | 90.802 | 88.239 | 2.82% |
| 8K | 40 | 103.653 | 101.734 | 1.85% |

七档配置/位置门禁无错误，输出token差异0、DSpark统计差异0。两轮共比较573,440个输出token。
全部16rank的21个C4层都确认捕获对应PTO token档位，不能用空hook记录冒充命中。
当前结果证明该组固定bank输入下的参考整网forward收益；未提供全模型中间tensor逐位一致证据。
单层融合输出0.195%的误差仍保留，不因token一致而抹掉。完整图模式的数值hook验收仍待完成。

收集器依赖补充：`collect_model_dependency.py` 来自e58报告提交，其新增
`compare_forward_setup` 来自同提交的纯CPU `performance.py`。首次混用71153依赖导致ImportError；
现将e58原始CPU依赖单独归档并显式加载，未修改运行代码或放宽验收条件。
原报告run_model.sh明确从71153归档执行，因此运行源继续冻结71153；其原始事件准备为lazy_per_forward。

完整rank JSON与日志已备份到本地 `matrix/`，原始trace保留227；rank0七档每层离线解析进行中。
七档不是独立重复运行的置信区间；每档160个样本来自同次执行的16rank×10步，具有相关性。
原参考CANN9.0.0/Python3.10与本次目标CANN9.0.1/Python3.12仍有环境差异，具体增益不要求完全相等。
复现结果纠正此前单层无收益便推断整模型无收益的错误；后续应迁移完整融合与调度组合。

## 独立 trace 的 CSA 完整层区间

CPU离线导出与原始分析器均exit0，14份rank0设备任务压缩文件及分析结果已备份本地。
下表是rank0独立3步、21个C4层共63个区间的平均值；包含HC pre到HC post，
PTO保留原收集器定义的前置metadata。与前一表的16rank×10步整网forward不是同一轮采样。

| 上下文 | 单卡BS | Native层区间(μs) | PTO层区间(μs) | 区间缩短 |
| --- | ---: | ---: | ---: | ---: |
| 128K | 4 | 893.69 | 706.69 | 20.92% |
| 128K | 8 | 1005.13 | 882.30 | 12.22% |
| 128K | 16 | 1291.68 | 1120.06 | 13.29% |
| 8K | 16 | 975.75 | 821.47 | 15.81% |
| 8K | 24 | 1175.79 | 977.51 | 16.86% |
| 8K | 32 | 1309.64 | 1152.20 | 12.02% |
| 8K | 40 | 1434.22 | 1327.45 | 7.44% |

FFN及其他attention区间也随执行变化；不能把整网差值全部归因于CSA，
不能将并发task busy时间简单相加当成关键路径。原始分析结果保留43层×3步的FFN边界及任务拆分。
所有新增NPU任务均已终止，未创建定时任务。七档期间持续在当前对话检查并汇报状态。
后续未完成项：真实模型共同输入下的中间tensor精度、Leaf完整模块迁移、迁移后同口径七档验证。

独立审查确认224份rank记录完整（7档×2实现×16rank），上述统计与捕获门禁成立。
特别保留参考运行的实际CANN event mode差异：Native=0、PTO=1。
因此这是完整参考配置对照，不能把差值全部归因于单个kernel；“耗时下降”也不等于相同比例的吞吐提升。
CPU位置及输出token/DSpark相同，仍不能证明device draft token、完整logits或cache逐位相同。

## 21:46 完整参考的全 C4 层共同输入精度诊断

本次仍为 nalinaly `71153bb3` 完整 performance 实现，不是 Leaf 生产 kernel 的新版本。
沿用上述精确工具链及已审计的离线 bank，B4/S6、TP1/DP=EP16、NZ2、atomic0、
deterministic0、EPLB关、eager；分别测8K与128K，生成16 token。
独立测试扩展覆盖21个C4层，每层每rank采前2次24-token调用。
原参考4124个受版本控制文件未修改；扩展只添加诊断入口与observer。
CPU状态恢复及指标测试2项通过，独立审查后提交。任务exit0，前后环境校验通过并释放全部设备。

每次先保存输入和声明写入页，运行PTO，恢复初态，再运行Native；保留Native的输出和cache继续。
超过采样预算或不匹配shape时同样运行Native，避免后续层输入混入先前PTO误差。
捕获五组cache/state物理页（indexer carrier包含key和scale）及三组scratch/output；
诊断内补齐performance版本的indexer参数别名，不改变生产接口。

| 上下文 | 配对数 | 融合输出relative L2最小值 | 中位数 | 最大值 | 逐位相同输出 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 672 | 0.001669% | 0.183727% | 0.949103% | 0/672 |
| 128K | 672 | 0.001694% | 0.236263% | 0.988056% | 0/672 |

每组672=16rank×21层×2样本，所有rank覆盖充分，输出均有限，未发现新增非有限cache项。
状态`MEASURED`表示数据采集充分，不表示数值精度通过。两组最大输出误差均在第18层第二次采样。
这些是`[24,4,4096]`、HC pre到HC post的完整融合输出指标；不是DSA-only输出，
不能直接与旧单层attention指标或其他提交的BSH结果比较。

| cache/state物理页 | 8K最大relative L2 | 128K最大relative L2 |
| --- | ---: | ---: |
| KV | 0.336264% | 0.330800% |
| compressed KV | 0.133164% | 0.117804% |
| compressor state | 0.025453% | 0.031410% |
| indexer compressor state | 0.024325% | 0.024273% |

五组cache/state在每个配对中均存在差异；packed INT8 indexer页只统计字节差异，
不能用整数L2冒充其中FP16 scale的数值精度。页指标包含触及页中的未改写部分，
不是逐有效slot误差，也不构成未声明页面的越界写验证。
BF16 ULP指标也有明显大于几个ULP的项，不能将本次结果概括为“只差几个bit”。

结论：完整参考配置的七档性能收益已复现，但真实模型共同输入下并非逐位精度一致。
此前输出token相同不等于tensor相同。本次不测性能，不证明独立PTO轨迹的累积误差、
完整logits、graph精度或所有形状。尚未保存Q/QR/heads中间值，不能据此归因到具体子算子。
后续迁移必须同时保留同口径的Native精度检查和性能比较，不将参考误差当作自动放宽门槛的理由。

本地证据目录`reports/dsv4-reference-all-layer-precision-20260928`保存测试扩展、源码身份、
32份rank结果、日志及rank0的84份配对张量；服务器原件保留。Leaf生产实现尚未迁移。

## 22:20 第一阶段迁移：完整 BSH 计算集合

在Leaf `e4fea8891`基线上增加`attention/pto_kernels/dspark_layer`，包含16个计算依赖模块，
来自冻结参考`71153bb3`的performance实现及其共享HC/Q投影/规约/NZ模块。
`SOURCE.json`保存每个原始文件SHA256与来源。除了包内导入重定向、导入排序与格式外，
计算AST一致，相对导入闭合；共享config对被引用常量的值不变。独立审查无静态阻断。
新增中央env `VLLM_ASCEND_PTO_CSA_ATOMIC_ADD`，默认1与参考相同；本次显式0。
内部完整kernel共56个tensor参数；本阶段尚未修改模型forward、旧DSA派发或生产初始化。

同一设备顺序运行原参考和迁入kernel，保留参考host权重/metadata/cache绑定及原始单层测试。
B4/S6、H8192、真实第2层权重、seed1024、合成hidden/history、NZ2/atomic0、deterministic0，
graph计时各20次、预热5次，metadata复用口径。环境同前述固定参考工具链。

| 指标 | 原版完整kernel | 迁入完整kernel |
| --- | ---: | ---: |
| 对应Native graph p50(μs) | 578.70 | 541.61 |
| PTO graph p50(μs) | 570.74 | 564.80 |
| 相对Native融合输出relative L2 | 0.195028% | 0.195028% |
| 相对Native输出不同元素 | 96,880/393,216 | 96,880/393,216 |
| graph A/B/A及保护区检查 | 通过 | 通过 |

额外直接加载两份`states.pt`逐字节对比：两轮Native的8组状态完全一致，
两轮PTO的8组状态也完全一致，包括最终输出、TopK及六个cache/state视图。
这支持本用例中完整计算集合迁入未改变结果；不是说PTO与Native逐位一致。
PTO p50相差约1.04%，但Native两轮本身变化约6.41%，不能认定为新的优化收益。
本次没有修改算术或调度策略，不把单层计时替代此前整模型七档性能。

首轮候选在运行前因诊断脚本过早导入PyPTO，被原始环境来源检查拒绝；
修复为先执行原activate，再加载候选，没有绕过检查。
第二次提交因等待设备默认600秒超时被调度器取消，无候选执行。
第三次改为提交后轮询，最终exit0。原失败日志与成功原始张量均已备份。
仓库新增`tests/pto_attn/run_full_layer_reference.py`提供可复用入口，
`test_full_layer_reference_harness.py`两项CPU回归通过，验证环境拒绝时不加载候选及正确加载时序。

当前范围：完整计算模块已迁入并通过上述迁移对拍，**Leaf生产模型入口尚未切换**。
后续必须接入原生attention半层、加载后初始化、graph准入与真正Native回退，
然后重新完成全层数值及同口径七档性能验证。不能将本阶段写成完整服务迁移已完成。

## 23:26 第二阶段：Leaf生产入口单层验收

基线`561492fa5`上的完整BSH接入修改。保留原生DecoderLayer签名、返回tuple及FFN，
由原生runner加载权重后初始化，C4 attention半层通过custom op调用完整融合kernel。
层对象持有权重和工作区；原生builder提供metadata、原生owner持有cache；
内部函数绑定56个tensor参数，无独立CSAServiceRuntime/NativeCSACall类链。
仍有必要的参数整理、compact metadata生成及零拷贝存储描述符，不能说“完全没有适配工作”。

环境继续固定CANN9.0.1、vLLM0.25.1、torch2.10.0、torch_npu2.10.0.post2、
PyPTO`3e87a843`、Simpler`a54c0509`、PTO-ISA`327cd586`、PTOAS0.66。
生产原生扩展复用既有编译产物，未覆盖现有环境。2980个源码文件SHA和工具链前后检查通过。

必要兼容处理：与参考一致保留Compressor wkv/wgate的ND权重；
CANN缺少TransposeBatchMatMulWeightNz时wo_a也保留ND。原生DSA工厂不再返回旧CSA实现，
保证新融合的回退真正执行Native。缓存规格不满足会显式报错，不承诺自动回退。

### 任务及结果

| 任务 | 状态 | 说明 |
|---|---|---|
| `task_20260928_231059_287902114524` | exit1 | 新测试驱动漏调用enable_custom_op；metadata构建失败，未执行CSA。保留失败日志。 |
| `task_20260928_232409_335427211168` | exit0 | 补回原版相同初始化后完成真实Leaf单层入口测试。 |

配置B4/S6/H8192、真实第2层权重、seed1024、合成hidden/history、NZ2、atomic0、deterministic0。
使用Leaf模型、metadata builder、缓存布局与真实custom op；参考仅提供fixture及计量助手。
输出范围是HC pre→norm→CSA→HC post的`[24,4,4096]`，不含FFN。

| 指标 | 结果 |
|---|---|
| 与冻结参考的Native八组状态逐字节比较 | 全部0差异 |
| 与冻结参考的PTO八组状态逐字节比较 | 全部0差异 |
| PTO相对Native输出relative L2 | 0.19502785% |
| 输出不同元素 | 96880 / 393216 |
| 输出最大绝对误差 | 0.03125 |
| graph A/B/A replay | PASS |
| metadata及cache写入边界 | PASS，非法slot外写入0 |
| 强制Native回退 | 八组结果与Native一致，未再次进入CSA |
| Native graph p50 | 575.70 μs |
| Leaf graph p50 | 555.61 μs |
| 本轮中位耗时下降 | 3.49% |

计时20次、warmup5次，两侧capture均包含metadata生成；不包含逐步Python host适配开销。
不能与上一阶段metadata reuse计时直接相减。只有一次运行，无独立重复置信区间。
八组状态为输出、TopK、SWA、compressed KV、state、indexer key、indexer scale、indexer state。
相对参考逐位一致不代表相对Native逐位一致，精度差异未消除。

CPU接口/存储/graph门禁及测试驱动回归14项通过。另在227真实依赖环境运行后端工厂CPU回归，2项通过，确认开关开/关均保持Native DSA。
强制eligible=False只证明回退代码；不证明真实非均匀请求、DP padding或调度图选择正确。
本次也未执行完整DecoderLayer/FFN、整模型21层接入、整模型共同历史数值或七档性能。
这些仍为下一阶段必要门禁，当前不能宣称完整生产迁移验收完成。

证据：`reports/dsv4-full-layer-integration-20260928/`保存两轮日志、
`source-manifest-single-v1.json`、`source-manifest.json`、结果`states.pt`、
`cross-reference-bitcompare.json`及逐文件`evidence-manifest.json`。
复现入口为`tests/pto_attn/run_full_layer_native.py`，需提供冻结参考的测试助手路径、完整checkpoint和新输出目录。

## 23:52 第三阶段：Leaf完整模型共同历史精度

生产版本`393134f1d13b0319c85c8ef7165b7c8ad25c40e5`。
任务`task_20260928_234136_369477911503`正常结束，exit0；8K和128K的前后源码/依赖检查均通过。
完整75分片DSV4模型，TP1/DP=EP16、B4/S6、EPLB关闭、eager、每请求生成16token。
模型包含43层，逐一检查21个C4层；每个rank每层采2次，共每档672组比较。
复用冻结参考已审计通过的Native离线prefill bank、同一工具链和采样口径。
每个worker记录并断言Leaf、pto_layer、DSA模块实际路径，确认没有误入参考生产源码。

测试钩子拦截各层实际`_pto_csa_operator`：保存slot声明的写页与scratch/output，执行PTO，
恢复所保存初态，再执行Native；样本预算外也保留Native计算路径。
因此后续层的输入历史由Native产生，不是独立CSA生成轨迹。
这里仅恢复声明写页；脚本本身不检查页外写入，不可用它单独证明所有cache均完整恢复。
该前提由此前单层guard提供限定范围证据。整模型计时不使用这些双执行钩子。

| 历史长度 | rank数 / 共同输入对数 | 输出L2中位数 | 输出L2最大值 | 相对Native输出逐位一致 |
|---|---|---|---|---|
| 8K | 16 / 672 | 0.183727% | 0.949103% | 0 / 672 |
| 128K | 16 / 672 | 0.236263% | 0.988056% | 0 / 672 |

两档所有rank的输出、cache指标和positions字段，与冻结参考此前同口径结果完全相同。
原始tensor另作逐字节核对：每档rank0的42个文件、1008对tensor，均为0字节差异；
两档合计84文件、2016对tensor。覆盖hidden、positions、初始页、Native/PTO输出、
Native/PTO写后页及page ids。原始tensor只保存rank0，其他rank不能从指标相同推断逐位一致。

结论：本次真实模型共同输入对照支持Leaf完整BSH迁移复现了参考的数值行为；
仍不能说它相对Native逐位精确，也不覆盖独立CSA轨迹的logits、graph整模型数值或全部请求形状。
所有输出有限，未出现新增的非有限cache值；`MEASURED`表示采集完整有效，不是精度达标阈值。

新测试钩子CPU回归2项通过：21层cache恢复、预算结束后的Native保留、原方法恢复、
特殊数值/ULP指标。独立审阅未发现当前bitcompare/performance入口阻断。
其他参考诊断入口尚未移植，脚本显式拒绝使用，不能声称已测试padding-capture等路径。

证据目录：`reports/dsv4-full-layer-integration-20260928/`的
`results/full-precision-v1/`、`full-precision-summary.json`、
`full-precision-h8192-cross-reference.json`、`full-precision-h131072-cross-reference.json`、
`full-precision-evidence-manifest.json`（186文件哈希）。源码为本节固定提交，
测试harness全部14个Python文件另有manifest；原始结果已从227备份本地。

下一阶段已提交同口径七档FULL_DECODE_ONLY性能矩阵；结果待任务结束后另记，
不使用上面eager双执行过程的耗时作为性能数据。

## 2026-09-29 00:20 第四阶段：Leaf七档完整模型graph性能

任务`task_20260928_235422_38979742701`结束，exit0。运行源码仍固定`393134f1d`，
后续文档提交不改变该快照。Native为同一Leaf源码关闭CSA；PTO开启完整BSH融合。
CANN9.0.1/vLLM0.25.1、完整DSV4权重、TP1/DP=EP16、DSpark出5验6、EPLB关闭、
NZ2/atomic0/deterministic0与参考一致。使用同一离线prefill bank和FULL_DECODE_ONLY。

捕获档位24/48/96/144/192/240，容量max_num_seqs40；128K token预算256，8K为400。
每档先预热一轮96token，再生成128token；各rank跳过8步后连续采10步设备`_model_forward`。
请求以pause→enqueue→DP barrier→resume统一入场。独立3步Level0 trace不混入此计时。

| 历史 | 单卡BS | 本轮Native均值(ms) | 本轮Leaf均值(ms) | 本轮耗时下降 | 此前参考复现下降 |
|---|---|---|---|---|---|
| 128K | 4 | 46.955 | 45.255 | 3.62% | 3.72% |
| 128K | 8 | 55.831 | 54.880 | 1.70% | 2.18% |
| 128K | 16 | 73.626 | 71.380 | 3.05% | 2.26% |
| 8K | 16 | 64.598 | 61.409 | 4.94% | 6.02% |
| 8K | 24 | 79.338 | 76.063 | 4.13% | 4.49% |
| 8K | 32 | 90.705 | 88.306 | 2.64% | 2.82% |
| 8K | 40 | 103.561 | 101.838 | 1.66% | 1.85% |

七档均通过完整收集门禁：224份rank记录（7档×2实现×16rank），每档对应的21个C4层均实际捕获CSA；
两侧采样CPU请求位置、配置、输出token和DSpark统计相同，共核对573440个输出token位置。
原收集器要求层键`model.layers.i.self_attn.attn`，Leaf观测记录`model.layers.i.self_attn`；
仅在独立收集器副本改这一处字面量，仍严格要求固定21层集合及每层对应`pto_tokens`命中。
未修改原始结果或放宽其他检查，修正前拒收日志与单行patch均保留；独立审查确认门禁强度保留。

这组结果支持完整迁移后复现了七档正收益；不是统计意义上的等速或稳定性证明。
每档只有一次独立运行，160个rank/step样本存在相关性；参考列来自另一次运行，
不能把两列收益差直接解释为迁移损失。forward不含draft、logits、采样、步间等待，
也不是吞吐提升百分比。输出token一致不代替graph整模型的数值精度验收。

测试脚本：`run_matrix_v1.sh`，结果`results/matrix-v1/forward.json`与`RESULTS.md`，
收集器修改记录`collector-layer-key.patch`。生产源码、测试harness及依赖都按各自manifest固定。
16卡任务结束后才开始CPU trace解析，避免在正式计时期间额外并行解析影响测量。

### 独立rank0层内trace

每档Native/PTO均为63个C4区间（21层×3步），CPU导出和解析正常结束。
该采集独立于正式forward计时；区间包含HCpre到HCpost，不是单个CSA kernel时间。

| 历史 | 单卡BS | Native区间均值(us) | Leaf区间均值(us) | 耗时下降 |
|---|---|---|---|---|
| 131072 | 4 | 895.61 | 703.89 | 21.41% |
| 131072 | 8 | 1019.04 | 874.44 | 14.19% |
| 131072 | 16 | 1288.41 | 1122.67 | 12.86% |
| 8192 | 16 | 980.17 | 806.71 | 17.70% |
| 8192 | 24 | 1176.83 | 967.52 | 17.79% |
| 8192 | 32 | 1309.27 | 1144.50 | 12.58% |
| 8192 | 40 | 1449.43 | 1315.02 | 9.27% |

原始rank0设备事件已按14组压缩保存为`device_tasks/rank0.json.gz`，随结果备份本地。
`model_gap_rank0.json`保留区间、分阶段耗时及分析限制；完整原始profiler目录保留227。
逐文件哈希见`matrix-evidence-manifest.json`。该trace不能用来声明EPLB开启或独立CSA轨迹数值通过。
