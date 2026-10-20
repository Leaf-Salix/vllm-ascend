# TND精度版：完整SPMD基线、流水测量与独立工具链（2026-10-20）

## 本轮结论

分支为 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，精度控制版本固定
`48c58ce7c467dfce63d5cb5c41afd225488d6012`。本轮没有修改77份CSA运行源、切换默认路由或取得新的优化收益。

当前精度候选是“前缀MIX加多任务后缀”，参考最新实现是完整24组常驻MIX，双方均能表达真实TND。
独立SDK已完成PyPTO/Simpler原生构建、7项CPU接口测试、完整130Tensor动态入口及最小信号MIX的CPU lowering。
LLVM源码构建已完成；实际PTOAS复用自有0.66 wheel并通过完整离线编译。最小信号设备门禁已通过；CSA的8K/T18设备回归也已通过，不能据此宣称已运行完整CSA SPMD或追平参考性能。

## 测量配置与范围

本轮单层测试：真实DeepSeek-V4-Flash-0731-w8a8第4层权重，合成hidden/history，TP1、B16、S6、history8191、NZ2。
覆盖HC-pre、输入Norm、CSA、O-proj、HC-post，原生metadata/cache生命周期保留。
确定性level1、HCCL确定性开启，实际ACLGraph/FULL、raw replay、DSA多流核对。
这不是DP16整模型吞吐测试，不能与整模型forward毫秒相混。

正式计时为同卡12个交替顺序窗口，每窗100回放，不删除离群值；swimlane、依赖图、PMU和CANN trace各自独立采集。
Native/PTO使用相同输入与独立初始化cache，八边界、A/B/A、只读检查及guard均通过，PTO回退被禁止。

| 实现 | 中位延迟 μs | 数值/口径 |
| --- | ---: | --- |
| 本轮Native SK0 | 940.8265 | 本环境控制；已实际启用DSA多流 |
| 冻结TND精度候选 | 1333.7268 | 八边界对Native逐位一致，比本轮SK0慢41.76% |
| 用户截图nalinaly动态SPMD | 644.08 | 用户提供参考值；尚未同环境受控复现 |

独立三回放CANN trace的完整跨度分别为924.898/1352.887 μs，不取代正式窗口中位数。
原始SK0/DFX记录的 `cross_bucket_scope` 误留“SK1 comparator”字样；实际options和执行为SK0，本报告据实纠正，原始记录保留。

## 用户提供的性能目标

截图描述：10月20日、S6、NZ2、真实第4层权重、合成hidden/KV；Native为SK1、静态编译、ACLGraph、DSA多流。
多任务外层为全部设备流的完整跨度；L1/SPMD取最早incore开始至最晚结束，包含内部等待。截图精确源提交尚未核实。

| 配置 | Native | 多任务外层 | 多任务L1 | 静态SPMD | 动态SPMD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 791.25 | 626.50 | 492.78 | 467.58 | 476.40 |
| 128K/B8 | 887.62 | 711.88 | 629.54 | 597.02 | 626.50 |
| 128K/B16 | 1177.88 | 950.50 | 873.16 | 778.94 | 802.20 |
| 128K/B24 | 1342.25 | 1197.75 | 1119.08 | 1095.70 | 1039.82 |
| 128K/B28 | 1704.38 | 1346.00 | 1280.12 | — | 1167.72 |
| 8K/B16 | 803.25 | 705.75 | 691.42 | 640.82 | 644.08 |
| 8K/B32 | 1107.00 | 1038.88 | 986.30 | 931.66 | 933.58 |

单位均μs。截图提升为 `baseline/dynamic-1`，与 `1-dynamic/baseline` 的耗时下降不同。
不能将截图644.08与本轮1333.73直接解释成某项改动的因果收益。

## DFX与PMU证据

DFX L1三窗各1219记录，AIC353/AIV866，72核均有记录。通过独立依赖图把task token映射到函数；registration ID不当作func ID。

| 项目 | 三窗中位跨度 μs | 含义 |
| --- | ---: | --- |
| 完整incore | 1244.14 | 有插桩，不是正式无插桩延迟 |
| QK/PV AIV | 255.72 | 含内部等待，与AIC重叠 |
| QK/PV AIC | 253.90 | 不能与AIV相加 |
| 前缀Vector/Cube | 177.22 / 174.66 | 两角色重叠 |
| O-proj整阶段 | 256.36 | 包括分组和调度间隙 |
| 无incore记录活跃空隙 | 110.12 | 不能等同纯AICPU开销 |

Sparse分组约755 μs包含很早启动的准备任务和后续间隔，不能称为attention纯计算。
各函数start/end/span独立取中位数，也不能拼成一条真实时间线。

外层CANN Level1 PipeUtilization每arm3回放，开启PMU而不开DFX/dep、不采正式窗口。
PTO executor的中位计数如下；各项可能重叠，不是可相加的耗时占比。

| 管线 | AIC | AIV |
| --- | ---: | ---: |
| MAC/Vector | 8.8% | 12.1% |
| Scalar | 37.1% | 50.1% |
| MTE2 | 28.1% | 8.2% |

CANN只看见整个Simpler executor，不能据此把MTE2等计数分配给内部QK/PV或O-proj。
Profiler有host-flow关联告警，非空device CSV及原日志保留。新Simpler明确拒绝kernel-mode enable_pmu，没有绕过检查。

## Native SK1资格与数值差异

旧13包明确为SK0，未冒充SK1。本轮39项官方SK1编译在自有目录完成。
私有custom OPP仅将CompressorMetadata只读Init指针加const；原安装、权限和数学未改。
官方打包要求OPP根可写，因此使用自有OPP副本，不改变原目录权限。

固定缓存runner限制B16/S6、history8191、L4，完整包/输入集合及SHA验证；仍调用官方安装和实际reselect，缓存缺失硬拒绝设备任务内重编。
新包实际安装194项静态文件、reselect一次、FULL/raw和guards通过，但八边界检查发现SK1/SK0不逐位一致，在计时前停止。
缓存安装/reselect不等于已证明每个算子均使用SK1，尚无本轮SK1性能结果。

另一次短任务导出SK0，八边界、hidden/position、初始cache哈希均复现历史金值。与已保存SK1比较：

| 边界 | 不同元素 | 最大绝对差 | relative L2 |
| --- | ---: | ---: | ---: |
| raw KV | 13 | 0.001953125 | 0.00000293175 |
| TopK索引 | 375 | 不适合作数值精度指标 | 不适合作数值精度指标 |
| 最终输出 | 11755 | 0.015625 | 0.000286178 |
| 压缩KV、主state、Indexer两组cache、Indexer state | 0 | 0 | 0 |

这些是同输入下两档Native的外部边界差异，尚未定位最早内部算子，不能认定Native缺陷或PTO精度回退。
本轮没有修改已验收的SK0精度金值，也没有把SK1静态包授予PTO的旧资格。
weight_details是相同checkpoint/加载元数据，不能表述为权重SHA证明。

## 独立工具链状态

用户批准后，所有新源码、venv、缓存和二进制均在自有 `csa-spmd-sdk-20261020`，原环境保留。
17899不可用，使用本机取得固定源码再rsync，没有使用共享代理或重编vLLM。

| 组件 | 固定提交 | 本轮状态 |
| --- | --- | --- |
| PyPTO | `2dfa7f120ba5e577566046d920845aa241bfa7bb` | 新core和Torch adapter原生构建；CPU门禁通过 |
| Simpler | `57e499238fd4ee3548947ad860be456aaeeb16b5` | 新绑定及a2a3/tensormap runtime构建；最小信号与冻结CSA T18设备回归通过 |
| PTOAS参考源 | `e5c1c432af8249bac2f905c7648ebafefec3c20a` | 配置完成；源码构建由现成wheel复用替代，未完成 |
| 实际PTOAS wheel | `0.66`，SHA见下文 | 隔离venv安装，四项完整离线编译通过；不冒认参考源提交 |
| LLVM/MLIR | `fa2fd1f75d742fec91ecf40d98c9c7961647ec42` | 两路源码构建完成并保留；当前wheel不依赖此树 |
| PTO-ISA | `327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d` | managed checkout与构建metadata一致 |
| CANN cmake | `a7ef9a7f04ea9efc5ad388cdb8cce0b0c1759c70` | 固定源码已准备 |

沿用Python3.12.14、Torch2.10.0+cpu、Torch-NPU2.10.0.post2、vLLM0.25.1、CANN9.2beta2。
GCC10.3.1、cmake4.4.3、ninja1.13.2与参考Python3.10/GCC15不同，未声称二进制环境相同。
新venv只读继承旧依赖，包绑定显式检查实际core/adapter/_task_interface路径，避免继承的editable finder抢占。

7项CPU接口/ABI检查通过；根上限256Tensor/128scalar，核内上限256Tensor/16scalar。
参考完整130Tensor动态根生成一个MIX，sync_start及signal操作存在，明确skip_ptoas=True。
额外130参数Torch注册实测被本PyTorch的64参数限制拒绝，保留失败记录。宽入口应按参考实现走直接JIT prepare/enqueue，并保留外层小schema；不能把native256参数上限当作Torch注册也支持256。
最小信号门禁使用24组MIX、不同owner数据、代际、毒化和尾部guard。132参数根经直接JIT路径完成PTOAS、CCEC及编排二进制组装，未初始化NPU；设备门禁已通过，CSA另有下文T18独立回归；不外推到整模型。
原始失败和修正过程保留，没有降低验证等级或改参考数学来绕过编译错误。

## 新工具链的完整编译结果

实际PTOAS来源为自有已下载wheel：`ptoas-0.66-cp312-cp312-manylinux_2_34_aarch64.whl`，
SHA256为`d121b5b5db9a23a98eb3c0acc74d0abaef8aaa90e4df0385e79feb67636ef997`。
在新独立`ptoas-wheel-env`离线安装，PTOAS_ROOT指向独立`toolchain-wheel`。
该wheel的Git提交没有得到证明，不能因版本同为0.66便等同e5c1c432参考源。
未修改原安装。PTOAS重复源码构建已在核验自身PID/cwd后SIGINT停止，全部已完成对象、ccache及LLVM保留。

| 无设备门禁 | 参数 | 结果 |
| --- | ---: | --- |
| 跨AIC/AIV信号MIX | 132 Tensor | 完整PTOAS、CCEC、kernel ABI编排二进制通过，NPU未初始化 |
| 冻结精度CSA完整入口 | 64 Tensor | 源码SHA一致，完整PTOAS、CCEC、kernel ABI编排二进制通过 |
| O-proj/HC-post常驻候选v8 | 16 Tensor | 完整PTOAS、CCEC和编排二进制通过，尚无设备精度结论 |
| nalinaly完整动态SPMD | 130 Tensor | 固定beaefa6f源、NZ2/atomic0，完整PTOAS、CCEC和编排二进制通过；未执行设备 |

常驻后缀只在独立诊断目录，未接入服务：24/48 worker领取同样逻辑任务，保留原Native K顺序、全8192列统一量化、BF16落点和HC-post mul/add顺序。
HC-post helper按实参展开避免动态Tensor元数据丢失；residual/output以同storage的`[T,16384]`描述符在host传入，实际设备harness须验证pointer/stride/storage_offset。
六处workspace写回保持同址原位副作用，生成IR确认没有丢失；唯一v4_wide_packet的pipeline改range以消除新SDK内存规划的L0C分支冲突，数学迭代次序不变。
生成实际PTO确认6处GM写回（4×accum、2×integer_accum），单MIX、sync_start，零额外GM分配。
最高片上地址为L1 256KiB、L0A/B/C各64KiB、AIV UB128KiB；容量检查和CCEC通过不证明设备数值/无死锁。

## 新SDK最小设备门禁

`task_20261020_162848_528985179`完成、exit0。验证132个实参走直接JIT，24 AIC/48 AIV的FIX写回与跨owner读取，实际FULL Graph、31次回放、A/B/A更新和尾部guard均通过。
代际实际为`[1,1,2,12,22,32]`：capture不假设会执行，首次显式replay推进1，后续每组推进10。
该任务排队约23分钟后执行；排队时间不是占卡时间。该结果仅授予此通信门禁，不自动授予完整CSA或常驻后缀的精度资格。
新SDK冻结CSA回归`task_20261020_165238_183401224185`已完成、exit0：8K、真实请求长度`[6,5,4,3]`共18行，实际FULL/raw、不补每请求S6、Native确定性开启。八个外部边界、A/B/A、只读检查、poison和保护区全部通过；没有采性能。该结果是单层真实权重、合成hidden/history的SDK回归，不能替代新SDK整模型8K/128K验收。

## 后续验收顺序

1. 保留实际wheel及完整编译SHA记录，完成设备同步门禁，不混淆参考源与wheel身份。
2. 最小信号MIX完整编译、审查实际FIX/缓存发布，再执行短设备FULL/代际门禁。
3. 已完成新SDK冻结精度候选的8K/T18八边界逐位回归；迁移resident仍须分别验证新调度。
4. 先O-proj/HC-post小后缀，逐步迁移显式logical-worker数学体；保留Native K顺序、BF16落点、QK K512与整行8192量化。
5. Scalar缓存可见性、FFTS READY/ACK、实际核内ABI与L1容量单独验证；不能机械嵌套旧SPMD。
6. 完整CSA有稳定收益后，先8K/B40再128K/B24验证192token及接受事件仍与历史Native一致。

每项NPU任务内750秒、外780秒；首差停止定位、不扩大配置、不跑整模型Native2。
当前没有新的性能版可替代精度基准，也没有完整resident的整模型验收结论。本轮提交的只是报告与实验材料；正式运行源码和默认路由不变。

## 证据与复现

本轮[证据索引](evidence_tnd_precision/spmd_sdk_baseline_20261020/README.md)保存原始结果、失败迭代和独立复核。
逐项[复现方法](evidence_tnd_precision/spmd_sdk_baseline_20261020/REPRODUCTION.md)及原始记录包中的`RUNS.json`区分CPU完整编译与设备任务。
