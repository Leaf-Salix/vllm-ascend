# DeepSeek V4 Flash PyPTO CSA 测试历史

本记录属于 `dev/pypto-dsv4-csa-tnd-main-20260924` 分支，覆盖从首次接入
`6c9d552a1` 到作用域修复 `9243a5f5e` 的提交。更新日期：2026-09-24。
后续改变 kernel、绑定、原生基线或测量方法时，在本文追加记录，保留旧结果，
以便按提交比较。这里的“通过运行”仅指程序完成；数值验收单独标注。

## 测试边界和比较规则

- 227 单卡测试使用 DeepSeek-V4-Flash-0731-w8a8 的真实第 2 层 C4 attention
  权重，固定 seed=62 构造 hidden states 与历史 cache/state。Native 与 CSA 使用
  独立但初值相同的缓存。它不是整模型生成，也没有测 TP/DP/EP=16、EPLB、
  DSpark 接收率或 GBS=16×4 吞吐。
- Native 精度基线开启 `torch_npu.npu.set_deterministic_level(1)` 和
  `HCCL_DETERMINISTIC=true`。`relative L2 = ||CSA - Native||₂ / ||Native||₂`；
  逐元素验收为 `torch.allclose(rtol=1e-2, atol=1e-2)`。记录输出及实际写入的
  cache/state，不能用整块未写历史缓存稀释误差。
- Graph 延迟是固定地址 ACLGraph replay 的单层 attention forward 时间，不含
  编译、权重准备、capture、scheduler 或模型其他层。正数“CSA 慢”表示
  `(CSA / Native - 1) × 100% > 0`。只有相同权重、输入、软件、设备、测量循环
  和精度语义的同组 Native/CSA 数字可直接相除；跨提交数字只用于定位趋势。
- 除特别注明，TND 测试环境为 CANN `9.2.0-beta.2`、Torch `2.10.0`、
  Torch-NPU `2.10.0.post4`、vLLM `0.29.0`、PyPTO
  `54957491ede07ad5d5015f5e69874f367113cf45`（包含 PR #2867）、Simpler
  `32dff953d07f6bd2aacab8532860f28aca6df931`。TP=1、128 槽 cache 页。
  测试使用已存在的独立环境；这些结果不证明全新安装可直接复现。

## 逐提交记录

| 提交 | 改动及实际验证 | 性能/精度证据与限制 |
| --- | --- | --- |
| `6c9d552a1` | 首次在 main 接入模型级 CSA adapter、PyPTO kernel 与 CPU 合约测试；19 个相关文件 AST 解析和 ABI 参数顺序静态核对通过。 | 当时本机缺少 `vllm`，pytest 在 conftest 导入阶段中止；没有可归属此提交的 NPU 精度或延迟结果。 |
| `22beafb39` | 修正 MLA storage 字段；新增 `test_kv_cache_interface.py` 回归用例。 | 提交包含测试代码；现有记录没有此提交独立的测试执行结果或 NPU A/B，不能记作通过。 |
| `45c0e1ae4` | 对齐固定 vLLM 版本的 NPU runner RoPE 路径；Python 编译、`git diff --check`、增量 pre-commit 通过。 | 当时 227 重测仍在待办，未记录此提交独立的 NPU 延迟/精度。 |
| `3b45322e7` | 分配时复用计划好的每层 KV cache spec；新增 model runner 回归用例。 | 现有记录没有该提交独立的测试执行结果或 NPU A/B。 |
| `362073aba` | 改成 `AscendDSAImpl.forward` 内的原生 metadata 接入，50 tensor ABI，删除 Python `token_valid` 计算；CPU 合约 25 项通过，单层 Graph A/B 见下表。 | B4/S6 的 8K、128K 输出 relative L2 约 1.83%、1.84%，均未通过 1e-2 allclose；不能视为精度等价。 |
| `39993cd43` | O-proj 改成整行动态量化语义；新增 `test_pto_oproj.py` 的 CPU 数学边界测试。 | 该测试仅覆盖 W8A8 O-proj 数学边界。没有记录此提交在 227 同口径的完整 CSA Graph A/B；此前 8K/128K 数据不能自动归属此提交。 |
| `9f1599864` | 使用原生 `query_start_loc` 驱动 TND 非等长请求；新增 CPU 合约及 ABI 校验。 | 后续 `f53cdacf8` 快照的 CPU 合约 29 项通过；此提交之后又修正了标量 dtype 才完成正式单卡 NPU A/B，不能把该数据单独归给它。 |
| `f53cdacf8` | 修正 TND scalar dtype；CPU 合约 29 项、ABI AST 校验、specialize/lower、B4 等长/非等长单卡 eager 与 Graph 通过；B16/T60 首次运行遇到 256 MiB ring heap 耗尽。 | 下表列出 B4 对拍及 B16 失败；最终输出约 1.5% relative L2，未通过精度验收。 |
| `9243a5f5e` | 将 Indexer 与 Attention/O-proj 分到兄弟 `pl.scope()`，TopK 输出保留父级零拷贝 view；B4 回归和 B16/T60 单卡运行通过。 | 6×200 replay 长测显示 B4/T18 CSA 慢 2.52%，B16/T60 慢 9.68%；输出精度仍未通过。 |

`6c9d552a1` 至 `3b45322e7` 的条目如实区分“有测试代码/静态检查”和
“有可核实的测试执行结果”。请勿将后续提交的 NPU 结果反向标记为早期提交通过。

### 原生 metadata 接入后的 B4/S6 记录

`362073aba` 的仓库 README 留存了以下单次单层 Graph 结果。真实权重、合成
hidden/history、固定 S6，按上下文长度分组；原始逐轮样本未随提交保存，
因此只作为历史定位点，不作跨提交显著性判断。

| 上下文 | Native graph | CSA graph | CSA 相对 Native | 最终输出 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| 8K | 0.5842 ms | 0.5731 ms | 快 1.90% | 约 1.83% |
| 128K | 0.7763 ms | 0.8159 ms | 慢 5.10% | 约 1.84% |

`39993cd43` 的 O-proj 变更需要区分 checkpoint 的真实量化方式。
旧的 `official-l3` BF16 O-B 数据属于另一实验分支与另一套环境；不能拿它与
本分支 W8A8 TND 测试直接比较。该提交的 CPU 用例也不证明整层输出对齐。

### 原生 TND：`f53cdacf8`

等长 B4 和非等长 B4 均完成单卡 eager、Graph capture/replay，CSA 入口实际执行，
输出 guard 未被误写。每组 Graph 测量窗口较短，仅作功能烟测。
对应单卡任务分别为 `task_20260924_125844_294850715570` 和
`task_20260924_130056_327280129560`，均退出 0。

| 请求长度 / 总 token | Native graph | CSA graph | CSA 相对 Native | eager / graph 输出 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| `[6,6,6,6]` / T24 | 0.5814 ms | 0.6085 ms | 慢 4.67% | 1.5042% / 1.5495% |
| `[3,4,5,6]` / T18 | 0.5768 ms | 0.5601 ms | 快 2.90% | 1.5536% / 1.7460% |

非等长 T18 的 compressed KV 逐元素相同，main/inner state relative L2
分别约 `3.04e-7` / `2.16e-7`；raw KV 为 `2.86e-3`，index key 为
`7.53e-3`，最终输出为 `1.55e-2`。两组输出均未通过 1e-2 allclose。

B16 请求长度为 `[1,2,3,4,5,6,1,2,3,4,5,6,3,4,5,6]`，T60。
首次设备任务在 CSA 首次执行时遇到 Simpler 256 MiB ring heap
head-of-line deadlock：已使用 `265,095,168` bytes，下次申请
`7,864,320` bytes。任务为 `task_20260924_125523_26342996500`；设备没有
进入可比较的 Graph A/B，这不是精度结果。

### 作用域修复：`9243a5f5e`

此提交的 NPU 验证先在与最终提交内容相同的工作树快照上执行。生成的
orchestration C++ 将 Indexer 的已知静态分配约 129.56 MiB（含 96 MiB
`pair_arena`、12 MiB `score_arena`）和 Attention/O-proj 的约 108.42 MiB
（含 48 MiB `partials`、24 MiB `o_packed_heads`）放入两个兄弟 scope。
首两版（任务 `task_20260924_140618_106971418693`、
`task_20260924_141803_143837220280`）在 C++ 编译阶段因 TopK SSA
别名跨 scope 失败；父级 view 方案编译通过。
这些编译失败没有执行设备 kernel。修复后 B16/T60 不再触发 ring heap deadlock。

| 用例 | 首次通过的任务 | 短测 Native / CSA graph | eager 输出 relative L2 |
| --- | --- | ---: | ---: |
| B4 `[6,6,6,6]`，T24 | `task_20260924_142117_158004131317` | 0.6084 / 0.5804 ms | 1.5042% |
| B4 `[3,4,5,6]`，T18 | `task_20260924_142245_161927419622` | 0.5854 / 0.5661 ms | 1.7820% |
| B4 `[3,4,5,6]`，T18 重复 | `task_20260924_142531_17036064917` | 0.6060 / 0.5407 ms | 1.5536% |
| B16 非等长，T60 | `task_20260924_142245_161848814870` | 0.7504 / 0.8401 ms | 1.4525% |

短测每轮仅 5 次，出现 B4 “CSA 快 3%～11%” 的假象。扩大为每轮 200 次、
6 轮后，得到可用于本次比较的结果（设备 Graph replay，单位 ms/forward）：

| 用例 | 任务 | Native 中位数 | CSA 中位数 | CSA 差距 | 去首轮均值差距 |
| --- | --- | ---: | ---: | ---: | ---: |
| B4 `[3,4,5,6]`，T18 | `task_20260924_143258_19254782899` | 0.5813 | 0.5960 | 慢 2.52% | 慢 2.03% |
| B16 非等长，T60 | `task_20260924_143258_192532413449` | 0.7232 | 0.7932 | 慢 9.68% | 慢 10.38% |

| 用例 | Native 六轮均值 | CSA 六轮均值 |
| --- | --- | --- |
| B4/T18 | `.5855, .5815, .5811, .5810, .5818, .5811` | `.6051, .5949, .5897, .5863, .5970, .5975` |
| B16/T60 | `.7252, .7231, .7118, .7237, .7129, .7233` | `.7986, .7936, .7928, .7985, .7911, .7918` |

B4 六轮 CSA 均慢 0.92%～3.35%；B16 六轮均慢 9.47%～11.38%。
本次精度仍未通过：B4/T18 eager 输出 relative L2 在独立任务间为
1.5536% 或 1.7820%，长测任务为 1.7820%；B16/T60 为 1.4525%。
重复的 B4 短测与修复前 eager 输出指标一致，但跨任务仍有波动，不能仅凭
作用域改动宣布逐元素输出不变。单层测量也不能外推到整模型吞吐。

### 确定性精度探针：`9243a5f5e` 的 B4/T18

2026-09-24 在与该提交相同的 TND kernel 工作树上，用真实第 2 层权重、
seed=62 的合成 hidden/history，测试请求长度 `[3,4,5,6]`、T18、
历史起点 8186、TP1 和 128 槽页。Native 开启
`torch_npu.npu.set_deterministic_level(1)` 与
`HCCL_DETERMINISTIC=true`。这次仅测 eager 精度，没有测 Graph 延迟。
一次性探针的 SHA256 为
`399ff61c9450eaff39c88271777cb43fae01bd7a1f78daf43c3e5b816ffa958d`。

| 任务 | 诊断 | 结果 |
| --- | --- | --- |
| `task_20260924_152756_49096928374` | 未过滤 Native scatter，逐次核对负槽和 indexer WqB 权重 | 4 次 scatter 中后三次各有 3 个 `[-1,127]` 槽；所核对权重始终未变。 |
| `task_20260924_152925_5860015193` | 只在诊断进程过滤负槽，再做 Native/CSA 对拍 | 最终输出 relative L2 为 1.494277%，`allclose(1e-2,1e-2)=false`。 |
| `task_20260924_153040_61598129924` | 不过滤负槽，使用相同输入做 Native/CSA 对拍 | 所有下列对拍指标与过滤组完全相同。 |

两次完整对拍的 Native Q、KV 投影、index 投影、TopK、attention
输出、O-proj 输出等 13 个已捕获阶段逐元素相同；过滤负槽没有改变这组
输入的 Native 结果。CSA 入口均实际执行，Native 的 indexer WqB 权重
没有变化。因此先前另一个分支上发现的负槽写坏权重问题，**不能解释本组**
约 1.49% 的差异；这不代表原生 scatter 在其他输入或内存布局下安全。

| 实际写入区域或输出 | CSA 对 Native relative L2 | 逐元素 1e-2 验收 |
| --- | ---: | --- |
| compressed KV | 0 | 通过 |
| raw KV | 0.285536% | 通过 |
| main state | 约 3.04e-7 | 通过 |
| inner state | 约 2.16e-7 | 通过 |
| index key | 0.753399% | 未通过 |
| index scale | 0.403080% | 通过 |
| 最终输出 | 1.494277% | 未通过 |

原始 JSON 和 Native 阶段张量保存在 227 独立实验目录的
`logs/tnd-precision/{filtered-b4-t18,unfiltered-compare-b4-t18}/`。
当前探针尚未导出 PyPTO 内部 Q、KV、TopK、attention 和 O-proj 阶段张量，
所以这些缓存差异不能单独证明最终输出的首个误差来源。

### 单层阶段定位：原生舍入边界的诊断副本

沿用上一节的真实权重、确定性 Native、TP1、B4/T18 输入及独立 cache。
仅在 `reports/dsv4-tnd-kernel-20260924/npu-ab/` 建立 PyPTO kernel
诊断副本，不修改正式源码。探针额外导出 Q、KV、QR、TopK 和 inverse RoPE
之后的 attention heads；第一组证明插入导出操作后，输出与原 CSA 逐元素相同。
三组任务保存的 13 个 Native 阶段张量也逐元素相同。原始 JSON 和张量已导出到
本地 `reports/dsv4-tnd-kernel-20260924/npu-ab/evidence/`；远端原件在独立
实验目录的 `logs/tnd-precision/`。

| 诊断任务及变体 | Q / raw KV 对 Native relative L2 | inverse RoPE 后 heads relative L2 | TopK 集合平均重合数 | 最终输出 relative L2 / 1e-2 allclose |
| --- | ---: | ---: | ---: | ---: |
| `task_20260924_154525_46844324644`，原 CSA 加导出 | 0.687693% / 0.285536% | 0.956048% | 507.06 / 512 | 1.494269% / 未通过 |
| `task_20260924_155025_133632710259`，Q/KV 补原生 BF16 中间舍入 | 0.043877% / 0.008113% | 0.750544% | 508.44 / 512 | 1.248881% / 未通过 |
| `task_20260924_155507_152590025368`，再补 Indexer 中间舍入及未折叠 Hadamard | 0.043877% / 0.008113% | 0.229699% | 511.72 / 512 | 0.702841% / 通过 |

对应 TopK 逐位置相同数分别为 `969/9216`、`1444/9216` 和
`7131/9216`。集合重合数用于区分排序变化与实际选点变化；它不等同于
最终输出验收。第一组原 CSA 导出与未插桩 CSA 输出逐元素相同。第三组
候选输出的平均绝对误差约 `0.001205`，最大绝对误差为 `0.011719`。

这些是 **eager 单层数值诊断**。Q/KV 副本补入投影输出和归一化、RoPE
输入处的 BF16 舍入；Indexer 副本补入反量化、Hadamard 归一化、权重投影
后的 BF16 边界，同时用原生 Hadamard 矩阵替代适配层折叠版本。
因此证据支持“Q/KV 和 Indexer 的数值边界均有贡献”，尚不能把剩余差异
归因于其中某一行。第三组还没有证明六类实际写入 cache/state 全部通过，
也未测 ACLGraph replay、延迟、B16 或整模型。单卡 cache 复核任务
`task_20260924_155829_171093213983` 已提交，记录本节时仍在队列中；
其结果应在完成后追加，不能把本节的输出通过当作整个 CSA 精度通过。

## 后续每次测试的记录方式

在每次影响 CSA 的提交或实验后，先保存原始 JSON/日志，再在本文**追加**一节。
一项记录至少包含以下字段；未测的字段写“未测”，失败也保留：

```text
日期与源码：vllm-ascend commit（若有工作树补丁，附 diff/hash）、
              PyPTO commit、Simpler commit、runner 脚本版本
环境：服务器/设备、CANN、Torch、Torch-NPU、vLLM、确定性开关
负载：模型与权重层、TP/DP/EP、B、每请求 query 长度、总 T、历史长度、页规格
精度：Native/CSA 入口、输出及实际写入 cache/state 的 relative L2、
      allclose 阈值与结果、重复运行是否稳定
性能：eager 或 Graph、测量边界、预热次数、轮数×每轮 replay 次数、
      Native/CSA 每轮值、中位数、相对差距
执行：静态/CPU/NPU 各阶段结果、任务或日志标识、失败阶段和根因
结论：通过了什么、未通过什么、与哪一条同口径历史记录可比较
```

延迟优化至少同时记录同组 Native；出现小于约 3% 的差距时增加轮数与
每轮 replay 次数，再根据轮间波动判断方向。精度未通过时，性能结果标为
“实验性能”，不能当作等价替换收益。未来 B32/B40、128K、整模型、
GBS=16×4 或 DP=EP=16 的结果必须另立负载记录，不能复用这里的单层数值。

## 2026-10-09：逐位精度攻坚（opus55-20261008，纯诊断，未改算法）

```text
日期与源码：2026-10-09。vllm-ascend dev/pypto-dsv4-csa-tnd-opus55-20261008，
              起点 04724c0e5，工作树无算法改动（shipped decode_sparse_attn_csa.py
              md5 b2f741093838241be81daae21ecf7e0b、decode_csa.py
              md5 6295507b28b3387788186315a392b8d3）。探针仅存在于 227 未入库的
              probe 副本（KV 行和 / 分子 / mm2 的 k=128 显式拆分，三个开关）。
              PyPTO、Simpler 同本文开头所述 TND 环境。
环境：227 单卡，CANN 9.2.0-beta.2、Torch 2.10.0、Torch-NPU 2.10.0.post4、
      vLLM 0.29.0；native 侧 set_deterministic_level(1) 与 HCCL_DETERMINISTIC=true。
负载：DeepSeek-V4-Flash-0731-w8a8 真实第 2 层 C4 权重，TP=1，S=6，block=128，
      seed=62 合成 hidden/history。本节上板配置为 phase-8189、phase-8194、
      uniform-b4；margin 全量统计枚举出 18 个配置（见精度文档的口径说明）。
精度：整层 output 验收口径 17 个配置，15 个对 native 逐位一致，2 个失败，
      各只有 1 个元素（786432 分之一）落到 BF16 边界另一侧：
      phase-8189 relative_l2 = 3.6052e-04、phase-8194 = 3.3952e-04。
      新增探针首次拿到设备 fp32 分子 m_oi 与商 n_full；
      bf16_rint(n_full) 对 heads 的 448 个 nope 列 0/688128 失配（三配置）。
      输入侧全部实测关闭：q 0/786432（×4 配置）；mm1/mm2 实吃的 640 行压缩 KV
      行和 15360/15360 全部对上（判别力已标定，换相邻行的相对偏差中位 1.62/1.75/1.35，
      比 fp32 归约噪声高 4 个数量级）；topk 槽位含顺序 0/12288（两个失败配置）；
      逐行 actCmpS2Size == 512、有效槽 512。
性能：未测。本节全部为诊断，无算法改动，不产生性能结论。
执行：task_20261009_122812_26017125597（kvrs-*，KV 行和探针，三配置 exit 0）、
      task_20261009_123707_276557120222（kvrs2-*，加分子探针，三配置 exit 0）、
      task_20261009_132219_393162810449（kvrs3-*，mm2 显式 4×k=128，三配置 exit 0）。
      每配置跑前 rm -rf build-<name> 强制重编。失败两次均为主机侧脚本 bug
      （未导出属性引用、名字守卫的 shell 展开），非 kernel 问题。
结论：mm2 压缩块的 K 粒度与 native 同构——ptoas 自己把一次 K=512 下成 4 条
      k=128 累加进 fp32 L0C，显式手写 4×k=128 后设备实测 0/688128 维变化、
      probe_vs_csa_output = 0.0，该通道关闭。
      剩余 2 个失败元素的机制已定性为 BF16 栅格的选择效应：flip 维恒为该行 448 维中
      离格边界最近的那一个（27/27 的 flip 维 margin ≤ 3.0 fp32 ULP，非 flip 维中位约
      16000），所需分子位移仅 0.3–1.7 fp32 ULP。
      风险曲线给出硬结论：每配置有 46–66 个维 margin < 3，所以任何
      「只是不同、不等于 native」的改动都会修好 1 个、打坏约 50 个，净变差；
      只有与 native 结构完全一致（残差精确为 0）的改动才可能有净收益。
      未通过：两个失败配置的 output 仍未逐位一致。
      可比较记录：本文 2026-09-24 各节；完整排除清单与方法论见
      docs/source/developer_guide/DSV4_CSA_PRECISION_20261009.md。
```

## 2026-10-09（第二节）：压缩块的 l 按 native 的 radix-8 取分块和 —— output 达成 18/18 逐位

```text
日期与源码：2026-10-09。分支 dev/pypto-dsv4-csa-tnd-opus55-20261008，
              本次改动 vllm_ascend/attention/pto_kernels/dspark/decode_sparse_attn_csa.py
              两处（新增常量 L_REDUCE_CHUNK = ATTN_K_TILE // 2；压缩块的
              qk_li = pl.row_sum(qk_exp, qk_reduce_tmp) 换成两条 64 宽 row_sum
              加一条 add，窗口块不动）。不新建 tile，复用 :342 的 qk_reduce_tmp。
              PyPTO、Simpler 同本文开头所述 TND 环境。
环境：227 单卡，CANN 9.2.0-beta.2、Torch 2.10.0、Torch-NPU 2.10.0.post4、
      vLLM 0.29.0；native 侧 set_deterministic_level(1) 与 HCCL_DETERMINISTIC=true。
负载：DeepSeek-V4-Flash-0731-w8a8 真实第 2 层 C4 权重，TP=1，S=6，block=128，
      seed=62 合成 hidden/history。精度覆盖全部 18 个验收配置
      （uniform-b4 / varlen-b4 / cfg-eq5 / cfg-uneq2 / cfg-mixed 加 phase-8184..8196）。
精度：整层 output 在 18/18 个配置上对 native relative_l2 = 0.0。
      heads 元素级 17/18 个配置零失配；phase-8196 余 2 个元素
      （t=14 h=41 dim=352、t=14 h=58 dim=348），是既有残差、不传导到 output。
      本次合计修好 6 个元素、新增 0 个：phase-8189 的 t18h6 dim266、
      phase-8194 的 t17h25 dim42 与 t23h42 dim359、uniform-b4 的 t2h57 dim151、
      phase-8196 的 t3h39 dim330 与 t15h53 dim271。
      改动前 phase-8189 = 3.6052e-04、phase-8194 = 3.3952e-04，现均为 0.0。
      m_oi 变化 0/688128 维（三配置），l 变化 261/299/255 行（主机预测 262/295/237）。
      四个 flip 行的设备 Δl 为 +2.000 / −2.000 / −1.000 / −2.000；
      窗口段 slot0 四行全 +0.000、全层 1536/1536 对旧 dump 逐位相同。
      phase-8196 的「既有而非新增」靠同一棵树只翻开关的 off 基线判定
      （OFF 失配 4 → 改后 2，修好 2、新增 0）。
性能：没测出可信代价。臂树 cp -al 建、候选文件先断链再打补丁、空对照是两份完全相同的
      shipped 树；同卡配对、按 rep 奇偶换先后、n=16，只比 graph.pto.median_ms 绝对中位数。
      uniform：空对照 +0.0011 ms (+0.17%)、候选 +0.0082 ms (+1.31%)、
               候选−空对照 +0.0072 ms (+1.14%)，候选符号一致 9/16；
      varlen： 空对照 −0.0047 ms (−0.75%)、候选 −0.0036 ms (−0.59%)、
               候选−空对照 +0.0010 ms (+0.16%)，候选符号一致 6/16。
      两形状符号一致性都没过门槛；逐样本跨度 ±0.02 ms ≈ ±3%，远大于 +0.007 ms 的中位位移；
      空对照自己在 varlen 上就是 −0.75%（两条臂同一份代码），所以地板不是零中心。
      两形状 +1.1% 与 +0.2% 不自洽 ⇒ 仍在噪声里。算术上 16 条向量指令对 0.63 ms 内核
      应在 0.1% 量级；要分辨 ±1% 需 n ≥ 64 或更安静的机器。
      全部 128 次 run 两条臂 accuracy_pass = True、relative_l2 = 0.0。
执行：task_20261009_155621_40405646273（kvrs5-*，三配置，2m31s，exit=0）、
      task_20261009_160406_8827347453 起的 kvrs6 分批（其余 15 配置，exit=0）、
      task_20261009_170241_40733052266（kvrs6off-phase-8196，off 基线）。
      生成码在 shipped 路径上核对：pto.sync.set 6→6、pto.sync.wait 9→9、
      pto.tload 30→30、pto.tstore 17→17 全不变，只有 pto.trowsum 1→3、pto.tadd 8→9。
      踩到的工具坑：队列有约 5 分钟的运行时自动终止，而 task-submit --list 的 Done 行
      duration 含排队时间；整趟 15 配置 sweep 与整趟 perf 都因此被砍，改小批并行后才通过。
结论：通过了什么——l 的归约树是精度残差的唯一贡献者（窗口 PV 与 mm2 的 tile K 贡献经
      实测均为精确的 0），改成 native 的 radix-8 分块后 output 达成 18/18 逐位，新增 0。
      未通过什么——phase-8196 的 2 个既有元素级残差未攻；b16、128K、整模型、
      动态 padding、DP16 全部未测，本节数值不可外推。
      可比较记录：本文 2026-10-09 第一节（同环境、同负载、改动前状态）。
      完整定位过程、排除清单与方法论见
      docs/source/developer_guide/DSV4_CSA_PRECISION_20261009.md（已随本次修订）。
```

## 2026-10-10：weights 投影的 K 切分改回单累加器（WEIGHTS_OK 4→1）—— heads 达成 18/18

```text
日期与源码：2026-10-10。分支 dev/pypto-dsv4-csa-tnd-opus55-20261008（起点 05e5d7c23）。
              本次改动只有 vllm_ascend/attention/pto_kernels/dspark/decode_indexer.py:86
              的 WEIGHTS_OK = 4 -> 1（含注释）。WEIGHTS_K_TILE = D // WEIGHTS_OK 随之
              由 1024 变 4096。不需要 if WEIGHTS_OK > 1 守卫。
环境：227 单卡，CANN 9.2.0-beta.2、Torch 2.10.0、Torch-NPU 2.10.0.post4、vLLM 0.29.0；
      native 侧 set_deterministic_level(1) 与 HCCL_DETERMINISTIC=true。
负载：DeepSeek-V4-Flash-0731-w8a8 真实第 2 层 C4 权重，TP=1，S=6，block=128，
      seed=62 合成 hidden/history。精度覆盖全部 18 个验收配置。
精度：heads 对 native：ON 18/18 全零失配；OFF 仅 phase-8196 的 t=14 h=41 dim=352 与
      t=14 h=58 dim=348 ⇒ 修好 2、新增 0。output relative_l2 两臂都 18/18 = 0.0。
      其余行实测一动不动：410 行逐位比，只有 phase-8196 的 t=14 变（512 槽里 301 个，
      ULP 差 -8..-6392），其余 409 行一位不动。集合 18/18 零坏行；8196 t=14 对 native 的
      topk 顺序失配 20 -> 0。两臂 native_stages 与 q 逐位相同，配对有效。
      主机门槛（全 18 配置 26240 个系数）：WEIGHTS_OK=1 对 native 26240/26240 逐位；
      WEIGHTS_OK=4 为 26238/26240，只差 phase-8195 与 phase-8196 的同一个 t=14 h=24。
      根因：native 的 weights_proj 是单一 fp32 累加器 256 步 k=16；4 段切分使该点偏高
      40 个 fp32 ULP（0x3df68028 对 0x3df68000），而 0x3df68000 恰好是 bf16 中点，
      两边取了不同格点。经 fp16 次正规 coef（1 ULP = 1.80%）放大，分数最多偏 6392 fp32 ULP。
性能：测不出可信代价。空对照 = off vs off2、候选 = off vs on，同卡配对、REP 奇偶换先后、
      n=10/形状；uniform 空对照 -0.0102 ms(-1.48%)、候选 +0.0008 ms(+0.12%)，符号一致 5/10；
      varlen 空对照 +0.0038 ms(+0.65%)、候选 -0.0045 ms(-0.72%)，符号一致 4/10。
      两形状「候选-空对照」符号相反，80 个绝对中位数跨 0.5826-0.7203 ms(22%)。
      但预言的 +4.5% 被排除：那在 uniform 上应是 +0.029 ms，实测配对差中位 +0.0008 ms。
      80 次 run 全部 accuracy_pass=True、relative_l2=0。
执行：36 个 task 全 exit=0，判定脚本 bin/opus55-ab/wokab_chk.py；臂树 src/va-wok1-{on,off,off2}。
      ⚠️ 先纠正了一趟无效数据：2026-10-09 那趟 wok1-* 的 18 配置里，补丁只进了 shipped 程序、
      没进 decode_indexer_probe.py 编出的探针程序（weights_proj_reduce_vllm.pto 仍 tadd=3），
      所以那趟「scores 一行没动、8196 残 2」是假象。本轮两份一起改。
      这是 probe_sync 盲区的又一个实例，新增 pl_check_idxtest_arm.sh 覆盖非 vllm 的
      indexer_test 入口（此前没有这道 lower 检查）。
结论：通过了什么——weights 投影改回单累加器后 heads 元素级达成 18/18，output 维持 18/18。
      未通过什么——短历史档（actCmpS2Size < 512）的 l 归约仍未对齐，k384 上现状
      output relative_l2 = 4.4910e-04，另立一节处理；b16、128K、整模型、动态 padding、
      DP16 全部未测，本节数值不可外推。
      可比较记录：本文 2026-10-09 两节（同环境、同负载）。
      完整定位与方法论见 docs/source/developer_guide/DSV4_CSA_PRECISION_20261009.md。
```

## 2026-10-10：压缩块 l 的低档归约改成 64 通道升序链（门 K == 512）—— output 修好 2 档

```text
日期与源码：2026-10-10。分支 dev/pypto-dsv4-csa-tnd-opus55-20261008（起点 5b92a3ecb）。
              本次改动只有 vllm_ascend/attention/pto_kernels/dspark/decode_sparse_attn_csa.py：
              新增常量 L_CHAIN_LANES = L_REDUCE_CHUNK（= 64），逐 token 读
              l_cmp_keys = (position_ids + 1) // COMPRESS_RATIO，并把压缩块的 l 归约按
              l_cmp_keys >= CMP_TOPK 分档：== 512 保持现有 radix-8 平衡树（两条 64 宽
              row_sum + 四路平衡合并，一位不动）；< 512 改走 [H//2, 64] FP32 通道累加器，
              按 chunk 升序每块两条 pl.add，循环后一条 pl.row_sum(64)。通道累加器是
              紧跟标量归约之后的独立 IfStmt，所以 K == 512 侧的分支 region 在 IR 里是
              字面空的。窗口块一行未动。patch 不含任何探针代码。
环境：227 单卡，CANN 9.2.0-beta.2、Torch 2.10.0、Torch-NPU 2.10.0.post4、vLLM 0.29.0；
      native 侧 set_deterministic_level(1) 与 HCCL_DETERMINISTIC=true。
负载：DeepSeek-V4-Flash-0731-w8a8 真实第 2 层 C4 权重，TP=1，block=128，seed=62 合成
      hidden/history。覆盖 18 个验收配置（全 K=512）+ 12 个短历史/单一 K 档
      K ∈ {100,384,385,416,447,448,449,460,480,481,510,512}。
精度：18 个验收配置（off 基线对同 HEAD 取）heads 两臂都 18/18 零失配、output
      relative_l2 全 0.0；两臂 heads/output/topk/scores 逐位相同，新增 0。
      12 个短历史档 output：base 有 2 档不为 0（k384 = 4.4910e-04 / 908 个元素、
      t1k481 = 3.5386e-04），链式后 12 档全部 0.0 / 0 个元素 ⇒ 修好 2 档、回退 0 档。
      12 个短历史档 heads 元素级：base 合计 20 个失配，链式后 7 个（修好 18、新增 5）。
      新增 5 个坐标：k384 (6,19,163)、k384 (8,58,424)、k448 (10,40,485)、
      t0k480 (5,61,291)、t62k510 (9,19,247)；残留 2 个：t1k385 (7,26,10)、
      t62k510 (9,36,321)。这 7 个都不传导到 output（同一批 dump 里 output 逐位
      失配 0、relative_l2 = 0.0）。s2047 标定档两臂都 0 失配。
      生成码：36 个 kernel 里只有 qk_pv.pto 变，规范化指令序列逐条 diff 为
      tadd +2（K<512 的 else region 内，region 深度 5）、trowsum +1（循环后 else，深度 2）、
      load_scalar +1 与 texpands +1（深度 1，无条件：读 position_ids 与 [32,64] FP32 清零
      seed）；sync.set 6、sync.wait 9、tload 30、tstore 17、tmatmul 12、tmatmul.acc 12、
      texp 2、tcvt 1、tmov 16 等全部不变。
      主机侧（用 va-lane64 臂 dump 的 fp32 qk_exp 当被加数，零 vexp 噪声）：升序链实现
      在 6 个可查行上逐位复现设备 l（6/6），逐块 qk_li 自校验 6/6，掩码列实测精确 +0.0。
性能：未测（用户指示精度完整解决前不跑性能 AB，含空对照）。
执行：kvrs9 共 108 个 task 全 exit=0（base/chain × 31 档 + 449/481 两个已作废的门）；
      臂树 src/vllm-ascend-h5b92（off 基线）与 src/vllm-ascend-h5b92-lk（候选），
      两者只差 decode_sparse_attn_csa.py 及其探针副本，其余文件 diff -rq 为空。
      判定脚本 bin/opus55-ab/{lk8_accept.py,lk_final.py,exact12.py,tkorder7.py}。
      ⚠️ 本轮纠正两个主机复刻口径错误，两次都会把「逐位相同」判成假通过：
      (1) probe_lp 的槽距必须取 T_PAD*H = 24576，不是 attn_lp.numel() // 8
          —— _PROBE_KV_ROWSUM = True 后 _PROBE_LP_ROWS = 8*T_PAD*H + T_PAD*640 = 442368，
          用 numel()//8 会读到 KV 行和区的未初始化值，l 读出来是 0x7fc00000(NaN)，
          而 NaN == NaN 在按位比较下会假通过。lsim.py / val.py 都要改。
      (2) valid_block_mask 整块跳过的压缩块，探针从未写过那 128 列，dump 里是未初始化值；
          主机模型必须跟内核一样跳过这些 chunk，不能当 +0.0 累加。K=384 上会直接得 NaN。
结论：通过了什么——按整层 output 口径，12 个短历史档从 2 档不为 0 变成全 0.0，
      18 个验收配置逐位不变、新增 0；升序链实现已被精确被加数证明忠于规格（6/6 逐位）。
      未通过什么——heads 元素级新增 5 个坐标（见上），所以「零新增」这条硬门槛未过；
      native 在 K ≤ 480 的真实结合方式尚未判定（8 个候选在 native 的 l 可行区间上
      全部落在 ±1 ULP 不确定度内，该方法无判别力）；t0k480 档缺 fp32 qk_exp dump，
      该行未查。b16、128K、整模型、动态 padding、DP16 全部未测，本节数值不可外推。
      可比较记录：本文 2026-10-10 前一节（同环境、同负载、同 HEAD 的 off 基线）。
      完整定位与方法论见 docs/source/developer_guide/DSV4_CSA_PRECISION_20261009.md。
```
