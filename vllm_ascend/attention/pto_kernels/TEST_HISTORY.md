# DeepSeek V4 Flash PyPTO CSA 测试历史

本记录属于 `dev/pypto-dsv4-csa-tnd-main-20260924` 分支，覆盖从首次接入
`6c9d552a1` 到 Hadamard 对齐提交 `34fd15d54` 的全部源码提交。更新日期：2026-09-24。
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
| `882405b33` | 建立按提交记录测试历史的文档；不改运行代码。 | 沿用前面各提交的证据；没有新的 NPU 对拍。 |
| `e0e81f51e` | 写入确定性 Native 的负槽 scatter 排查结果；不改运行代码。 | B4/T18 约 1.4943% 的输出差异不能由该组负槽解释；见下文。 |
| `8b9d430aa` | 写入单层 Q/KV、Indexer 和 attention heads 的阶段探针结果；不改运行代码。 | 诊断副本最低输出 relative L2 为 0.7028%，不是此提交正式 kernel 的结果。 |
| `b39b2b4d3` | 仅将 Q/KV 的 14 处原生 BF16 舍入边界加入正式 TND kernel；与已测 Q/KV 诊断副本 AST 一致，本地相关合约 29 项通过。 | 尚未在该正式提交上重跑 NPU 精度、Graph 延迟或整模型；不能把包含 Indexer 修改的 0.7028% 记作此提交结果。 |
| `67ef8afa7` | 正式 sparse attention 在 inverse RoPE 前把归一化 heads 舍入 BF16，CPU golden 同步该边界；Python 编译与增量 pre-commit 通过。 | 后续仅改文档的 `99296493f` 快照在 227 完成正式 B4/T18：输出 relative L2 `1.231282%`，未通过精度验收；Graph 结果见下文。独立诊断副本的 `0.376710%` 不能当作正式提交结果。 |
| `3982ad343` | 依原生在线 softmax 顺序，将 sink 计入初始和，概率舍入前使用累计最大值；同步 CPU golden。诊断副本 `sparse_attn_csa` 的 AST 与正式源码相同；227 CPU 合约 29 项通过。 | 正式 B4/T18 输出 relative L2 `1.229696%`，仍未通过精度验收；固定 Native 输入的诊断 heads 差异从 `0.030104%` 降到 `0.005752%`。Graph 结果和限制见下文。 |
| `34fd15d54` | 将原生 Hadamard 矩阵直接传入，在矩阵乘法之后缩放并保留 cache 量化前的两次 BF16 舍入；227 CPU 合约 29 项通过，单卡 B4/T18 完成。 | index_key 和 index_scale 从 `0.753399%`、`0.403080%` 降至逐位相同；最终输出仍为 `1.229696%`，精度未通过。Graph 结果见下文。 |

`6c9d552a1` 至 `3b45322e7` 的条目如实区分“有测试代码/静态检查”和
“有可核实的测试执行结果”。请勿将后续提交的 NPU 结果反向标记为早期提交通过。

`882405b33`、`e0e81f51e`、`8b9d430aa` 只改文档。`b39b2b4d3`
没有该提交自己的 NPU A/B；`67ef8afa7`、`3982ad343` 和 `34fd15d54` 已有短测，
但仍没有 B16、长时间 Graph 或整模型结果。跨提交的短测延迟不能直接
代替同组性能结论，精度未通过时也不能视为等价替换收益。

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
该版探针尚未导出 PyPTO 内部 Q、KV、TopK、attention 和 O-proj 阶段张量，
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
归因于其中某一行。这些诊断未测 ACLGraph replay、延迟、B16 或整模型；
不能把单层输出通过当作整个 CSA 精度通过。后续 cache 复核和输入注入结果如下。

### Cache 复核与 Native 输入注入：独立诊断副本

以下四次任务沿用同一 B4/T18 输入、确定性 Native、真实 C4 第 2 层权重，
均只修改独立诊断脚本。正式分支的 Indexer、sparse attention 代码没有因
这些实验改变；每个任务均成功退出。表中“heads”是 inverse RoPE 后、
O-proj 前的输出；两列差异均为相对 Native 的 relative L2。

| 任务与诊断 | heads | 最终输出 | 能说明什么 |
| --- | ---: | ---: | --- |
| `task_20260924_155829_171093213983`，六类有效写入 cache/state 复核 | 0.229699% | 0.702845% | compressed KV、index key、index scale 逐元素一致；raw KV 0.008113%，main/inner state 约 `3.04e-7` / `2.16e-7`；六类均通过逐元素验收。 |
| `task_20260924_161018_220927013747`，候选 heads 送入原生 O-proj | 0.229699% | 0.702845% | 同一组 heads 经原生 O-proj 与 PyPTO O-proj 的输出逐元素一致；剩余误差位于 O-proj 前。 |
| `task_20260924_164035_45942720423`，attention 只注入 Native TopK | 0.085967% | 0.595854% | TopK 选点或排序有贡献，但不能解释全部剩余差异；PyPTO Indexer 仍执行并写 cache。 |
| `task_20260924_165327_179152121329`，再注入 Native RoPE 后 Q | 0.084888% | 0.594030% | Q 的额外贡献很小；尚未排除 raw KV 读取、稀疏索引/mask 和 attention 算术。 |

四次实验的候选 cache 写入指标相同；TopK 和 Q 注入只改变 attention 的
读取输入。未注入时 TopK 为 `7131/9216` 个位置完全相同，集合平均重合
`511.72/512`；顺序差异可能影响归约结果。原始 `result.json` 和阶段张量
保存在工作区 `reports/dsv4-tnd-kernel-20260924/npu-ab/evidence/` 的相应
任务目录；该目录不属于本 Git 仓库，本文保留了可跨提交比较的数值摘要。
下一步应单独验证 Native raw KV 读取，再区分地址/mask 与计算顺序；
目前没有可归属这些诊断副本的 Graph 性能结论。

### 正式 Q/KV 舍入提交：`b39b2b4d3`

此提交只改 `qkv_proj_rope.py`，新增 14 处原生 BF16 中间舍入边界。
提交前将正式源码与已完成单卡测试的 `qkv_proj_rope_bf16_probe.py`
做 AST 比较，结果一致；`test_pto_attn_029.py` 在本地跳过仓库级
`conftest` 后 29 项通过，Python 编译、Ruff E501 和 `git diff --check`
通过。仓库级 `conftest` 在本机因缺少 `vllm` 无法导入；pre-commit 的
密钥扫描因本机缺少 `gitleaks`、下载脚本缺少 `wget` 未运行，其余适用项
通过。独立 Q/KV 诊断副本在相同 B4/T18 输入下的最终输出 relative L2
为 1.248881%，但正式提交尚未直接跑该对拍，也没有包含后续 Indexer
诊断修改。因此 0.702845% 与 0.594030% 均不能作为此提交的精度指标。

### Inverse RoPE 前 BF16 边界：单因素诊断

`task_20260924_174025_54771919064` 在同一单卡 B4/T18、确定性 Native、
相同权重/输入/cache 下完成，退出码为 0。独立诊断副本在一次 PyPTO
sparse attention 计算中同时导出 inverse RoPE 前 BF16 heads、原 FP32
输入的 inverse RoPE heads、以及先将归一化结果舍入 BF16 再做 inverse RoPE
的 heads。正式源码没有因此修改。原路径的 Q、KV、QR、TopK、heads、输出，
以及 Native 所有保存阶段，与上一次 Native TopK＋Q 注入任务逐元素一致；
因此新导出缓冲没有改变对照结果。

| 与 Native 对拍的阶段 | 原路径 relative L2 | BF16 边界版本 relative L2 | 逐元素相同占比：原路径 → 边界版本 |
| --- | ---: | ---: | ---: |
| inverse RoPE 前 heads | 0.031252% | 0.031252% | 58.89% → 58.89% |
| inverse RoPE 后 heads | 0.084897% | 0.032232% | 57.92% → 59.03% |
| 最终输出 | 0.594030% | 0.376710% | 29.46% → 42.58% |

边界版本最终输出是将该版本 heads 送入**原生 O-proj**得到的诊断值；
此前同一组 heads 的原生 O-proj 与 PyPTO O-proj 已逐元素一致，但本次未在
正式 CSA 内运行第二次 PyPTO O-proj。两版本的 NoPE 448 维逐元素相同；
RoPE 64 维对 Native 的 relative L2 从 0.276496% 降到 0.045079%。
本次 0.29 原生 `npu_sparse_attn_sharedkv` 返回 BF16 heads 后才调用 inverse RoPE；
当时的 `decode_sparse_attn_csa.py` 在构造 BF16 值后，仍用舍入前的 FP32
值做 inverse RoPE。此次实验确认该数值边界是主要放大点之一；之后由
`67ef8afa7` 修正正式源码，但该提交仍待独立 NPU 复测。

边界修正后，inverse RoPE 前 heads 仍有 0.031252% 差异，仅 58.89%
逐元素相同；最终输出也未达到逐位一致。本次没有测 Graph、B16 或整模型，
不能将 0.376710% 归属正式源码提交。接下来应先在 attention 归一化前后
比较 Native 与 PyPTO，并分别排除 raw KV 读值、稀疏索引/mask、softmax
和 QK/PV 归约顺序。原始 JSON 与阶段张量保存在工作区
`reports/dsv4-tnd-kernel-20260924/npu-ab/evidence/rope-boundary-b4-t18/`。

### Native raw KV 读取：单因素诊断

`task_20260924_181846_19423658778` 在相同单卡 B4/T18、位置 8186、
确定性 Native 和真实 C4 第 2 层权重下完成，退出码为 0。独立诊断副本仅让
PyPTO sparse attention 读取同次 Native 的 raw KV cache；PyPTO 自身的 raw
cache 写入仍执行并用于对拍。Native Q、TopK 注入及 inverse RoPE 前 BF16
舍入双版本与上节一致；正式源码没有修改。

两次任务的 13 个 Native 阶段张量逐元素相同。PyPTO 的 Q、KV、QR、TopK
阶段张量也逐元素相同，六类 cache/state 写入对拍指标完全相同；因此下面的
变化可归于 sparse attention 读取的 raw KV。

| 与 Native 对拍 | 原 raw KV 读取 | 注入 Native raw KV | 变化 |
| --- | ---: | ---: | ---: |
| inverse RoPE 前 heads relative L2 | 0.031248% | 0.030104% | 降 0.001144 个百分点 |
| BF16 边界后的 heads relative L2 | 0.032232% | 0.030771% | 降 0.001461 个百分点 |
| BF16 边界 heads 经原生 O-proj 的输出 relative L2 | 0.376710% | 0.366880% | 降 0.009830 个百分点 |

最后一行仍是把诊断 heads 送入**原生 O-proj**的代理值，并非正式 CSA
源码的第二次 O-proj。注入后，inverse RoPE 前绝对值至少 0.01 的 BF16
元素全部在 Native 的 1 ULP 内；最终输出这一范围内只有约 79.67% 在
1 ULP 内，约 5.03% 超过 4 ULP。raw KV 写入本身相对 Native 的
relative L2 仍为 0.008113%，说明注入的是读取输入，不是修复了写入。

结论：raw KV 差异只解释了剩余 heads 误差的一小部分。Native Q、TopK、
raw KV 已固定，compressed KV 的实际写入行逐元素相同；接下来应核实
attention 实际访问的 compressed KV 行、稀疏索引和 mask，再比较 QK、
softmax、PV 及归一化的数值边界与归约顺序。当前不能把剩余误差直接归因于
某一个乘法或 softmax。原始 JSON 与阶段张量保存在工作区
`reports/dsv4-tnd-kernel-20260924/npu-ab/evidence/native-raw-b4-t18/`；
实验脚本 `decode_csa_stage_probe_native_raw.py` 和
`precision_probe_native_raw.py` 的 SHA256 分别为
`4ce7eebc46ee5db321b20f459bcf78fbec6a0c4ec8bc52e52e427ba9e87a6ca1`、
`cf9d7ccad1fe6da80249b5718852d81a747d043dff8114fb78af878c8722cd73`。
本次未测 Graph、B16 或整模型。

### Native compressed KV 读取：单因素诊断

`task_20260924_191051_288095615984` 在与上节相同的 B4/T18、位置 8186、
确定性 Native、真实第 2 层权重和单卡环境下完成，退出码 0。独立诊断副本在
已注入 Native Q、TopK、raw KV 的基础上，仅让 sparse attention 再读取
Native compressed KV；PyPTO 的全部 cache 写入仍执行。此任务没有修改正式源码。

与上一任务逐张量比较，保存的 13 个 Native 阶段和 10 个 PyPTO 阶段
**全部逐元素相同**。inverse RoPE 前 heads relative L2 仍为 `0.030104%`，
BF16 边界后 heads 仍为 `0.030771%`，送入原生 O-proj 的输出仍为
`0.366880%`；压缩 KV 读取不是这组输入剩余误差的来源。实际写入的
compressed KV 行也逐元素一致。整块 cache 中未初始化的历史/空页不能直接
比较；本任务的整块比较出现不等和 NaN，不作为有效行差异证据。

本地原始结果在
`reports/dsv4-tnd-kernel-20260924/npu-ab/evidence/native-both-b4-t18/`。
下一步对照 227 实际加载的 `npu_sparse_attn_sharedkv` 源码，单独验证
sink、softmax 及 PV 归并顺序。该源码的
`sparse_attn_sharedkv_scfa_block_vector.h` 与本 checkout 文件 SHA256 同为
`e0580ca8c6f84ac25875532c25412120b06001d692d002af2522f2852f65bcd9`；
源码一致不等于最终加载二进制已被完全证明，后续还需核对 OPP 构建产物。

### 在线 softmax 数值顺序：诊断与正式提交

227 自有 OPP 源码的 `sparse_attn_sharedkv_scfa_block_vector.h` 显示在线
softmax 从 `max=sink`、`sum=1` 开始，并在每个 S2 块持续更新；tiling 源码
`sInnerSize_=512`。本分支原先每 128 行独立取最大值、先舍入 BF16 概率，
再合并输出，并在最终分母补 sink。两种数学表达式接近，但舍入位置不同。

固定 Native Q、TopK、raw KV、compressed KV 的同组 B4/T18 诊断：

| 变体与任务 | inverse RoPE 前 heads relative L2 | BF16 边界后 heads relative L2 | heads 经原生 O-proj 的输出 relative L2 |
| --- | ---: | ---: | ---: |
| 旧归并，`task_20260924_191051_288095615984` | 0.030104% | 0.030771% | 0.366880% |
| sink 初值，`task_20260924_211253_117741812005` | 0.030104% | 0.030771% | 0.366382% |
| sink 初值＋累计最大值，`task_20260924_211516_124009615551` | 0.005752% | 0.005841% | 0.189485% |

三个任务的 13 个 Native 阶段张量逐元素相同；PyPTO Q、KV、QR、TopK
和六类有效 cache/state 写入指标保持相同。PyPTO TopK 与 Native 逐位置
相同 `7131/9216`，每 token 集合交集平均 `511.72/512`；此表 attention
均注入 Native TopK，故不能据此推断正式路径改善相同。原始结果在工作区
`reports/dsv4-tnd-kernel-20260924/npu-ab/evidence/` 对应任务名目录。

失败尝试也保留：首版 sink 诊断的 `pl.tile.full` 初值在 PyPTO
`ResolveBackendOpLayouts` 触发 `[32,1]` 与 `[1,32]` 布局冲突；改为从
原有 `running_m` 用 `muls`、`adds` 构造同布局初值后运行通过。256 行
分块任务 `task_20260924_211259_118158630696` 在编译期因 Mat buffer
`884736 > 524288` bytes 失败，未执行设备 kernel，没有精度数据。
原生 512 行与 PyPTO 128 行分块仍是未解决的结构差异；不能直接把 tile
常量改到 512。

正式源码单卡结果均使用同一 B4/T18、位置8186、真实 C4 第 2 层权重、
确定性 Native、128 槽页和独立 worktree。两个任务的 Native 输出逐元素相同。
每组 Graph 是 6 轮、每轮 30 次 replay，作为短测记录：

| 正式代码快照与任务 | 输出 relative L2 | 精度验收 | Native / CSA Graph 中位数 | 同组 CSA 相对 Native |
| --- | ---: | --- | ---: | ---: |
| `99296493f`（代码同 `67ef8afa7`），`task_20260924_211619_12626708910` | 1.231282% | 未通过 | 0.5814 / 0.5619 ms | 快约 3.36% |
| `3982ad343`，`task_20260924_212611_14808655928` | 1.229696% | 未通过 | 0.5646 / 0.5491 ms | 快约 2.75% |

正式输出仅下降 `0.001586` 个百分点，远小于固定输入诊断的改善。
说明 Indexer/TopK 或 Q/KV 等上游差异仍占主导；下一项任务
`task_20260924_213336_1524231443` 单独撤掉 Native TopK 注入以量化其贡献。
两个正式快照的 227 CPU 合约各 29 项通过，Graph capture/replay 完成，
CSA 调用计数各为 37，输出 guard 未改。未测 B16、整模型及长时间延迟；
上述同组延迟只保留为实验数据，不构成性能收益结论。

### TopK 候选集合：单因素诊断

`task_20260924_213336_1524231443` 在 227 单卡完成，退出码 0。保持
Native Q、raw KV、compressed KV 注入和在线 softmax 诊断实现，仅把 sparse
attention 的 TopK 输入从 Native 改为 PyPTO 自己算出的 TopK。两组保存的
13 个 Native 阶段张量逐元素相同；PyPTO Q、KV、QR、QR scale 和 TopK
也逐元素相同，只有 attention 及其后续结果改变。

| TopK 来源 | inverse RoPE 前 heads relative L2 | BF16 边界后 heads relative L2 | 最终输出 relative L2 |
| --- | ---: | ---: | ---: |
| Native，`task_20260924_211516_124009615551` | 0.005752% | 0.005841% | 0.565954% |
| PyPTO，`task_20260924_213336_1524231443` | 0.213835% | 0.214313% | 0.687120% |

PyPTO 与 Native 的 TopK 逐位置一致 `7131/9216`，但排序差异本身
不是主要误差：14 个候选**集合完全一致**的 token，换顺序后 heads
相对变化最大仅约 `0.0011%`。另外 4 个 token 的集合交集为
`511、511、510、511/512`，heads 相对变化约
`0.40%～0.55%`。因此应继续核对临界候选的 Indexer 分数、量化输入
和 TopK 截断边界。原始结果和阶段张量在工作区
`reports/dsv4-tnd-kernel-20260924/npu-ab/evidence/running-max-own-topk-b4-t18/`；
此实验是独立诊断副本，不是正式源码的 Graph 性能测试。

### Indexer Hadamard 缩放位置：正式提交

提交 `34fd15d54` 把原生 Hadamard 矩阵不带缩放地传给 kernel，在矩阵乘法后
缩放，并在 Indexer cache 量化前按原生顺序做两次 BF16 舍入。227 上的独立
工作树 `vllm-ascend-34fd15d54` 经 29 项 CPU 合约测试通过；单卡任务
`task_20260924_214652_176158827118` 在同组 B4/T18 真实第 2 层权重、
确定性 Native 和 Graph 6 轮×30 replay 下退出码 0，CSA 调用计数 37。

| 正式提交 | index_key relative L2 | index_scale relative L2 | 最终输出 relative L2 | Native / CSA Graph 中位数 |
| --- | ---: | ---: | ---: | ---: |
| `3982ad343` | 0.753399% | 0.403080% | 1.229696% | 0.5646 / 0.5491 ms |
| `34fd15d54` | 0%，逐位相同 | 0%，逐位相同 | 1.229696% | 0.5668 / 0.5522 ms |

两次最终输出的 relative L2 数值完全相同；该修正解决 Indexer cache
写入精度，不解决当前输出误差，精度验收仍未通过。raw KV 和 compressed
cache 的指标不变。Graph 数值仅为短测记录，未测 B16、整模型或长时间
延迟。原始结果保存在工作区
`reports/dsv4-tnd-kernel-20260924/npu-ab/evidence/formal-34fd15d54-b4-t18/`。
下一步优先定位 TopK 截断附近的少数候选差异，并继续隔离未注入 Native
Q/cache 时的输出误差。

## 后续每次测试的记录方式

在每次影响 CSA 的源码提交或实验后，先保存原始 JSON/日志，再在本文
**追加**一节，提交并推送文档。源码提交写入上面的逐提交表；本文件的文档
提交可通过 `git log -- TEST_HISTORY.md` 追溯，避免在文档内引用自身尚未
确定的提交号。尚未完成的任务写“待验证”，任务完成后再追加实测，不回填为
之前源码提交的运行结果。
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
