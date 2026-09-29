# DeepSeek V4 Flash PyPTO CSA 测试历史

本记录属于 `dev/pypto-dsv4-csa-tnd-main-20260924` 分支，覆盖从首次接入
`6c9d552a1` 到当前原生数值对齐系列的源码提交。更新日期：2026-09-29。
后续改变 kernel、绑定、原生基线或测量方法时，在本文追加记录，保留旧结果，
以便按提交比较。这里的“通过运行”仅指程序完成；数值验收单独标注。

## 测试边界和比较规则

- 227 单卡测试使用 DeepSeek-V4-Flash-0731-w8a8 的真实第 2 层 C4 attention
  权重，固定 seed=62 构造 hidden states 与历史 cache/state。Native 与 CSA 使用
  独立但初值相同的缓存。它不是整模型生成，也没有测 TP/DP/EP=16、EPLB、
  DSpark 接收率或 GBS=16×4 吞吐。
- Native 精度基线开启 `torch_npu.npu.set_deterministic_level(1)` 和
  `HCCL_DETERMINISTIC=true`。`relative L2 = ||CSA - Native||₂ / ||Native||₂`；
  历史记录包含 `torch.allclose(rtol=1e-2, atol=1e-2)`；2026-09-28 起优先检验
  原始字节一致性与 ULP，不再把旧阈值通过作为精度完成。记录输出及实际写入的
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

## 2026-09-28：按原生计算顺序进一步对齐

基线是 `34fd15d54` 的正式源码，文档 HEAD 为 `1ddf5d5fc`。
参考独立 opus55 分支的定位思路，重新核对本仓库原生 DSV4、QLI、
RMSNorm Dynamic Quant 和 Sparse Attention C++，分开提交以下修改：

| 提交 | 变更与正确性依据 | 当前验证范围 |
| --- | --- | --- |
| `1f4fec753` | 原生 `impl.wkv` 投影输出进入共享 PyPTO KV norm/RoPE；QR 对齐完整 K 累加、半区平方和、实际归一化值 amax 和标量除法；Q/O-proj 先合并反量化系数。 | CPU 合约、数学边界检查及 227 编译通过；NPU 待验证。 |
| `a68e6c4e9` | Indexer 对齐 BF16 投影和 Hadamard、FP16 权重/系数/scale、fixpipe 的 ReLU 与 `2^-10`/FP16 边界、新块优先的 TopK 合并。 | CPU 数值回归及 227 编译通过；NPU 待验证。 |
| `0106bfd4f` | Attention 按窗口128和压缩512两个逻辑 chunk 计算，概率按原生 `CAST_ROUND` 舍入，压缩 PV 完整 K 累加并沿输出 N 分块。 | 6项 CPU 回归、227 specialize/lower/codegen 通过；NPU 待验证。 |
| `49cba0a23` | Compressor 对齐 TH NORMAL 的 K 起点旋转、128维 MMAD、交错物理行和8→4→2→1池化归约；保留设备端 TND query bounds。 | 3项 CPU 回归及完整 kernel 编译通过；NPU 待验证。 |
| `f03093456` | Compressor RMS 按64列块进行半区平方和归约，再执行 Sqrt→Div→FP32 gamma。 | 原有主/inner RMS 数值回归和完整 kernel 编译通过；NPU 待验证。 |
| `c63fbdc96` | 用长度差平方和判断等长，避免设备 bool 转 INT32 时 True 为 -1。 | CPU 模拟设备 mask 语义、生成 C++ 检查通过；与下一提交组合的硬件结果见第三批。 |
| `80047af08` | inner compressor 独立采用原生 head128 的投影 K 顺序和 dBase16/64 规则。 | B4/T24、B16/T60 六类实际写入 cache/state 全逐位一致；B4 输出一致，B16 输出仍有差异。 |
| `d8d180c09` | 投影等长判定先裁掉尾部零长请求，保留中间空请求。 | 58项 CPU、完整编译通过；尾部 `[6,6,0]` 输出逐位一致，中间 `[6,0,6]` 仍需 attention 定位。 |
| `bdf2cf0fb` | Indexer weights 从 split-K4 加法改为完整 K4096 连续累加。 | 60项 CPU、完整编译通过；B16 真实 QLI 输入全逐位一致，TopK 集合一致；最终输出仍994处不同。 |

这些改动没有改变 TND 请求边界、slot/block table 的来源或原生 cache 所有权，
没有引入按 S6 对齐的服务 Graph 限制。ABI 从51增至52个 tensor，新增
`kv_projected`。KV norm/RoPE 复用本分支已有实现；原生 WKV 是正式执行路径，
不是将同一次 Native 对拍结果注入 CSA。

### 环境与静态验证

- 227 使用既有 CANN 9.2.0-beta.2、ATB ABI1、Torch 2.10.0+cpu、
  Torch-NPU 2.10.0.post4、vLLM 0.29.0。
- PyPTO `54957491ede07ad5d5015f5e69874f367113cf45`；
  Simpler `32dff953d07f6bd2aacab8532860f28aca6df931`；PTOAS 0.63。
- 复用已有扩展和环境，仅编译修改后的 kernel；框架未重新安装。
- 共48项 CPU 回归通过：35项 metadata/KV 合约、2项 O-proj、2项 Indexer、
  6项 attention、3项 compressor。Compressor 的构造性舍入用例在旧池化实现上失败，
  在本轮实现上逐位通过；CPU DSL 检查也覆盖了有效/无效 slot 与非等长请求。
- QR sqrt 整数中点修正另经独立数学复核；前提是正 normal 输入且 VSQRT
  误差在1 ULP内。生成 C++ 确认小张量位于私有 UB，整数平方/移位保持64位，
  标量除法保留 FP32，存在相应 V/S 同步。
- 首次 specialize 因循环变量推导的 tile 宽度不是编译常量失败；改为
  constexpr 参数的两个显式 chunk 调用后，specialize、lower、codegen通过。
- 增量 pre-commit 通过；本机缺少 gitleaks 可执行文件，该 hook 未执行。
  `bash format.sh ci` 的 shellcheck 因本机缺少可执行文件而失败；ruff、codespell、
  typos、clang-format、markdownlint 等已通过，不将缺工具记为源码失败。

### 硬件对拍安排与证据边界

- 原生反量化顺序小任务：`task_20260928_092241_28589849376`。
- 第一批正式候选（`0106bfd4f`）B4/T18 和 B4/T24：`task_20260928_093210_314681911248`。
- 第二批正式候选（`f03093456`，增加 compressor 对齐）相同负载：`task_20260928_094410_365671727193`。
- 两个负载仍是模型真实第2个 C4 层权重、合成 hidden/history、TP1、128槽页、
  start_pos=8186，长度分别为 `[3,4,5,6]`、`[6,6,6,6]`。
- Native 启用 deterministic level 1 和 `HCCL_DETERMINISTIC=true`。
  本轮增加原始字节一致性和 Native Graph 重复性检查；旧 allclose 不能代替本轮精度目标。
- 首轮仅使用每组6轮、每轮5次 replay 验证 Graph 行为，不用于性能结论。
- 反量化小任务已完成：T18/T24 共172032个 BF16 输出，Native 重复运行与
  合并 scale 参考均逐位相同。不过顺序乘 scale 参考也逐位相同，此随机样本
  **不能单独证明两种计算顺序的区别**；源码边界与实际模型阶段对拍仍是依据。
- 本节更新时正式候选任务已开始执行，尚无完整精度结果；不得将 CPU/编译通过表述为精度通过。
- 远端是以 `34fd15d54` 为基线的独立 worktree 加本轮源码快照；修改文件 SHA256
  记录在本机 `reports/dsv4-tnd-kernel-20260924/native-align-20260928/stage1-manifest.json`，
  第一批源码对应前三个提交的组合；第二批对应五个提交，另记录 `stage2-manifest.json`。
  不覆盖旧实验 checkout。

尚待核验：CANN softmax 内部归约树、短上下文的负槽 compact 顺序、完整 TopK
tie-breaking、B16/128K/整模型。阶段探针须先证明与对应正式源码输出和 cache 逐位一致，
才可用于解释模块误差。

### 第一、二批实测结果（09:55补充）

前三个提交组合与后两个 compressor 提交分别使用独立源码快照。
下表所有已完成项的 Native Graph 重复运行逐位一致，CSA 没有回退原生，
eager 与 Graph 的输出差异数量相同。表中的 state 数量仅统计实际写入行。

| 源码 / 用例 | Graph 输出不同元素/总数 | 输出 relative L2 | main / inner state 不同元素 | Native / CSA Graph ms |
| --- | ---: | ---: | ---: | ---: |
| `0106bfd4f` / b4-t18 | 0/73728 | 0.000000% | 28620 / 4037 | 0.5805 / 0.6686 |
| `0106bfd4f` / b4-t24 | 0/98304 | 0.000000% | 40812 / 9597 | 0.6971 / 0.7929 |
| `f03093456` / b4-t18 | 0/73728 | 0.000000% | 0 / 4037 | 0.5891 / 0.6121 |
| `f03093456` / b4-t24 | 0/98304 | 0.000000% | 40845 / 9597 | 0.6001 / 0.6377 |
| `f03093456` / b4-t18-p8189 | 0/73728 | 0.000000% | 0 / 4032 | 0.5847 / 0.5938 |
| `f03093456` / b4-t18-p8194 | 0/73728 | 0.000000% | 0 / 4078 | 0.5759 / 0.6020 |
| `f03093456` / b16-t60 | 3267/245760 | 0.116280% | 0 / 13421 | 0.7316 / 0.9615 |

本轮6×5次 replay 用于验证 Graph 行为。环境存在其他卡上的并发任务，
这些短测数字仅留作历史记录，不能据此宣称性能改善或稳定的回退比例。
第一批与第二批 B4/T24 的 compressed cache 均为2/4096元素不同，
最大绝对误差6.1035e-5；所有列出的 raw/index key/index scale 均逐位一致。
B4/T18 两个新增相位8189、8194和B16/T60的 compressed cache逐位一致。

阶段探针任务 `task_20260928_094613_37458769572`、
`task_20260928_095015_391228918413` 使用本次正式源码自动导出中间张量。
先证明探针与正式kernel输出、三个完整cache allocation逐字节相同，才进行归因。

| 探针 | Q / raw KV / 主QR codes和scale | TopK | inverse RoPE后的heads | 同heads经原生O-proj |
| --- | --- | --- | --- | --- |
| B4/T18，8189 | 全部逐位一致 | 全部逐位一致 | 1/589824不同，仅1 ULP | 与CSA输出逐位一致；最终输出与Native也相同 |
| B16/T60，8186 | 全部逐位一致 | 65/30720位置不同；仅token56集合有一枚替换 | 30622/1966080不同，主要集中token56 | 与CSA输出逐位一致 |

B16只有token56的最终输出不同，其余token输出逐位一致。token27仅有两个TopK
位置互换且输出不变。**主QR codes/scale不是Indexer的128维query/scale**，
因此尚不能认为QLI的所有输入已一致；下一步需区分Indexer投影/量化输入、score
归约与TopK边界。不能把B4逐位一致推广到B16或整模型。

### 第三批：Compressor设备分支和inner投影

- `c63fbdc96`：修正等长判断。已检查的生成C++把`pl.cast(predicate, INT32)`
  的True转为-1，B4等长计数成为-4，导致错误选择dBase64；改用长度差平方和。
  此根因解释了第二批非等长main state精确、等长仍有FP32末位差异。
- `80047af08`：inner compressor独立对齐head128投影的K128累加、列相关256偏移；
  使用自己的dBase16/64选择规则，不复用head512的32/64参数。
- 当前54项CPU回归通过；新增测试使用True→-1的设备mask语义，旧等长判断会失败。
  227完整kernel编译通过，生成C++确认两个投影均已消除bool计数。
- 第三批相同B4/T24、B16/T60负载：`task_20260928_095429_409899525342`，
  本节记录时仍在执行，不提前写作精度通过。源码身份另存`stage3-manifest.json`。
- 独立复核另发现Native会裁掉query bounds中的尾部零长请求，再判断等长；
  当前正长度小测不覆盖此边界，后续单独修复和验证，不排除中间零长请求。

### 第三批新增实测与padding后续（10:00补充）

`80047af08` 的 B4/T24 已完成：eager/Graph最终输出、六类实际写入的cache/state
全部逐位一致，Native Graph重复逐位一致。主/inner投影和等长分支的修正已获得
该负载的硬件证据，不能把这项成功覆盖尚在执行的B16。

`d8d180c09` 补齐Native对尾部零长请求的处理，仅在投影分块选择中裁剪尾部；
中间空请求仍保留。58项CPU测试通过，227完整kernel编译通过，生成C++确认
两个active_requests变量均保留循环phi和条件更新。独立padding脚本只允许
`[0,6]`长度且至少一个token，其余对拍流程与ab.py相同。硬件专项仍待记录。

B16 TopK单因素诊断 `task_20260928_095712_1993526209` 继续运行Indexer，
只在attention消费端替换Native TopK；同时保存Native QLI真实输入。
该任务属于归因工具，不将注入版当作正式源码精度。

### 第三、四批完整结果与Indexer诊断入口（10:06补充）

- `80047af08` B16/T60已完成：六类实际写入cache/state全部逐位一致；
  eager/Graph输出仍3267/245760不同、relative L2 0.116280%，集中在token56。
  Native Graph重复逐位一致。投影修复消除state差异，并未解决该TopK选点。
- `d8d180c09` 尾部空请求`[6,6,0]`、T12、start8186的任务
  `task_20260928_095922_8732217657`已完成该子用例：输出和六类cache/state
  在eager逐位一致，Graph输出与Native及Native重复运行也逐位一致。
  这是query bounds中的尾部空请求覆盖，不代表已验证所有token容量padding情况。
- 同任务的中间空请求`[6,0,6]`首轮输出871元素不同，六类cache/state逐位一致；
  需继续模块归因，不能将尾部padding结果推广到中间空请求。
- `task_20260928_095712_1993526209`在CLI解析时退出，未运行kernel；
  后续显式传入长度和位置重提，不能记作源码精度失败。
- `task_20260928_100239_2457642357`中原生QLI输入已成功保存，随后额外的
  Hadamard hook检查失败。实际Native走`quantize_update_cache_and_select_topk`
  融合路径，绕过`quantize_query`，因此属于探针覆盖问题；普通基线对拍有效。
  第二版诊断同时捕获融合入口，保持实际QLI query/scale/weights为必检项。

### B16单因素与真正Indexer输入定位（10:13补充）

`task_20260928_100514_31132030226`在`80047af08`基础上只向attention
注入同次Native TopK：输出relative L2从0.116280%降到0.021758%，
heads从30622个不同元素降到8个（均1 ULP），但最终输出仍994个不同元素。
同heads经两边O-proj依然逐位一致，说明微小attention差异可跨过量化边界，
不能仅以heads的1 ULP判定整个替换已经逐位对齐。此处属于诊断注入，未合入正式路径。

`task_20260928_100655_4098433694`的第二版Indexer导出完成；正式输出和
全部cache allocation的等价门槛通过，真实QLI输入结果为：

| 阶段 | 不同元素/总数 | 说明 |
| --- | ---: | --- |
| RoPE后query / Hadamard后query | 0 / 491520，各自一致 | 原生融合quantize/scatter入口已覆盖 |
| 128维INT8 query | 0 / 491520 | 真正的QLI输入 |
| FP16 query scale | 0 / 3840 | 比较时无损扩FP32，差异为0 |
| weights | 2 / 3840 | token40、token56各一个；最大绝对误差1.220703125e-4 |

按Native真实输入重建token56的2047个候选，五种CPU归约均完整复现Native
Top512；657、1627的参考分数相差约1.915e-8，并非同分。CPU重建不是直接
导出的Native score，但结合输入定位，下一单因素应检查weights投影，
不应先改TopK tie规则。源码当前明确将K4096分成四段，再Vector合并；
后续验证单完整K累加是否消除这两个weights差异。

### weights误差的单因素证据与修正候选

Native和CSA的weights仅`[40,14]`、`[56,55]`不同。token56/head55分别是
`-0.0157470703125`、`-0.015869140625`。只将Native参考中的这一个值换为CSA值，
其余query/cache/scale保持原值，三个不同归约版本均完整复现CSA的63处TopK
排名变化和657→1627替换；换回Native值则完整复现Native。token40对应的
FP16乘积系数两边均归零，故不影响TopK。这是CPU输入单因素闭环，正式修复仍需NPU复测。

`bdf2cf0fb`将vllm Indexer weights改为完整K4096连续累加，保留两次BF16
边界和原有scale；四份partial staging变成一份完整投影staging。
独立静态审查确认仍在同一Indexer scope，数据依赖与TopK消费者保留。
60项CPU回归通过；新BF16临界值用例能区分旧分段累加和连续累加。
原探针与新增Indexer导出探针的TopK也已用保存tensor确认逐位置一致。

### weights修复硬件闭环与attention读取顺序排查（10:28补充）

正式源码`bdf2cf0fb`，第五批快照和`stage5-manifest.json`对应；任务
`task_20260928_101701_70308530658`依次执行Indexer导出、B16 Graph、
中间空请求Graph和B4/T24回归。复用前述完整环境和确定性配置，未重装框架。

Indexer探针输出与正式路径的output、完整cache allocation及TopK逐位等价。
真实QLI query（491520元素）、FP16 scale（3840）、weights（3840）、
RoPE和Hadamard全部逐位一致。独立读取保存tensor确认：前一版仅有的
`[40,14]`和`[56,55]`两处weights已修正，Native输入在两次实验间也逐位相同。
60个token的TopK集合全部一致；只剩token27的索引431/432（从0开始）交换key109/1196，
token56排名已完全一致。此负载的weights误差已闭环。

| 第五批负载，start8186 | eager / Graph最终输出 | 六类有效写入cache/state | Native / CSA Graph中位数 |
| --- | --- | --- | ---: |
| B16/T60，前述非等长请求 | 均994/245760不同；relative L2 0.021758%，最大绝对误差0.00390625 | 全逐位一致 | 0.733076 / 0.994856 ms |
| B3/T12，`[6,0,6]` | 均871/49152不同；relative L2 0.042117% | 全逐位一致 | 0.568952 / 0.618212 ms |
| B4/T24，`[6,6,6,6]` | 均逐位一致，98304个输出元素 | 全逐位一致 | 0.587570 / 0.721158 ms |

三组Native Graph重复逐位一致，任务已退出0。Graph仅6轮×5次，记录为实验耗时，
当前不据此选择计算顺序。

B16阶段导出只剩8/1966080个heads元素不同，每个1 BF16 ULP；其中
`[56,11,181]`为Native `0.0091552734375`、CSA `0.00909423828125`。
最终994个差异全在token56，最大最终ULP为310。同heads输入两边O-proj仍
逐位一致。因此不能把中间heads的1 ULP表述为最终输出已经控制到几个ULP。

另一个独立诊断`task_20260928_102307_95344216166`在第三批源码上保持
Native TopK注入，仅把compressed KV读取改为Native `CopyInKv`的物理地址
成对排序规则。CPU检查15360对、7684对实际交换，完整kernel编译通过。
设备探针的Q/KV/QR/scale、Indexer结果和cache等价检查全部通过；与原TopK
注入相比2个heads改变，但仍8处1 ULP，最终输出完全未变，仍994处不同。
所以该读取顺序实验未解释剩余误差，未合入正式kernel。
后续继续导出归一化heads和softmax状态，定位归约或计算边界。

本节证据在工作区`native-align-20260928/evidence/`的`stage5-*`与
`stage3-topk-pair-b16-t60`目录，属于前述`reports/dsv4-tnd-kernel-20260924/`。
探针脚本、AST变换manifest、源码SHA256和独立复核报告与证据一同保存。

### attention归约源码核验与后续诊断（10:44补充，设备待执行）

核对227实际CANN9.2 SDK的`SoftmaxFlashV2` WITHOUT_BRC路径、PTO ISA
`03e45c4bda48a6909feb239f0acc45f04176c2a1`和第五批生成的`qk_pv.cpp`，
发现512列求和的外层分组不同：Native在当前M>1条件下执行三层连续8元
`BlockReduceSum/vcgadd`，当前PTO通用路径先合并64列分组，再`vcadd64`。
128列两边都先逐元素加前后64列，最后分别为两层`vcgadd`与`vcadd64`。
这只是已确认的实现差异，尚未证明是剩余8个heads差异的根因。

本轮新增两项独立诊断，均不修改正式源码：

| 诊断 | 验证内容 | 静态结果和当前状态 |
| --- | --- | --- |
| `task_20260928_103646_133420325347` | 导出inverse RoPE前BF16 heads、最终FP32 m/l/o；Native重复同一只读attention op并切换LSE，检查输出不变 | 完整CPU编译通过；等待NPU。必须先通过所有原有阶段、cache、最终输出及guard与正式路径逐位一致的门槛。 |
| `task_20260928_104327_150481722851` | 固定Native TopK，仅把128/512的chunk sum改成按8元分组归约，检查heads及最终输出 | AST逆变换、四组CPU分组测试、完整CPU编译通过；等待NPU。候选`cab8efeb…`，尚未合入正式路径。 |

第二项候选使用masked `vcadd8`模拟原生连续8元分组；`vcadd8`与`vcgadd8`
内部舍入尚未证明相同，不能只凭外层形状一致认定逐位等价。CPU正概率临界值
证明改动会产生2 FP32 ULP差异，但它不是设备精度结果。

诊断编译失败也保留：第一版m/l/o的3D外参reshape无法推导跨函数metadata，
改成独立T×H维度的2D直接外参后编译成功；归约候选先遇到Tile参数被当作
Tensor元数据，再遇到128分支仍检查512 reshape，最终生成显式128/512两个
函数消除无效类型分支。数值主体均有可逆AST或文本检查，未修改编译器/环境。
另有独立FP32 `Log(l)+m` helper已编译，只有来源报告的complete、probe_valid、
native_lse_repeat_valid全部通过才允许读入，不从LSE反推Native sum的逐位结果。

本节更新时227的16卡由另一任务占用，上述任务均未取得设备结果。
完整源码出处、候选manifest及CPU检查保存在工作区
`native-align-20260928/native-softmax-reduce/`，阶段探针在同级目录。

## 2026-09-28：当前 TND 与 nalinaly 同逻辑输入精度对比

当前运行源码为 `bdf2cf0fb`，参考为 nalinaly
`0d1b8ea7292ffefc6e90cac0e2f4308749be53db` 的精度版本。
单卡任务 `task_20260928_105743_222713730819` 完成，exit=0；
`complete=true`、`probe_valid=true`，诊断副本与正式路径的 output、
cache allocation 及输出 guard 检查通过。参考使用已完成的 BSH 对拍原始张量。

### 配置与可比性

- 模型：DeepSeek-V4-Flash-0731-w8a8，第2层真实 C4 权重；合成 hidden/history。
- TP=1，B=4，每请求 S=6，总 T=24，起始位置8191；单层 eager 精度诊断。
- 当前环境：CANN9.2.0-beta.2、Torch2.10.0、Torch-NPU2.10.0.post4、vLLM0.29，物理页128。
- 参考环境：CANN9.0.1、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1，物理页32。
- 当前 PyPTO `54957491ede07ad5d5015f5e69874f367113cf45`，
  Simpler `32dff953d07f6bd2aacab8532860f28aca6df931`；复用已有环境。
- Native 确定性 level1，`HCCL_DETERMINISTIC=true`。
- 按逻辑位置生成相同六类历史及 hidden，保留各自物理页布局。
  两边逻辑输入 SHA256 均为
  `eb841527b5fda671b680d4c30db61722816963a87cd53cc78c3af5e3e519fd3d`。
- 额外读取保存的 Native 张量交叉比较：两环境的 Q、QR、QR scale、
  inverse RoPE 后 heads、最终 output 均逐元素一致。
  因此本用例的已比较 Native 基准一致，但不宣称两环境所有算子普遍等价。

### 相对各自 Native 的结果

| 指标 | 当前 TND | nalinaly |
| --- | --- | --- |
| 最终 output relative L2 | 0 | 0.000743908035733065（0.0743908%） |
| 最终 output 不同元素 | 0 / 98,304 | 4,853 / 98,304 |
| 最终 output max abs | 0 | 0.00390625 |
| heads relative L2 | 1.4583752610230643e-6 | 1.2195475738471703e-5 |
| heads 不同元素 | 5 / 786,432 | 1,115 / 786,432 |
| heads 最大 ULP，全部有限元素 | 1 | 93 |
| raw KV 不同元素 | 0 / 12,288 | 1 / 12,288，1 ULP |
| Q、QR、QR scale | 逐位一致 | 逐位一致 |
| 其余五类 cache/state 有效写入 | 逐位一致 | 逐位一致 |
| 同一 heads 经两边 O-proj | 逐位一致 | 逐位一致 |

两边最终输出均通过固定 `rtol=atol=1e-2`；当前输出进一步达到逐位一致。
当前 heads 仍有5个元素各差1 BF16 ULP，本组未传播到最终 output，
不能据此宣称 attention 内部已完全对齐。nalinaly 的全元素 heads 最大 ULP
为93；若只看其报告中 reference abs>=0.01 的元素，则最大 ULP 为1，
不可混用过滤后与未过滤的指标。

本轮没有测性能、Graph replay、非等长或整模型；不得覆盖之前 B16/T60
等用例仍有输出误差的结论。当前分支只在这组同逻辑输入上更接近 Native。

原始证据为 `canonical_manifest.json`、`result.json`、`native_stages.pt`、
`csa_stages.pt`、`cross_native_comparison.json`；独立 runner 位于工作区报告
`native-align-20260928/nalinaly-canonical-comparison/`，通过输入哈希断言，
未改动正式 kernel。环境预检通过；没有重新安装或编译框架依赖。

## 性能优化提交索引

| 提交 | 实际改动 | 验证结果与限制 |
| --- | --- | --- |
| `60dd8eb18` | O-B token tile 128→32。 | 三形状输出保持一致；早期跨任务耗时下降有Native漂移，不能全部归因于改动。 |
| `f993dbddc` | Indexer scale直接广播，删除全1张量及乘法。 | 精度不变；早期跨进程3.2%/3.9%收益未被同进程复测确认，仅保留冗余计算简化。 |
| `cd9484671` | Q-A独立行块最多分两组，完整K累加不变。 | 四形状跨版本逐位一致；T18/T60/T96配对中位快1.01%/2.31%/0.63%，T24无收益。 |
| `f2bfcbee0` | main/inner compressor投影K循环两级流水。 | 四形状跨版本逐位一致；相对QR两组版配对中位快1.38%～3.47%，仍慢于同轮Native。 |
| 本节2026-09-29源码提交 | Attention的m/l/O/alpha留在四份8-head UB状态中。 | 四形状输出/cache逐位保持；每组24/24轮更快，延迟中位数下降1.79%～3.09%，仍慢于Native。 |

## 2026-09-28：O-B M32 单因素性能优化

基线源码 `bdf2cf0fb`，候选只将 `PROJ_B_MM_T_TILE` 从128改为32。
完整CPU编译通过；任务 `task_20260928_143748_399861312872` 完成，exit=0。
沿用上述9.2环境、真实层2权重、seed62、起始8186、页128、确定性Native。
同卡串行执行三种形状，每实现Graph预热50次，20轮×200次replay，交替顺序。
未改变K累加、原生BF16边界、全行量化、反量化和scope。

| 形状 | 旧CSA ms | M32 CSA ms | 原始耗时下降 | 旧Native ms | 本轮Native ms |
| --- | --- | --- | --- | --- | --- |
| B4/T24 | 0.703966 | 0.669310 | 4.9% | 0.601740 | 0.578554 |
| B4/T18 | 0.667467 | 0.627442 | 6.0% | 0.578190 | 0.593557 |
| B16/T60 | 0.978992 | 0.967355 | 1.2% | 0.722948 | 0.713423 |

三组保存的Native与CSA输出分别和优化前逐位一致；两个B4与Native输出逐位
一致，B16仍为994个不同元素、relative L2=0.00021757883036598835。
Graph精度也保持相同结果。候选没有新增输出精度误差。

上述性能为不同任务之间的比较，Native存在漂移，不能将全部降幅归因于M32；
B4有收益迹象，B16收益未确定。用户批准先合入此单因素改动，后续需同任务
交替复测。候选没有改变外部TND输入，也没有把服务padding固定到S6。

证据：独立报告 `native-align-20260928/oproj-m32/`；
远端每组 `result.json`、`outputs.pt`、`baseline_exact.json`。
候选文件SHA256为
`3917b1842ac87915c15caf4b31dd9e256b4d3bd28fca05f1694a3a2e94fa4b70`。

## 2026-09-28：M32 后批量清零候选，无明确收益

基线为 `60dd8eb18` 的M32；候选只把O-B尾部逐行清零改成每次最多8行，
以valid shape限制最后一块，保持原生全行统一量化。参考pypto-lib main
`fbe92bf` 的批量清零思路，没有移植其按组量化路径。
384种token数的清零覆盖范围静态检查及完整CPU编译通过。

首任务 `task_20260928_154525_302949013989` 在环境预检失败：runner切换源码
时覆盖PYTHONPATH，遗漏CANN acl路径，未进入NPU计算。保留CANN路径后重提
`task_20260928_154621_306561918937`，完成exit=0。
同任务单卡串行运行M32/pad8，三种形状的先后顺序交错，每实现仍为Graph
预热50次、20轮×200次replay。其余模型、层、输入、软件环境同上一节。

| 形状 | M32 CSA ms | pad8 CSA ms | pad8相对变化 | M32轮Native ms | pad8轮Native ms |
| --- | --- | --- | --- | --- | --- |
| B4/T24 | 0.671889 | 0.665590 | 快0.94% | 0.565822 | 0.589269 |
| B4/T18 | 0.635593 | 0.638367 | 慢0.44% | 0.569270 | 0.562874 |
| B16/T60 | 0.975533 | 0.993985 | 慢1.89% | 0.719935 | 0.742319 |

六个运行的保存eager输出均与此前M32基线逐位一致，Native保存输出也一致。
Graph两个B4均与Native输出一致；B16均为994个不同元素，relative L2为
0.00021757883036598835。没有保存跨版本Graph输出张量，不能将相同误差
指标称为跨版本Graph逐位比较。

结论：候选无一致明确收益，同卡不同进程的Native仍有漂移；不足以证明
pad8本身造成加速或退化。保留M32，pad8不合入，避免为未证实收益增加代码。
下一步应以当前版本profile确定主要耗时，而不是扩大清零分块盲试。
候选与完整原始记录位于独立报告 `native-align-20260928/oproj-pad8/`。

## 2026-09-28：最新参考实现与 M32 分阶段 profile

参考pypto-lib main `fbe92bf`、nalinaly分支
`e58ddc94d77c93a0a8a85ab2db4bd773da9dff84`。只更新参考引用，未切换工作树。
两者的NZ权重布局、分块和调度可借鉴；按组量化、反量化乘法顺序等数值路径
与当前Native 0.29契约不同，不能直接替换当前已对齐的统一量化。
参考实现的B16/S6对应T96，本分支变长B16/T60不能只按B16标签比较。

参考仓库 `csa_qa_matrix_20260928/README.md` 的正式记录确有加速：
源码30f2b228性能算子，TP1、DP=EP16、DSpark出5验6、EPLB关闭、NZ mode2、
确定性关闭、atomic0；七档CSA profile均值降低9.45%～23.30%，无profiler
整模型forward均值降低1.04%～7.11%。这是不同执行路径/负载的参考证据，
不能直接写作本分支收益；其中两档P95仍未过其尾部验收。

当前M32计时泳道任务 `task_20260928_161918_122046517983`、独立依赖图任务
`task_20260928_162538_139566812096` 均exit=0。两次kernel id/name映射一致。
计时未同时开启dep-gen；以下为DFX执行区间包络，包含调度空隙且有重叠，
不能相加当整层延迟，也不包含外部原生KV投影和完整Graph调度边界。

| 区间 | B4/T24 μs | B16/T60 μs |
| --- | --- | --- |
| 首任务至TopK发布结束 | 290.96 | 409.54 |
| QK/PV | 88.78 | 154.52 |
| O-A投影 | 59.68 | 81.58 |
| 统一量化 | 17.64 | 18.42 |
| O-B投影 | 36.46 | 59.92 |
| O-B反量化 | 8.24 | 11.72 |
| 首末AICore执行跨度 | 555.70 | 814.88 |

B16尾部清零仅3.48μs，发生在Attention前，解释了上一候选收益不明确。
主要空间在TopK前投影/压缩/Indexer链、Attention和O-proj，不能继续仅调清零。
原始证据：独立报告 `m32-profile/evidence-b4/`、`evidence-b16/` 的泳道JSON、
`mapped_summary.json`，以及对应独立dep-gen输出。

## 2026-09-28：O-B K1024 单因素，无明确收益

基线运行代码 `60dd8eb18`；候选只把O-B INT8 K分块256改为1024，
不改变M32、全行统一量化和INT32累加/反量化语义。CPU完整编译通过。
任务 `task_20260928_163529_194427229698` 完成exit=0；模型、软件环境、
seed62、start8186、页128与前述M32一致。单卡同任务各实现单独进程，
三组先后顺序交错；每实现Graph预热50次、20轮×200次replay。

| 形状 | M32 CSA ms | K1024 CSA ms | 候选相对变化 | M32轮Native ms | 候选轮Native ms |
| --- | --- | --- | --- | --- | --- |
| B4/T24 | 0.596771 | 0.610168 | 慢2.25% | 0.589223 | 0.597471 |
| B4/T18 | 0.568169 | 0.571022 | 慢0.50% | 0.566344 | 0.574954 |
| B16/T60 | 0.919086 | 0.920265 | 慢0.13% | 0.755743 | 0.733738 |

本轮新增保存Graph输出：三组Native/CSA的eager和Graph输出分别在两个版本
之间逐字节一致。两个B4输出等于Native；B16仍为994个不同元素，relative L2
为0.00021757883036598835，候选未引入新的差异。

结论：无明确加速，不合入。不同轮次Native有漂移，且本轮M32本身也比上一轮
快，不能把跨任务变化归因于候选。证据为独立 `oproj-k1024/` runner、各组
`result.json`、`baseline_exact.json`、`graph_outputs.pt` 及
`oproj-k1024-graph-comparison.json`。

## 2026-09-28：NZ 权重布局候选，工具链阻塞

基线运行代码仍为 `60dd8eb18`。借鉴参考实现的NZ布局，独立O-B候选在
prepare_weights中一次性打包INT8权重为分组NZ字节，kernel用NZ视图读取。
因当前编译器的切片非负性证明约束，独立incore任务按列块循环token tiles；
不改变O-A舍入、全行统一量化、INT32求和及反量化顺序。真实尺寸CPU
打包/解包检查及完整静态编译通过，独立审查未发现生产路径算术/索引问题。

- `task_20260928_165748_281200619207`：exit=1。Native/M32运行完成，
  候选在入口预检失败：打包后的Torch-NPU tensor为format30，PyPTO要求基础格式。
- `task_20260928_170640_303052232130`：使用独立nz-v2目录，打包后调用
  已有ND格式转换。入口预检通过，但首次JIT时报
  `ValueError: Lowering changed the kernel entry parameter ABI`，exit=1。
  当前PyPTO `54957491ede07ad5d5015f5e69874f367113cf45` 在生成kernel
  artifact时拒绝入口参数shape/dtype/direction变化；此次NZ候选未执行。
- 另一个仅O-A BF16 NZ候选在静态编译阶段被NZ切片非负性证明检查拒绝，
  没有提交NPU任务。需要先调整索引表达，再验证同类入口ABI问题。

结论：这些是现有工具链下的接入阻塞，不是候选精度失败，也没有有效候选
性能结果。没有绕过ABI校验、改共享工具链或合入正式源码。
独立记录在 `native-align-20260928/oproj-nz/`、`oproj-a-nz/`；
各任务失败JSON和首次编译日志均保留。后续先验证不依赖NZ入口的O-A分块。

## 2026-09-28：O-A token 分块32，无明确收益

基线为 `60dd8eb18`；独立候选仅将 `PROJ_A_ROW_TILE` 128改为32，
O-B仍为32。完整CPU编译通过；任务 `task_20260928_173652_173262922568`
完成exit=0。沿用前述三形状、真实权重/seed62、start8186、确定性Native、
Graph预热50次、20轮×200重放和同任务交错运行顺序。每个进程额外记录uptime。

| 形状 | 基线CSA ms | O-A32 CSA ms | 候选相对变化 | 基线轮Native ms | 候选轮Native ms |
| --- | --- | --- | --- | --- | --- |
| B4/T24 | 0.681069 | 0.677581 | 快0.51% | 0.599203 | 0.593340 |
| B4/T18 | 0.645369 | 0.651393 | 慢0.93% | 0.581692 | 0.568479 |
| B16/T60 | 0.980308 | 0.996995 | 慢1.70% | 0.737177 | 0.739439 |

三组eager和Graph的Native/CSA输出分别与基线逐字节一致。没有降低精度；
也没有稳定加速，因此不合入。缩小token分块不一定减少实际设备工作，
还会增加较大T下的任务数量，不能仅依据名义补齐行数估算收益。
证据：`oproj-a32/evidence/` 三组JSON及Graph逐位比较汇总，候选文件哈希
保存在同目录上一级 `manifest.json`。

## 2026-09-28：Indexer scale 直接广播

运行基线 `60dd8eb18`。候选只将Indexer反量化的
`row_expand_mul(full(ones), qr_scale_tile)` 改为
`row_expand(acc_fp32, qr_scale_tile)`，删除全1中间量与一次乘法。
广播目标只提供shape/dtype/valid shape，返回新的SSA值，不覆盖accumulator；
后续仍先乘两个FP32 scale，再乘转换后的accumulator、舍入BF16。
独立审查通过；生成C++的两处TROWEXPANDMUL已替换为TROWEXPAND。

任务 `task_20260928_175157_232707421950` 完成exit=0，沿用同样的三形状、
模型层/输入/环境、Graph预热50次、20轮×200重放、形状间交错版本顺序。
本节代码改动和测试记录一起提交；CPU相关合约45项通过。

| 形状 | 基线CSA ms | 广播CSA ms | 原始耗时变化 | 基线轮Native ms | 广播轮Native ms |
| --- | --- | --- | --- | --- | --- |
| B4/T24 | 0.687337 | 0.665410 | 快3.19% | 0.577782 | 0.595140 |
| B4/T18 | 0.657507 | 0.631852 | 快3.90% | 0.569393 | 0.580004 |
| B16/T60 | 0.979919 | 0.984714 | 慢0.49% | 0.732763 | 0.726619 |

三组eager和Graph的Native/CSA输出分别跨版本逐字节一致。B4仍与Native
输出一致；B16仍保留基线994个不同元素和relative L2=0.00021757883036598835。
这次没有消除既有B16数值差异，也没有新增差异。

两个B4的运行顺序相反，均见延迟下降；但Native仍有漂移，不将单任务结果
称为跨机器/负载稳定收益。B16的逐轮范围基线0.975987～0.984739ms，候选
0.978708～0.991191ms，有重叠，尚无收益。保留这项减少冗余计算的简化，
后续继续测量；不能宣称当前已整体快于Native。
证据：`indexer-broadcast/evidence/`、候选 `manifest.json`、远端
`oproj-idx-broadcast-graph-comparison.json`。

## 2026-09-28：Q/O-proj 扩展直接广播

以 `f993dbddc` 的Indexer广播为基线，在独立候选中将Q和O-B反量化的
全1乘法改为直接col/row expand。scale乘法、INT32转FP32和最终BF16
边界保持不变，独立静态审查及完整CPU编译通过。
任务 `task_20260928_181217_30624869476` 完成exit=0；三形状eager和Graph
的输出均与基线逐字节一致。沿用50预热、20×200 Graph计时，其余条件同前。

| 形状 | Indexer版CSA ms | 扩展版CSA ms | 变化 | Indexer轮Native ms | 扩展轮Native ms |
| --- | --- | --- | --- | --- | --- |
| B4/T24 | 0.605292 | 0.610847 | 慢0.92% | 0.580369 | 0.580991 |
| B4/T18 | 0.576388 | 0.580669 | 慢0.74% | 0.573037 | 0.568959 |
| B16/T60 | 0.926923 | 0.917947 | 快0.97% | 0.739072 | 0.725393 |

没有一致收益，扩展版暂不合入。尤其本轮Indexer基线自身较上一轮明显变快，
跨进程漂移大于这些小优化的差距。因此继续用同一进程中的Native、M32、
Indexer版、扩展版四路Graph交替测量，优先采用该受控复测的结论。
证据：`scale-broadcast/evidence/`、`manifest.json` 和Graph逐字节比较汇总。

## 2026-09-28：同进程四路复测，修正小优化收益判断

为排除前述跨进程漂移，改用同一进程同时注册Native、M32基线
（`60dd8eb18`）、Indexer直接广播（`f993dbddc`）和Q/O扩展广播。
三个CSA模块采用独立Python package、JIT对象和custom op命名空间，
共享同一权重、hidden及metadata，各有独立cache和output。
运行时记录源码路径、文件SHA256与调用计数，禁止静默回退Native。
Q/O扩展仍为独立实验，没有合入正式分支。

测试保持真实C4层2权重、seed62合成hidden/history、TP1、页128、
start_pos=8186和确定性Native；CANN9.2、torch_npu2.10.0.post4、
vLLM0.29及PyPTO/Simpler提交与前述测试相同。
B4/T24长度为[6,6,6,6]，B4/T18为[3,4,5,6]，B16/T60为
[1,2,3,4,5,6,1,2,3,4,5,6,3,4,5,6]。
每形状50次Graph预热，四路的全部24种排列各测200次replay；
每种实现处于各顺序位置均为6次。cache重置、同步和精度检查在计时区外。

- 首次任务 `task_20260928_183448_312163832420`：exit=1。
  三个CSA已执行且eager输出跨版本逐位一致；随后诊断脚本在没有forward
  context时调用compressor metadata producer失败。该失败属于测试脚本，
  没有候选性能结果。保留原始脚本及失败记录。
- 修正后的独立 `ab-v2.py` 仅为该metadata调用设置forward context，
  没有删除或放宽任何精度/稳定性断言。
- 任务 `task_20260928_184109_33443831159`：exit=0，三组全部完成。

Graph每轮平均延迟的24轮中位数，单位ms/attention forward：

| 形状 | Native | M32基线 | 当前Indexer广播 | Q/O扩展候选 | 当前版相对Native |
| --- | --- | --- | --- | --- | --- |
| B4/T24 | 0.579327 | 0.660354 | 0.657840 | 0.653141 | 慢13.55% |
| B4/T18 | 0.596191 | 0.647539 | 0.647230 | 0.644288 | 慢8.56% |
| B16/T60 | 0.719525 | 0.990406 | 0.987625 | 0.986830 | 慢37.26% |

同轮配对变化也仅为小幅趋势，并非每轮都改善：

| 形状 | Indexer相对M32配对变化中位数 | Indexer更快轮数 | 扩展相对M32配对变化中位数 | 扩展更快轮数 |
| --- | --- | --- | --- | --- |
| B4/T24 | 快0.186% | 15/24 | 快1.145% | 18/24 |
| B4/T18 | 慢0.061% | 12/24 | 快0.525% | 17/24 |
| B16/T60 | 快0.317% | 13/24 | 快0.292% | 16/24 |

**结论修正：先前跨进程观察到的Indexer 3.19%/3.90%下降未被本轮确认。**
保留已合入的冗余计算简化，但不能将其宣传为稳定3%～4%加速。
Q/O扩展的幅度仍小且与前轮方向不一致，暂不合入；当前还没有达到快于Native
的目标。不能把这里的同进程绝对耗时与不同进程历史数据直接相减归因。

精度与工作负载稳定性：

- 三个CSA的eager输出、Graph输出、六类有效cache/state写入跨版本逐位一致。
- 四路各自的首次Graph replay与全部重复计时之后，输出和六类有效写入均
  逐位一致；Native独立重复检查也通过。输出guard完整，无NaN遗留。
- 两个B4的输出与Native逐位一致；B16保留相同的994/245760个差异元素，
  relative L2=0.00021757883036598835，max abs=0.00390625。
  本轮优化没有增加误差，也没有消除既有B16误差。
- 仅验证单层、所列三形状及输入；没有证明128K、整模型或DP16性能。

原始证据保存在独立报告 `same-process-broadcast/evidence/`：各形状
`result.json`（含24轮原始延迟、调用计数、源码来源）、`pre_timing.json`、
`replay_checks.json`，以及独立runner/hash manifest。远端保留Graph输出tensor。

### 参考实现与后续优化优先级

本轮只更新参考引用：pypto-lib main为`fbe92bf`，nalinaly分支为
`e58ddc94d77c93a0a8a85ab2db4bd773da9dff84`，未改参考工作树。
[nalinaly七档记录](https://github.com/nalinaly/vllm-ascend/blob/e58ddc94d77c93a0a8a85ab2db4bd773da9dff84/tests/pypto_test/results/csa_qa_matrix_20260928/README.md)
明确报告CSA profile均值快9.45%～23.30%，正式forward均值快1.04%～7.11%。
该结果使用其性能算子、TP1/DP=EP16、EPLB关闭、NZ mode2、确定性关闭；
不能当作我们确定性单层TND配置的受控对照，也不否认它确有加速结果。

后续重点应从广播/清零等小改动转向已观察到的主要耗时：

1. TopK前的投影、压缩与Indexer链：检查任务分组及依赖造成的启动间隙，
   参考其QR/KV自适应分组；保留当前full-K、INT32求和及BF16舍入边界。
2. Sparse attention跨query流水与访存复用：保留真TND寻址、mask、
   softmax与inverse RoPE数值边界，逐项对拍后再判断收益。
3. O-proj权重NZ布局：当前先被PyPTO入口ABI检查阻塞，必须用最小案例
   验证物理布局与入口契约，不能绕过校验或声称已有性能收益。

参考性能版的按组量化和atomic拆K不直接移植；每个改变都必须维持当前
逐位基线，并使用同进程交替Graph对拍判断性能。

## 2026-09-28：Q-A 独立行块分组

基线`f993dbddc`（分支HEAD `50ec05e28`仅增加文档）。Q-A原先按N列分8个
任务，每个任务串行处理所有M行块。候选将相互独立的行块分为最多2/3组，
仍保持每块M16/M64、N128、K256、完整FP32累加及后续BF16/RMS/量化边界。
参考QR/KV任务分组方向，没有引入split-K或atomic。
T=1…512的行覆盖检查通过：无遗漏、无重叠；两个候选完整静态编译通过，
独立审查通过。混合dense/tail仍可能出现空任务，因此额外测试T96。

任务`task_20260928_211439_357380519015`全部完成exit=0。
继续使用前节同进程Native/base/g2/g3、独立cache/custom op、确定性Native、
24种排列×200次Graph replay及50次预热；同模型层、seed62、start8186、
TP1、页128和CANN/PyPTO/Simpler环境。新增B16/T96为16个长度6的请求。

| 形状 | Native ms | 基线CSA ms | 两组CSA ms | 三组CSA ms | 两组配对变化中位数 | 两组更快轮数 |
| --- | --- | --- | --- | --- | --- | --- |
| B4/T24 | 0.591061 | 0.606660 | 0.608043 | 0.609270 | 慢0.117% | 8/24 |
| B4/T18 | 0.570846 | 0.582628 | 0.576633 | 0.581930 | 快1.008% | 24/24 |
| B16/T60 | 0.728870 | 0.922622 | 0.901159 | 0.935006 | 快2.310% | 24/24 |
| B16/T96 | 0.876143 | 1.095672 | 1.089113 | 1.095707 | 快0.627% | 24/24 |

两组的T60配对改善范围1.789%～2.655%，T18为0.587%～1.377%，
T96为0.094%～1.130%；T24范围从快0.649%到慢0.810%，无明确收益。
三组T60每轮均回退，配对中位慢1.470%，不保留。
两组在T60每任务最多两个行块，三组也仍为两个；增加第三组并不减少最长任务
的块数。该事实不能单独证明设备竞争是回退原因。

四形状eager/Graph输出及六类有效cache写入均跨版本逐位一致，四路各自的
重复replay输出/cache稳定性和输出guard也通过。两个B4仍与Native输出
逐位一致；T60仍为relative L2=0.00021757883036598835。新增T96的基线和
两候选均为relative L2=0.0013848491075071758，属于已有基线数值差异，
不是本次分组引入；不能宣称新增形状与Native逐位对齐。

选择两组方案，拒绝三组；保留T24无收益和大形状仍慢于Native的限制。
原始证据：独立报告`qr-row-groups/`中的两份候选源码/patch、hash manifest、
checklist及四形状`evidence/*/result.json`、`pre_timing.json`和
`replay_checks.json`。CPU相关合约及提交检查见本提交验证记录。

## 2026-09-28：Compressor 调度与两级流水

基线为已推送的QR两组版`cd9484671`。两个单因素候选：

- gate：仅将inner compressor调用的跨main projection依赖换为共同的
  `late_dep`。两投影各读自己的权重、写独立scratch；各cache池内部
  projection/pool/state commit/key/scale写入依赖均保留。
- pipe：仅将main/inner原生对齐投影的K循环`stage=1`改为`stage=2`，
  保留基线gate、128列K块、旋转起点、首次初始化和完整累加顺序。

两者独立审查、精确预检和完整CPU编译通过。pipe首次编译命令在预检阶段因
SOURCE_ROOT仍指gate而被拒绝，未开始编译；改为独立shell并正确设置来源后
通过，原失败日志保留。生成C++审查确认按0,2,…30遍历，每轮顺序执行两个
128列块；初始化、Acc别名和MTE2/MTE1/M/FIX同步没有明显异常。
生成代码的静态L1地址跨度main为48KiB、inner为24KiB；这是静态地址检查，
不是运行期峰值。当前工具链没有生成memory_after报告。

任务`task_20260928_213852_293991611`完成exit=0。仍使用相同环境、权重层、
seed62/start8186、TP1/页128、确定性Native和四路同进程Graph协议：
50预热，全部24种排列×200次replay，四形状及长度同前节。

| 形状 | Native ms | QR两组基线 ms | gate ms | pipe ms | pipe配对变化中位数 | pipe更快轮数 |
| --- | --- | --- | --- | --- | --- | --- |
| B4/T24 | 0.578207 | 0.671142 | 0.656283 | 0.650106 | 快2.990% | 24/24 |
| B4/T18 | 0.575525 | 0.634451 | 0.636519 | 0.627368 | 快1.383% | 22/24 |
| B16/T60 | 0.748123 | 0.970542 | 0.952146 | 0.945582 | 快2.546% | 24/24 |
| B16/T96 | 0.876179 | 1.174070 | 1.170050 | 1.133010 | 快3.471% | 24/24 |

pipe的T60每轮均改善1.235%～4.010%，T96改善2.584%～4.690%。
gate配对中位变化依次为快2.173%、慢0.239%、快1.669%、快0.520%，
对应更快轮数24/7/23/19。其收益不够一致，暂不合入。

四形状的候选eager和Graph输出、六类有效cache写入均与基线逐位一致；
四路各自重复replay稳定性及输出guard全部通过。B4与Native输出逐位一致，
T60/T96仍保留前节基线relative L2=0.0002175788/0.0013848491。

选择两级流水方案，CPU相关合约45项通过。此处收益是相对QR两组版，不能
与前一任务百分比直接相加；下一轮放入本轮开始前的`f993dbddc`运行代码，
直接测累计收益及叠加gate的效果。证据在独立报告`compressor-schedule/`：
候选源码/hash、生成C++和静态地址汇总、四组完整JSON及逐位/稳定性结果。

## 2026-09-28：最终累计收益与调度叠加复测

任务`task_20260928_220739_13491017226`完成exit=0，四形状全部通过。
同进程四路为Native、优化前`f993dbddc`（`50ec05e28`同运行代码）、
已合入`f2bfcbee0`（QR两组+两级流水）、额外叠加解除main→inner gate。
保持上述相同模型层/输入/软件环境/确定性设置、50预热、24种排列×200次
Graph replay和独立cache/custom op。此次直接复测累计收益，不相加前两轮百分比。

| 形状 | Native ms | 优化前 ms | 已合入 ms | 叠加gate ms | 已合入中位耗时下降 | 已合入相对Native |
| --- | --- | --- | --- | --- | --- | --- |
| B4/T24 | 0.569836 | 0.660682 | 0.643877 | 0.650972 | 2.54% | 慢12.99% |
| B4/T18 | 0.567515 | 0.644459 | 0.633375 | 0.633897 | 1.72% | 慢11.61% |
| B16/T60 | 0.729002 | 0.973153 | 0.955365 | 0.950072 | 1.83% | 慢31.05% |
| B16/T96 | 0.881827 | 1.187269 | 1.145190 | 1.138179 | 3.54% | 慢29.87% |

表中下降率由两版本各自24轮中位耗时相除计算。逐轮配对另行检查：
已合入版相对优化前的变化中位数依次为快2.799%/1.561%/1.822%/3.502%，
更快轮数24/22/24/24；T60每轮改善1.237%～2.819%，T96为2.387%～4.537%。
这些是当前单层输入下的收益，不是整模型或跨机器吞吐结论。

叠加gate相对已合入版的配对变化为慢0.922%、快0.012%、快0.550%、快0.716%，
更快轮数5/12/20/21。小形状回退、大形状增益不足1%，不合入该附加改动；
正式分支保留两组Q-A和两级流水，没有删除跨main的gate。

精度再次核验：

- 三个CSA版本的eager输出、Graph输出、六类有效cache/state写入逐位一致。
- Native和三个CSA各自从首次replay到全部计时结束，输出/cache均逐位稳定，
  guard检查通过，无静默回退；三个CSA调用计数均为7（capture后replay不调用Python）。
- B4/T24与B4/T18对Native输出逐位一致。
- B16/T60仍有994/245760个差异元素，relative L2=0.00021757883036598835，
  max abs=0.00390625；B16/T96仍有10747/393216个差异元素，
  relative L2=0.0013848491075071758，max abs=0.0078125。
  本轮没有增加误差，也没有消除这些既有Native差异。

原始证据在独立报告`combined-validation/evidence/`，每组含完整24轮延迟、
来源文件SHA256、eager/Graph/cache和稳定性JSON；`summarize.py`重新执行所有
逐位结果断言后生成`summary.json`。最终仍慢于Native；后续需要继续优化主要
算子与访存，不能把当前小幅改善描述成已达到替换后快于原生的目标。

## 2026-09-29：Attention flash 状态留在 UB

### 来源、边界与实现

父提交 `7a0ddd686`，基线代码等价 `f2bfcbee0`；本节源码随测试记录一并提交。
唯一改动文件是 `dspark/decode_sparse_attn_csa.py`，已验证四组硬件记录的
8个来源文件SHA256与本地一致，另外7个文件与基线完全相同。
候选文件SHA256：

```text
7b69794f5d32683a33e2abbc1f0008c120bcf151c77d97235821a6888ea506bd
```

将每个AIV的累计m/l/O和alpha保留在四份独立8-head UB状态里，两个flash
chunk完成后才写最终GM结果。保留WIN128、CMP512分块、8-head规约、FP32
PV完整K累加及所有BF16舍入边界；PV、归一化、inverse RoPE、输出打包函数
与基线AST相同。保存alpha替代从同一FP32 max重新计算exp。

基线每个token的累计O有一次初始化store、两次merge load及两次merge store；
候选只有最终store，按64×512×FP32估算减少512 KiB显存访问。这是逻辑流量
估算，不是实测HBM流量，也不意味着整层延迟同比下降。

生成代码独立审查通过：UB最高分配端点147872 bytes（约144.4 KiB），低于
184 KiB预算；四份16 KiB O的地址分别是32800、98464、114912、131360，
两轮更新复用相同地址，没有存活期间覆盖；alpha跨PV等待保存完整。
两次PROB_READY/PV_READY协议保留，包括空chunk路径。
固定展开要求`H // 2 == 4 * SOFTMAX_HEAD_TILE`，源码增加相应断言。

### 环境与协议

227设备11，任务`task_20260929_094538_39932304516`，最终exit=0，四组complete=true。
CANN9.2.0-beta.2、Torch2.10.0+cpu、Torch-NPU2.10.0.post4、vLLM0.29.0，
PyPTO `54957491ede07ad5d5015f5e69874f367113cf45`，
Simpler `32dff953d07f6bd2aacab8532860f28aca6df931`，PTOAS0.63。
复用现有环境，没有重新构建或安装这些框架。

真实DeepSeek-V4-Flash-0731-w8a8的C4权重（layer=2）；固定seed62的合成hidden/history，
TP1、page128、start_pos8186（约8K历史）。请求长度依次为：

- B4/T24：`[6,6,6,6]`。
- B4/T18：`[3,4,5,6]`。
- B16/T60：`[1,2,3,4,5,6,1,2,3,4,5,6,3,4,5,6]`。
- B16/T96：16个6。

每个形状单独Python进程；Native、当前CSA、候选在同进程使用相同权重、输入、
metadata及三套独立cache/output。Native开启deterministic level1和
HCCL_DETERMINISTIC=true。50次预热；三臂6种排列重复4次，共24轮，每轮200次
Graph replay；每轮前重置cache并同步，重置时间不计入NPU事件。
独立算子名及调用计数确认实际CSA执行，无静默回退。

### Graph 实测

单位ms/attention forward。下降按中位数之比计算；配对变化负数表示候选更快。

| 形状 | Native | 优化前 | UB状态版 | 延迟下降 | 配对变化中位数 | 更快轮数 | 仍慢于Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B4/T24 | 0.594740 | 0.663641 | 0.651790 | 1.79% | -1.982% | 24/24 | 9.59% |
| B4/T18 | 0.581114 | 0.629893 | 0.614597 | 2.43% | -2.248% | 24/24 | 5.76% |
| B16/T60 | 0.722484 | 0.957322 | 0.927706 | 3.09% | -3.099% | 24/24 | 28.41% |
| B16/T96 | 0.894760 | 1.128545 | 1.108173 | 1.81% | -1.789% | 24/24 | 23.85% |

四组配对变化范围分别为[-3.988%,-0.270%]、[-4.119%,-1.413%]、
[-3.986%,-2.178%]、[-2.850%,-0.531%]；各24/24轮同方向。
接受这个单因素改动。仍未达到CSA快于Native的目标，不能将本轮数值与之前
不同进程/设备状态的绝对延迟直接相减，也不能叠加不同轮次的百分比。

### 精度与复现证据

- 四组eager/Graph输出及六类有效cache/state写入，候选与优化前全部逐位一致。
- 三臂从首次replay到全部计时结束，输出及cache逐位稳定，guard检查通过。
- B4/T24与B4/T18输出对Native逐位一致。
- B16/T60对Native仍有994/245760个不同元素，relative L2=
  0.00021757883036598835，max abs=0.00390625。
- B16/T96对Native仍有10747/393216个不同元素，relative L2=
  0.0013848491075071758，max abs=0.0078125。
  这些是既有差异，当前优化没有增加或修复它们；不是全场景逐位Native对齐证明。
- 本地相关45项UT通过；生成代码与同步关系经过独立评审。

原始JSON、24轮数据、runner、manifest、生成代码和编译日志保存于报告
`native-align-20260928/attn-ub-state-20260929/`；`summarize.py`先重新断言
所有精度/稳定性条件，再生成`summary.json`。远端对应日志前缀
`attn-ub-state-20260929-{b4t24,b4t18,b16t60,b16t96}`。

编译失败尝试也保留：Tile参数无法经过现有jit.inline；32-head局部assemble
产生ptoas TMOV形状错误，fillpad又导致pad模式不匹配，set_validshape未解决；
外部pl.inline宏也未被当前JIT解析。最终采用四份独立8-head状态并显式展开，
避免上述不支持的构造，没有修改框架或绕过ABI/形状校验。

本轮验证范围为单层、约8K历史的上述四组输入；未证明128K、整模型、DP16
吞吐或DSpark接收率。下一候选是按请求复用Indexer连续缓存，只复制INT8 key
和FP16 scale并保持原score计算；其额外scratch峰值与无效页语义尚待验证。

## 2026-09-29：Indexer 按请求连续缓存试验（未合入）

基线 `f894dba31`，独立候选只修改 `dspark/decode_indexer.py`。
候选 SHA256：`5de64ef953998694d6f19b4f56452cac0670cd8c2797173ad7d119c25bcf329d`。
没有重装或重新构建 vLLM、PyPTO、Simpler。环境沿用上一节。

### 实现和边界

每次 forward 等待 Indexer cache 写入，然后按请求把分页 INT8 key 和 FP16 scale
原始字节复制到连续临时缓存。该请求所有 token 共用，单次 score 迭代的 key
读取由 12 次 32 行改为 2 次 192 行。scale 仍在原来的位置转 FP32 并相乘；
INT32 dot、ReLU、2^-10、FP16 舍入、head coefficient、规约和 TopK 顺序不改。
不跨 decode 步复用旧数据，不修改外部 ABI、原生 metadata 或 cache 所有权。
每个请求保留自身可见长度和无效物理页掩码；末尾额外初始化 384 行 padding。

新增 scratch 限制为 32 MiB，超出则使用原始分页读取，实际只分配最小 dummy
缓存。B4/8K 新增约 1.21 MiB、B16/8K 约 4.82 MiB、B4/128K 约 16.44 MiB；
B16/128K 预计需要 65.76 MiB，触发回退。独立生成代码评审确认 bool 转 INDEX
为 0/1，分配量、repack/score 依赖以及 Indexer scope 生命周期正确。
Indexer 同 scope 载荷 T=96 约 136.03 MiB、T=384 约 145.03 MiB，
加满预算后仍低于每 ring 256 MiB（数值不含小额对齐及运行时元数据）。

CPU 检查覆盖 67,133,440 个有效 key 地址、单叶/多叶、页尾、非单调物理页、
0/-1 页和 scale 原始位。首次编译因 INT8 tile.full 不受 PTOAS 支持而失败；
改为 INT16 零字节 reinterpret INT8 后通过，没有修改框架。

### 8K Graph 对拍

任务 `task_20260929_102933_51760511213`，completed/exit=0。
四组配置、真实权重、seed、确定性 Native 和计时边界与上一节相同。
同进程三臂，24轮×200次 replay，50次预热，6种顺序交错；单位 ms/attention。

| 形状 | Native | 当前 CSA | 连续缓存候选 | 延迟增加 | 候选更快轮数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B4T24 | 0.585996 | 0.640942 | 0.662818 | 3.41% | 0/24 |
| B4T18 | 0.575244 | 0.637657 | 0.666514 | 4.53% | 0/24 |
| B16T60 | 0.715723 | 0.927691 | 0.972664 | 4.85% | 0/24 |
| B16T96 | 0.909512 | 1.117698 | 1.138454 | 1.86% | 0/24 |

四组 eager/Graph 输出、六类有效 cache/state 均与当前 CSA 逐位一致；
所有计时 replay 后输出/cache 保持逐位稳定，guard 通过。
B4 两组对 Native 输出逐位一致；B16/T60 与 B16/T96 的既有 Native 差异
保持不变，relative L2 仍分别为 0.00021757883036598835、0.0013848491075071758。
每组24轮候选都更慢，当前证据不支持合入。这不证明所有请求内复用方案都无效：
本候选额外增加完整 GM 整理及依赖，具体时间归因尚未做单阶段 profile。

### 长上下文补充

任务 `task_20260929_104004_12292178142`，227设备1，completed/exit=0。
B4/T18长度仍为 `[3,4,5,6]`；跨leaf组 start_pos=32765，128K组为131066。
B16/T60长度同8K节，start_pos=131066。相同三臂协议，24轮×100次Graph replay，
50次预热；不是32K/128K整模型吞吐测试。单位ms/attention。

| 形状 | Native | 当前 CSA | 候选 | 延迟变化 | 候选更快轮数 | 路径 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B4/T18，32K跨leaf | 0.618868 | 0.669838 | 0.687691 | +2.67% | 2/24 | 连续缓存 |
| B4/T18，128K | 0.781827 | 0.832203 | 0.887282 | +6.62% | 0/24 | 连续缓存 |
| B16/T60，128K | 0.996178 | 1.557310 | 1.546948 | -0.67% | 20/24 | 预算回退 |

三组候选与当前 CSA 的 eager/Graph 输出、六类有效cache/state逐位一致，
长replay稳定；三组输出也与确定性Native逐位一致。B16/128K由运行时长度和
32MiB判定回退原始分页读取，约0.67%的下降不作为连续缓存收益。
配对变化范围依次为[-0.457%,7.435%]、[3.639%,9.335%]、[-2.286%,0.506%]。

**结论：不合入此候选，正式kernel继续保留基线实现。** 实际启用完整连续缓存的
六组形状，中位数均退化约1.86%～6.62%。实验表明这套“先完整整理到GM再读”
方案没有收益，并不否定后续按请求分组、直接在L1复用key tile的可能性。
后者需要单独实现与逐位对拍，本轮未实现或声称有效。

全部七份原始JSON、逐轮计时、候选源码、CPU映射检查、生成代码、runner及
SHA256 manifest保存在报告 `native-align-20260928/indexer-repack-20260929/`。
远端日志同名前缀；`summarize.py`与`summarize-boundary.py`重新断言精度、
cache与稳定性后生成两份summary。失败候选保留于独立实验目录，没有覆盖正式源码。
本次提交仅记录文档；未重复执行与未改kernel无关的UT或安装框架。

## 2026-09-29：对照 nalinaly 的两个 Indexer 入口

参考远端更新到 `3b27c7fdf39f7c1c36d13a84745af6e84ce8fbdd`。
必须区分 `deepseek_v4_flash_dspark/decode_indexer.py`（精度入口）和
`deepseek_v4_flash_dspark_perf/decode_indexer.py`（性能入口）。前者保留192-worker
连续缓存；后者采用直接分页读取、不同score分块/调度及scale预取，不能把前者的
历史repack收益作为后者当前快于Native的原因。参考日志§235也明确旧mode1限制已失效。

### 候选一：192 worker + 每页一次读取（拒绝）

基于上一节独立repack候选，48改192 worker；128个INT8 key与FP16 scale尾
一次读16640字节，再做UB子视图。生成代码确认一次TLOAD，无额外切片复制。
保留128槽页、32MiB预算、padding/无效页规则和所有score算术。
与参考32-key/4160字节物理页不同，B4/8K仅76个搬运单元，192worker并不能全利用。
不将参考48→192的收益直接外推。

227设备1，task_20260929_105734_27424877218，completed/exit=0。
基线仍是f894dba31。环境、权重和四组8K输入同前，Native确定性开启；
同进程三臂、24轮×200次Graph、50次预热。单位ms/attention。

| 形状 | Native | 当前CSA | 192worker候选 | 延迟增加 | 候选更快轮数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| b4t24 | 0.588120 | 0.645300 | 0.713363 | 10.55% | 0/24 |
| b4t18 | 0.576957 | 0.617243 | 0.679871 | 10.15% | 0/24 |
| b16t60 | 0.734976 | 0.939884 | 1.013582 | 7.84% | 0/24 |
| b16t96 | 0.864526 | 1.119439 | 1.175992 | 5.05% | 0/24 |

四组eager/Graph输出及有效cache与当前CSA逐位一致，计时后重复replay稳定。
Native误差与前一节基线相同。候选四组24/24轮更慢，拒绝合入，不继续扩展此候选。
未做单因素48/192同场比较，不能将整个退化全部归因于worker数。
原始数据及源码在报告 `native-align-20260928/indexer-repack-nal-20260929/`。
候选源码SHA256：`cba0d39f74185d20809771b6e560f55bcc0d2a5a1f57b756e931935ae2bb30eb`。

### 候选二：原生物理页内64行合并

参考性能入口的直接分页读取方向，保留本分支128行key/2行scale的原生布局。
仅把key微块32行改64行，每个384行score tile从12次变6次加载；不要求不同物理页
连续，不新增repack/scratch/task，不改scale、matmul、舍入、head规约或TopK。
SCORE_TILE=384、lane192、两阶段pipeline保持不变。无效行仍由原有mask处理。

源码及生成代码独立review通过；每轮实际6次64×128 TLOAD，MTE2→MTE1依赖保留。
穷举67,133,440个有效地址一致；直接执行候选AST地址循环的72项CPU测试通过，
覆盖两种lane、第二leaf、非单调页、负页/零页、全部目的行恰好写一次。
这些CPU检查不代替NPU精度验证。

任务task_20260929_111222_308806130753，227设备1，completed/exit=0。
同前基线f894dba31、相同环境/权重/输入，确定性Native，同进程三臂24×200 Graph。
原始JSON沿用测试工具的 `repack` 键名表示本候选，实际没有执行repack。

| 8K形状 | Native | 当前CSA | 页内64行 | 延迟变化 | 候选更快轮数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| b4t24 | 0.586736 | 0.641727 | 0.649891 | +1.27% | 3/24 |
| b4t18 | 0.577393 | 0.620975 | 0.621581 | +0.10% | 8/24 |
| b16t60 | 0.743872 | 0.937794 | 0.935502 | -0.24% | 17/24 |
| b16t96 | 0.876296 | 1.114852 | 1.106903 | -0.71% | 22/24 |

四组eager/Graph输出及有效cache与当前CSA逐位一致，重复replay稳定；对Native差异
保持基线数值不变。8K的变化约−0.71%～+1.27%，方向混合，不能认定整体收益。
候选源码SHA256：`705b9470dc6beb3c475733b8046afd9e3a3e76a3429ed4e099399a8a1fcefb43`。

长上下文任务task_20260929_112227_325888610066，227设备0，completed/exit=0。
32K组start_pos32765，128K组131066；长度同前，三臂24×100 Graph、50次预热。
单位ms/attention。每轮重置cache在计时外进行。

| 形状 | Native | 当前CSA | 页内64行 | 延迟变化 | 候选更快轮数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B4/T18，32K跨leaf | 0.621522 | 0.610071 | 0.612197 | +0.35% | 9/24 |
| B4/T18，128K | 0.790461 | 0.765098 | 0.773910 | +1.15% | 2/24 |
| B16/T60，128K | 0.999733 | 1.486676 | 1.499749 | +0.88% | 0/24 |

三组eager/Graph输出及有效cache与当前CSA逐位一致，重复replay稳定；输出也与
确定性Native逐位一致。B16/128K配对变化全部为正（+0.745%～+1.093%），
不能将8K B16的小幅下降外推到长上下文。此轮B4/128K基线本身已快于Native，
但候选比该基线慢，不能把跨轮基线漂移算作候选收益。

**结论：两项候选均不合入，正式源码保持f894dba31的kernel。** 本次共11组NPU
配置全部完成，候选相对对应基线没有新增精度误差，但未取得可保留的整体性能收益。
参考性能版的Cube/score panel、流水及scale预取仍是独立因素，本轮未移植或声称验证。
不改变原生cache布局或舍入边界来换取表面速度；后续应按阶段profile选择下一项。

第二项全部源码、72项CPU测试、地址穷举、生成代码、原始JSON、逐轮统计及
manifest保留于 `native-align-20260928/indexer-page64-20260929/`。
本次正式提交仅更新测试文档，未修改运行代码、重装环境或合入失败候选。

## 2026-09-29：PV0 与第二段 softmax 重叠

基线 kernel f894dba31395c7ec31162fe5cb34e36f57a8d9c2（工作树 HEAD 78d1cff5d，后续均为文档提交）。
候选只把第二段 softmax 提到等待第一段 PV 之前；保存第一段独立 alpha，
保持 O 的两次累加顺序与 PROB0 → wait PV0 → PROB1 → wait PV1 协议。
AIC、概率 BF16 舍入、寻址和 cache 布局不变。参考原生流水的阶段重叠原则，
没有照搬原生跨迭代三阶段流水。

源码和生成代码独立 review 通过。UB 峰值 148000 bytes，原为 147872；
两段 alpha 分配不重叠。生成代码确认第二段 softmax 在 wait PV0 前，
PROB1 仍在第一段 O 更新后。CPU 执行实际 AIV AST 的 8 项比较覆盖空压缩段、
空窗口及局部 mask，概率、m/l/O 逐位一致；该检查不模拟异步设备可见性。

任务 task_20260929_115530_372158531370（8K，设备3）及
 task_20260929_120842_40521428773（长上下文），均 completed/exit=0。
环境沿用 CANN 9.2.0-beta.2、Torch 2.10.0、Torch-NPU 2.10.0.post4、vLLM 0.29.0、
PyPTO 54957491ede07ad5d5015f5e69874f367113cf45、Simpler 32dff953d07f6bd2aacab8532860f28aca6df931、PTOAS 0.63。
真实 DeepSeek-V4-Flash-0731-w8a8 C4 layer2 权重、seed62 合成 hidden/history、TP1、页128。
确定性 Native、独立 cache，同进程三臂交错测量，24轮，8K每轮200 replay、长上下文100，预热50。
8K start_pos8186，32K 32765，128K 131066；请求长度沿用前节 B4/T18、B16/T60 非等长配置。
表中为 ms/attention forward；变化是逐轮配对变化的中位数，因此不必等于两臂中位数之比。

| 形状 | Native | 原 CSA | overlap | 配对延迟变化 | 更快轮数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| b4t24 | 0.583092 | 0.661885 | 0.658026 | -0.483% | 16/24 |
| b4t18 | 0.572405 | 0.627482 | 0.619480 | -1.355% | 21/24 |
| b16t60 | 0.731901 | 0.939173 | 0.920757 | -1.953% | 24/24 |
| b16t96 | 0.877907 | 1.118298 | 1.096212 | -1.945% | 24/24 |
| b4t18_leaf_boundary | 0.625261 | 0.675652 | 0.674753 | +0.307% | 11/24 |
| b4t18_128k | 0.783074 | 0.825452 | 0.809041 | -1.523% | 17/24 |
| b16t60_128k | 0.977192 | 1.547039 | 1.533661 | -0.739% | 22/24 |

七组 eager/Graph 输出、六类有效 cache/state 与原 CSA 均逐位一致，计时后 replay 稳定。
对 Native 的精度保持不变：8K B16/T60 relative L2 0.00021758、B16/T96 0.00138485；
其余五组输出逐位一致。不能将“未新增误差”表述为所有负载均与 Native 逐位一致。

结论：保留该优化。8K B16 两组24/24轮更快，约1.95%；128K B16 22/24轮更快，
配对约0.74%。32K B4变化在噪声内，不声称有收益。当前 CSA 在部分负载仍慢于 Native，
不以跨任务绝对时间漂移计算收益，未验证整模型或16卡吞吐。

证据：`native-align-20260928/attn-pv-overlap-20260929/` 下保存 runner、生成代码、
CPU检查、原始JSON与逐轮汇总。候选源码 SHA256：`ba1a9b8edc2cf5380f18fc25bda88d63c19fc30277987557e1ff71ec1d22802a`。

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
