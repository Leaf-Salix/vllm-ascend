# DeepSeek V4 Flash PyPTO CSA 测试历史

本记录属于 `dev/pypto-dsv4-csa-tnd-main-20260924` 分支，覆盖从首次接入
`6c9d552a1` 到当前原生数值对齐系列的源码提交。更新日期：2026-09-28。
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
