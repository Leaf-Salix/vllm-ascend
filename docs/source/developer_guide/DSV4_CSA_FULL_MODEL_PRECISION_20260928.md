# 完整 DeepSeek-V4：Leaf、BSH 与原生精度对照（2026-09-28）

## 结论

四组完整模型均运行完成、退出码为0；Leaf与BSH的S6替换覆盖检查通过。
在四组都保存到的**94个相同token历史预测位置**上，Leaf与其原生对照的平均概率分布差异为2.4146%，BSH与其原生对照为2.3601%。两者的平均差距接近，但不能证明完整模型精度相等，更不能证明“仅差几个bit”。Leaf与BSH直接比较也存在差异。

这里的概率分布差异是 total variation（TV）：全部词表预测概率的绝对差之和除以2，0表示相同，100%表示完全分离。它不是回答错误率。

## 版本与工作负载

| 项目 | 本次记录 |
| --- | --- |
| Leaf 分支 | `dev/pypto-dsv4-csa-v0.25.1rc1-cann9.0.1` |
| Leaf 完整提交 | `c4383cfea0077f79edd68b4e6ed3950ab9a2242f` |
| BSH 参考分支 | `nalinaly/dsv4-flash-pto-v0.25.1rc1` |
| BSH 完整提交 | `0d1b8ea7292ffefc6e90cac0e2f4308749be53db`，`PTO_CSA_VARIANT=precision` |
| vLLM-Ascend 基线 | `v0.25.1rc1` |
| vLLM | 运行记录为 `0.25.1`；安装目录无 Git 元数据，生成的版本文件 commit_id=None，不臆造提交号 |
| Torch / Torch-NPU | `2.10.0+cpu` / `2.10.0.post2` |
| CANN / PTOAS | `9.0.1` / `0.63`；实际import和环境路径保存在各rank JSON中 |
| PyPTO HEAD | `54957491ede07ad5d5015f5e69874f367113cf45` |
| Simpler HEAD | `166852bfa658c259478b39e1991a7fd5e7379ac5` |
| 模型 | 完整 `DeepSeek-V4-Flash-0731-w8a8`，不是official-l3三层模型 |
| 并行 | TP=1，DP=EP=16，EPLB开启 |
| 输入 | 每rank同样4条不同输入：8192 token结构化合成背景，加中英问题；不是64条独立prompt |
| 生成 | 每请求128 token，temperature=0、seed=0、ignore_eos；DSpark出5验6 |
| 调度 | 每rank最多4请求，max_num_batched_tokens=2048，max_model_len=9216；实际batch会下降，不是全程固定GBS64 |
| 模式 | eager，async_scheduling=False，prefix caching关闭 |
| 采集 | 每rank前16个纯decode窗口的完整目标logits、输入token、position、已接受历史、draft长度；全部输出token及top20 logprob |

依赖版本限制：2026-09-28 11:44（北京时间）的事后核查发现PyPTO有未提交的 `_kernel_abi.py`、`torch/shutdown.py` 和测试修改，子模块引用也有修改；Simpler子模块自身工作树干净。补丁和状态已保存。运行日志记录import路径但没有逐文件hash，所以本次不能被描述为“纯净PyPTO HEAD”，也不能仅凭事后补丁证明运行时每个依赖文件完全相同。

## 四组定义与运行覆盖

| 组名 | 源码快照 | CSA | 请求的KV页大小 | 完成请求 | S6步骤 / CSA覆盖步骤 |
| --- | --- | --- | --- | --- | --- |
| native128 | Leaf c4383cfea | 关闭 | 128 | 64×128 token | 784 / 不适用 |
| leaf | Leaf c4383cfea | 开启 | 128 | 64×128 token | 656 / 656 |
| native32 | BSH 0d1b8ea7，原生架构 | 关闭 | 32 | 64×128 token | 784 / 不适用 |
| bsh | BSH 0d1b8ea7，自定义CSA架构 | 开启 | 32 | 64×128 token | 624 / 624 |

S6覆盖按每次forward各目标层的调用计数增量检查，要求至少21层各调用一次；无覆盖失败记录。仅表示该路径实际运行，不代表数值通过。

native128与native32同时存在源码快照、页设置差异，不能把其输出差异全部归因为页大小。日志中的draft模型初始化另有page=32提示，不应将该提示直接当成target缓存配置证据。源码快照及完整日志均保留供继续核查。

## 公平比较：四组共同历史子集

对每个logits行，核对prompt token、绝对位置、已接受token前缀、本轮query前缀。只比较四组都出现的相同条件，不比较已分叉的文字历史。以下采用rank0，其他rank的逐对比较另行保留；16个rank都输入相同4条prompt，每组生成token也在各rank间相同，但不据此推断rank间logits逐bit相同。

| 对比 | 共同位置数 | 平均逐行logits相对L2 | 平均概率TV | 最大概率TV | 最可能token相同 | 固定allclose通过行 |
| --- | --- | --- | --- | --- | --- | --- |
| Leaf vs native128 | 94 | 3.6779% | 2.4146% | 33.5136% | 91/94 | 0/94 |
| BSH vs native32 | 94 | 3.8080% | 2.3601% | 30.4354% | 93/94 | 0/94 |
| native32 vs native128 | 94 | 1.3949% | 1.0013% | 13.4955% | 91/94 | 66/94 |
| Leaf vs BSH | 94 | 2.8220% | 1.8960% | 16.3095% | 89/94 | 13/94 |

固定allclose为 `rtol=atol=1e-2`，未放宽；logits相对L2与单层attention output相对L2不是同一个指标。最大TV说明个别位置差异明显，不能只看平均值。平均差距接近也不是统计等价性证明。

### 子集覆盖

| 问题尾部 | 四方共同历史行数 | 绝对position范围 | 各组保存的总行数 |
| --- | --- | --- | --- |
| 天空为什么是蓝色 | 54 | 8223–8258 | 每组总计384行，含4条prompt |
| 合并两个有序列表 | 12 | 8210–8221 | 同上 |
| 火车平均速度 | 24 | 8206–8218 | 同上 |
| 二分与线性查找 | 4 | 8192–8195 | 同上 |

相同position可能对应不同draft前缀，因此行数可大于position个数。交集只有94/384行（24.48%）；提前分叉的困难窗口更容易被排除，存在选择偏差。共同token历史也不保证缓存数值相同：累积舍入、压缩状态和draft恢复的影响都包含在本次端到端差异内，不能直接归因某一个kernel模块。

## 辅助结果

逐对比较使用各自共同历史，覆盖集合不同，不能直接排精度优劣：

| 对比 | 16rank匹配行 / 各组保存行 | 汇总logits相对L2 | 最可能token一致率 |
| --- | --- | --- | --- |
| native128 → Leaf | 2688 / 6144 | 5.1569% | 2560/2688 |
| native32 → BSH | 3792 / 6144 | 5.0989% | 3728/3792 |
| native128 → native32 | 3520 / 6144 | 1.7151% | 3472/3520 |
| BSH → Leaf | 2416 / 6144 | 4.8891% | 2288/2416 |

四组每次draft平均均为5 token。计数器定义的平均接受长度（`1 + accepted_tokens / num_drafts`）分别为native128=2.7838、Leaf=3.3397、native32=3.0351、BSH=3.6084。它不是每次调度实际输出步长，也不证明目标3.8已达到；各组自由生成轨迹不同，不能用此结果直接判断性能或精度更好。

此前“0/64序列完全一致”只表示每条128-token输出至少一处不同，并非64条回答全部错误。现在已有数值证据，但还不足以保证全模型与BSH等价。

## 证据、复算和未覆盖项

原始证据包标识：`full-alignment-20260928-c4383cfea`。包含四组结果、每rank JSON、1024个logits采样文件、采集脚本、Leaf/BSH Python源码快照及逐文件SHA256清单。模型权重未重复打包；模型配置hash、文件名/大小/mtime及依赖事后补丁在provenance附件中。共享编译扩展仍是外部符号链接，因此此包是数据复算包，不是可脱离原环境运行的完整容器镜像。

本机与227各保存一份归档；确切位置、归档hash及下载校验记录在证据包旁的 `README.md`、`archive-verification.json`。分析产物为 `comparison.json` 和 `common-history-analysis.json`，脚本为 `compare_full_alignment.py` 和 `analyze_common_histories.py`。

CPU复算命令（先解包，将ROOT设为results目录）：

```bash
OMP_NUM_THREADS=2 python compare_full_alignment.py "$ROOT" --out comparison.json
OMP_NUM_THREADS=2 python analyze_common_histories.py "$ROOT" --out common-history-analysis.json
```

分析公式已用完全相同输入、整体平移logits和改变top1三项CPU检查验证，并经独立方法审阅。本次不报告速度：采集含CPU同步和落盘，会干扰时延。

仍未完成：完整模型graph、128K、不同单卡batch、分歧后的固定共同序列重放、逐层误差定位、全部128生成位置的logits覆盖和缓存张量直接比较。已保存的数据足够复算当前共同历史结果，但没有落盘所有层的KV/中间张量，不能承诺所有后续定位都无需重跑。
