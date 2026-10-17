# d253 Attention 逐 tick：三轮只读分析

本轮将优先范围缩到**压缩候选采集与 KV 发布推进**，尤其每个 query 的第一个压缩块。SWA 占 gather 记录的比例很小，AIV 两 lane 没有稳定的单侧失衡。数据仍不能把这段时间唯一拆成 Scalar 页表、随机 KV 读取、UB 依赖或 cache 成本，因此不再提出第二候选；下一项只推进主会话已选、另有源级证明的 consumer 两层冗余 guard 删除。

唯一作业 `task_20261018_032708_262386210870` 的 [result.json](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_tick_h131072_b16_v1/result.json) 为 complete/PASS。固定 128K/B16/S6，d253 源，完整输出、TopK/score、状态/cache、保护与信号检查通过；三轮 A/B/A 各有 1632 条 Attention tick，共 4896 条。它是插桩诊断，不是完整 L1 性能测量，不涉及 Native forward/token 验收。没有追加编译或设备测试，冻结源码/运行入口未改。

统计和原始文件引用完整保存在 [ANALYSIS.json](ANALYSIS.json)。每轮先按物理核对有效区间求和，再对逐核总和取中位数；逐块表则直接对该块的 192 条 AIV 半块记录取中位数。两种口径不能混用。

工作对应关系由 [attention_stream_stages.py](source_snapshot/attention_stream_stages.py:141) 的实际循环确定：AIV 的 gather 是 tick，softmax 是 tick−1，merge 是 tick−3；AIC 的 QK 是 tick，PV 是 tick−2。有效 work 为 0..19，query=`group+24*(work//5)`，block=`work%5`。block 0 是 SWA，1..4 为压缩块。AIV 物理核为 `24+2*group+lane`。统计按各项自己的 work 分类，没有把同 tick 中不同 query/block 的 softmax/PV 误归到 gather 块。

| 区间逐核累计中位数（µs） | 第 0 轮 A | 第 1 轮 B | 第 2 轮 A |
| --- | ---: | ---: | ---: |
| AIV gather/ring 发射 | 68.30 | 66.26 | 67.39 |
| 其中 SWA（每核 4 次） | 3.41 | 3.36 | 3.40 |
| 其中压缩（每核 16 次） | 64.76 | 62.89 | 63.80 |
| AIV softmax→prob 发布发射 | 25.49 | 23.05 | 23.77 |
| AIV PV gate/merge，含 PV wait | 24.42 | 24.39 | 24.58 |
| AIV 最终输出/reset 发射 | 13.45 | 15.08 | 13.21 |
| AIC KV FFTS wait | 114.12 | 108.97 | 116.65 |
| AIC PROB FFTS wait | 0.80 | 0.87 | 0.78 |

逐核压缩 gather 占全部 gather 的比例中位数约 94.94%–94.97%。表内各项存在嵌套；SWA/压缩各自的中位数也不必精确加成总体中位数，不能将上表相加当作端到端耗时。

| AIV 单半块 gather 中位数（µs） | 第 0 轮 A | 第 1 轮 B | 第 2 轮 A |
| --- | ---: | ---: | ---: |
| SWA block 0 | 0.68 | 0.67 | 0.66 |
| 压缩 block 1 | 6.20 | 6.12 | 6.19 |
| 压缩 block 2 | 3.90 | 3.84 | 3.82 |
| 压缩 block 3 | 2.68 | 2.66 | 2.64 |
| 压缩 block 4 | 2.98 | 2.92 | 2.92 |

首压缩块较慢不是只出现于整个 helper 的首个 query。按每核处理的四个 query 次序，block 1 的中位数分别为：A0 的 6.35/6.23/6.20/6.12，B 的 6.44/6.08/6.07/6.03，A2 的 7.28/6.39/6.07/6.01 µs。三轮 block 1 的最慢单条为 11.92/9.00/18.14 µs，均保留在原记录。它提示持续的块位置差异，但没有每候选有效行数/物理地址/cache 事件，不能据此认定特定 cache 层或 DMA 吞吐是原因。

[gather 源码](source_snapshot/attention_stream_stages.py:271) 的范围同时包含 64×512 BF16 清零、Q/L/position 几何检查、Scalar 索引与页表、KV 读取及 ring 写发射。SWA 已按页 run 搬连续行；压缩路径是 64 行循环，每有效行读取索引、计算页列、读取页号、检查范围并提交单行加载。当前生成代码没有压缩循环内的显式逐行 wait。因此块差异支持缩小审查范围，不能直接推出“DMA 串行”或将整个 gather 时间视为可省工作。

两个 lane 的 gather 累计中位数分别为：

| 物理 lane | 第 0 轮 A | 第 1 轮 B | 第 2 轮 A |
| --- | ---: | ---: | ---: |
| lane 0 | 68.01 | 66.58 | 67.45 |
| lane 1 | 68.57 | 65.88 | 67.05 |

慢 lane 在轮次之间反转；没有证据以固定 lane 重分工换取大幅收益。最慢核也必须区分定义：

| 观测 | 第 0 轮 A | 第 1 轮 B | 第 2 轮 A |
| --- | --- | --- | --- |
| 最晚 ATTN_VECTOR 发布的 AIV core（group/lane） | 57（16/1） | 37（6/1） | 34（5/0） |
| 上述核 tick 区间累计（µs） | 164.74 | 153.62 | 158.20 |
| 上述核 gather 累计（µs） | 73.92 | 68.42 | 64.16 |
| gather 累计最大的 AIV core / µs | 37 / 77.94 | 24 / 69.66 | 25 / 80.58 |
| tick 累计最大的 AIV core / µs | 57 / 164.74 | 49 / 160.78 | 24 / 162.54 |
| KV wait 累计最大的 AIC core / µs | 6 / 125.16 | 22 / 119.84 | 13 / 126.36 |

最晚发布与最大 gather 并不恒为同核，入口错峰、同组另一 lane、SCORE 等待及其它区间共同影响完成时间。JSON 保留各定义前三名的开始/结束、发布与全部区间总和，避免只挑一条尖峰归因。

按现有共同时钟域，把同组两 lane 的 gather 区间各与对应 AIC KV wait 相交，先取两 lane 并集再累计，24 组的累计中位数为 **49.57/47.06/51.48 µs**。这说明 AIC 等 KV 时有相当一部分时间与本组 AIV gather 区间重合；这是时间关系，不是可直接节省的量。每块较晚的 ring store 发射到较晚的 KV notify 发射，中位 1.60/1.58/1.47 µs；当前 T=96 按既有协议在前一块 softmax 后发 KV，不能把这段当成纯通信浪费。AIC wait 结束比两 lane 中较晚的 KV notify 发射晚约 1.12/1.12/1.10 µs（中位），仍包含生产 pipe 依赖/发送完成及后续推进，不能直接据此修改事件协议。

局部 `ring_local_join` 累计中位仅 0.06–0.08 µs，`tick_reuse_waits` 约 0.04 µs；**这些 `wait_flag` 前后 Scalar marker 很可能只覆盖发射区间，不能推断 MTE/UB 的真正等待接近零**。它们与 A3 会阻塞后续流水的 `wait_flag_dev` 不同。本轮没有检查最终机器码或加入完成 fence，不能把这些局部数值用作删除同步的依据。

全有效块跳清零的排重结果为：

- MASTER [相关记录](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md:6747) 与 [qk_valid_gather.py](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/qk_valid_gather.py:1) 已实现“64 行全有效时 create，全部行覆盖；非法行回退清零”。末版全有效分支没有 TEXPANDS/TFILLPAD，整链及全有效→混合负→全负→全有效同图往返通过；旧任务 `task_20261017_020426_59480118626` 中位 839.84 µs，未采用。其实际静态 PIPE_ALL 从 9 增到 11，不能把清零次数减少预记为收益。
- 那版在 AIV 做 row_min 并有 8 行展开；“plan 记录 128 行全有效位、保留当前 64 行循环”并非完全相同的已测表达。但本轮没有分离清零成本；相同 64 KiB 初始化的 SWA gather 只有约 0.67 µs，也不支持将首压缩块约 6.2 µs 主要归因于初始化。按主会话决定，不推进这条新表达。
- direct-L1、MGATHER、独立行 Tile、mi/li 环、地址/物理 row plan、query metadata 以及早 KV 阈值的既有边界保持关闭，证据与失败原因见 [此前有限只读复核](../attention_aiv_d253_audit/README.md)。本轮不将它们重新命名为候选。

当前两层 consumer guard 的删除触点位于较大的压缩 gather 区间，具有源码可省工作依据；它仍保留 page/index 读取与原 KV 行搬运，不能承诺消掉本表中某个固定比例。正确性证明、实际生成差异与完整 L1 对照由该独立候选自己的证据负责，本分析不替它预记通过或收益。

最后的计时边界保持不变：旧 phase_clock 每 notify 的 DCCI/DSB 仍存在；新 tick hook 没有额外硬件 fence，却有 Scalar 记录与编译重排约束开销。tick_end 不含自身记录写入、下一 tick 初始化、最终 helper drain/release；publish_end 是最后 TEXPANDS 发射之后，非 Vector/DMA 完成。所有区间均保留原依赖且部分嵌套；本轮不作不同作业总时差比较、不计入性能目标。
