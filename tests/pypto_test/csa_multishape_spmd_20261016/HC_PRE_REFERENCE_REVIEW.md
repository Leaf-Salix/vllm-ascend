# 用户指定 HC_pre 优化与当前 CSA 单 SPMD 的对应关系

2026-10-17，用户指定参考 canonical 仓 `vllm_ascend/ops/pypto/hc_pre.py`。
本轮已读取实际文件，并保存 `hc_pre_reference_20261017.py` 文本快照；不修改原文件、不做hash。
独立worktree中的同名文件不能默认等同于用户指定文件。

## 有直接参考价值的部分

1. 第179～299行：K分段内流式供数，AIV加宽后的同一份FP32输入用于投影供数与平方和统计，
   用 aic_gather 交给AIC，减少完整FP32输入写回/读取GM。当前hc_stream_pre候选仍在GM的view_2中
   store/load，这与独立HC_pre的组内供数不是同一实现。
2. 第230～246行：显式交错L0双缓冲，先准备块0/1，算0后准备2，再算1，保留同一FP32 Acc及K顺序。
   这是把下一次搬运提前并设计实际活跃区间，不能仅以pl.pipeline(stage=2)名义推断重叠。
   可作为CSA投影/Indexer局部流水参考；不同L0C容量和现有事件要逐项核对。
3. 第140～153、370～381行：按阶段重分工，全体48 AIV分别承担Sinkhorn与mix，token少时再切D；
   投影的token×K分片与尾段token/D分片彼此独立。这是当前融合大SPMD重排阶段应借鉴的工作量设计。
4. 第503～585行小档Sinkhorn标量迭代、第667～710行设备标量门值与连续D区间/整行mix，减少小tile
   gather/broadcast/转置及过多碎片。不能把这些算法直接套到所有attention数学；要核对规模和计算顺序。
5. 第775～800行按T区间选择M/K/供数宽度/尾段行数，依据核波次和存储容量，保留真实尾块。
   CSA还要考虑B、历史长度和query分组，不能照搬HC_pre的阈值或仅按T统一分档。

## 之前的 HC 候选不等价于完整吸收

`hc_front_fusion.py`只是T96的reduce→split→mix三阶段结合，原数学及中间GM保留，Sinkhorn独立。
`hc_stream_pre.py`保留原4份K4096投影部分和、原32项RMS顺序，走GM流式交接。
因此这些候选未优于现选，不证明独立HC_pre整套策略在完整CSA中无价值，也不能声称已全部移植。

## 实际移植的边界

- HC_pre没有input_layernorm。当前CSA核体也先把混合结果round到BF16，再转FP32统计RMS并执行norm
  （冻结core_bodies_ir.py第357～381行）；串接BF16 mixed本身不会新增这处舍入。
  真正精度风险包括投影部分和、RMS分段归约、Sinkhorn加法/除法顺序的改变。
- 独立实现第364行是全参与者硬件syncall。把这一屏障嵌入完整CSA前必须确保所有参与核都到达且
  不破坏当前点对点进度图；不能把局部阶段等待随意改成整算子全局会合。
- standalone变快不能代替完整CSA L1收益。当前所有七档仍需各自达到同轮动态PTO的0.70。

## 本轮 QK 负结果与参考方向的关系

`qk_row_load_pipeline_v1/lowering_audit.json`证明stage=2没有把两行DMA提前：
TLOAD→MTE2/V wait→TMOV→第二行TLOAD→wait→TMOV，两行还共用一个UB地址。
更关键的是现选索引缓存基线原本可以直接对目标行连续提交TLOAD，没有候选引入的逐行assemble等待。
所以关闭当前候选，不上卡、不扫stage；不能把错误的基线瓶颈假设带到下一轮。
HC_pre给出的方向是先核对真实供数与消费关系，再改变实际数据复用和分工。
