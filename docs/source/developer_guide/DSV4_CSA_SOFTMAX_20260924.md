# G11：Softmax 累计最大值对齐

基于 G10 独立候选，文档基线 eedec695e。本轮没有合入正式服务算法。
保持 G10 环境、真实 C4 第 2 层权重、TP1/B4/S6/block128、seed62 合成 hidden/history。
原生确定性 level1 与 HCCL 开启，权重不变、guard、输入和源码 hash 核验通过。

## 对齐内容

原生在概率舍入 BF16 前使用包含 sink 和历史块的累计最大值。CSA 原先各块独立求最大值，PV 阶段再合并。
候选增加独立的 Softmax producer 最大值状态；保留 128 分块和原有 PV 流水线，避免读取滞后的 PV 最大值。
这仍未对齐原生 512 分块以及所有归约顺序。

## 单因素混合诊断（8K）

固定 native QR 注入 Indexer；此表不代表完整 CSA。

|指标|G10 control|running max|再对齐 sink 分母|
|---|---:|---:|---:|
|同输入 heads relative L2|0.029867%|0.005577%|0.005577%|
|同输入 heads 不同元素|316183|25292|25291|
|同输入 heads 超过 4 ULP|32850|2249|2249|
|混合最终输出 relative L2|0.380709%|0.242896%|0.242896%|

同输入指 Q/cache/TopK 完全相同。running max 以外的上游模块指标未变化。
最大 ULP 受零附近及跨符号值影响，应同时检查绝对误差和参考值幅度，不能仅凭最大 ULP 判定误差规模。

sink 实验将在线 sum 初始化为 1，末尾不再加 sink；修正死加载后，最终输出与 running max 一致。
故本轮完整复测只采用已验证的 running max，不包含 sink 改动。

## 完整单层 Graph：无插桩、无 native QR 注入

baseline=G10 aligned，aligned=仅再加 running max。12 轮×100 次 replay，顺序轮换，reset 与校验不计时。
单位 ms/attention forward；百分比为相对原生输出的 relative L2。

|长度|原生延迟|baseline 延迟|aligned 延迟|baseline 误差|aligned 误差|
|---|---:|---:|---:|---:|---:|
|8K|0.574836|0.596304|0.592847|0.444658%|0.334139%|
|128K|0.770945|0.879161|0.877537|0.378008%|0.224183%|

延迟基本持平，不据小幅差异宣称稳定收益。aligned 相对原生仍约慢 3.1%/13.8%。
输入与原生输出 hash 等于 G10，本轮 baseline 输出 hash 等于 G10 aligned。
精度尚未达到少数 ULP，继续分别定位 Q 上游误差与同输入 attention 残差。
不代表整模型、真实请求推进、动态 padding 或 DP16 测试。

## 失败与修正记录

- producer 与 PV 最大值共用初值 tile 导致编译拒绝，分别加载后通过。
- sink 常量初值触发布局推导错误，沿用原布局生成初值后通过。
- sink 首次运行退化，异常集中在首轮 worker 的固定 head/维度；生成代码中死加载与输出加载重叠。
  删除无用 m_mi 加载后退化消失，支持该定位，但尚未形成编译器最小复现。
- 完整 Graph 两次在启动/证据检查失败：遗漏 run.py，随后遗漏生成脚本。
  补齐全部依赖并无设备检查导入及证据文件后完成本轮结果。

完整证据见 [JSON](DSV4_CSA_SOFTMAX_20260924.json)。
