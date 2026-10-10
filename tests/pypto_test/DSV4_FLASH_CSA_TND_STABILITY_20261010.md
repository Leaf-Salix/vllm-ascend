# TND 精度稳定性扩展与首次失败定位（2026-10-10）

## 当前结论

正式精度版本为 `99854b2b591975356d2fd090089348227ddc5600`，分支为
`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
本轮未修改正式精度包或性能包。此前四档逐 bit 通过只代表那些输入，不能推广为所有上下文长度均通过。

按用户要求：发现第一项 Native 精度失配即暂停矩阵，定位该输入，不继续完成全部测试。
47 项计划仅执行前三项，前两项通过，第三项失败；其余 44 项保持未执行。

| 正式版本用例 | 请求长度 / T | heads 不同元素 / 最大 ULP | 最终输出不同元素 / 最大 ULP | 全 cache/state、TopK | 状态 |
| --- | --- | --- | --- | --- | --- |
| start8191、B4、seed62、layer4 | 6,6,6,6 / 24 | 0 / 0 | 0 / 0 | 全逐 bit | A/B 输入及重放稳定性通过 |
| start8189、B4、seed62、layer4 | 6,6,6,6 / 24 | 0 / 0 | 0 / 0 | 全逐 bit | A/B 输入及重放稳定性通过 |
| history259、B4、seed62、layer4 | 6,3,1,1 / 11 | 3 / 1 | 210 / 298 | 全逐 bit | 首次失配，立即暂停 |

第三项 heads relative L2 为 `7.3833234776e-6`，最终输出为 `8.3537867143e-5`。
这些误差指标不能替代逐 bit 判据；第三项并未执行后续完整重放和计时，不能记录为稳定性通过。

## 测试协议与参考分支

参考 Opus55 精度分支 `dev/pypto-dsv4-csa-tnd-opus55-20261008`，提交
`05e5d7c23f3ea999c08991a7e331769d276f366f` 的测试覆盖设计。
其 18 档包括 start8184…8196、等长与非等长请求，压缩长度均为 512；
本轮补入短上下文、压缩规约分档边界、多 batch、额外 seed 和 layer2。
Opus55 的全 output 通过不等同于全部 heads/state 通过，也不是不同环境间受控性能对比。

本轮固定真实 DeepSeek-V4-Flash-0731-w8a8 C4 第4层权重，合成 hidden/history。
Native 开 deterministic level1、HCCL 确定性、static + SuperKernel，atomic0。
各臂独立 cache、graph owner、注册名；禁止 PTO 回退。
精度包显式复用 Native HC/norm/O-proj，PTO 执行 Q/KV、compressors、Indexer、attention。
TND 总 token 数为请求长度之和，没有 padding 到 B×6。

环境：CANN/ATB 9.2.0-beta.2，Torch 2.10.0、Torch-NPU 2.10.0.post2，
vLLM 0.25.1、vLLM-Ascend 0.25.1rc1 lineage，PyPTO `3e87a843`、Simpler `a54c0509`、PTOAS 0.66。
框架和已编译二进制复用，没有重装。只通过单卡 task-submit 上卡。

原始报告的旧 `candidate_manifest` 标识固定 core/base，不能用它代表本轮精度源码。
实际候选身份由 `ab_manifest` 的正式提交 `99854b2b5` 和 23 文件 SHA 确定并校验；
结构化记录保留 `ab_manifest`，未将旧 base 提交混作新的精度提交。

硬门禁包含完整八类输出/state 的 raw-bit 比较、实际 Native O-proj 输入 heads，
compiled/raw、100 连续 replay、reset、hidden A/B/A、B 输入跨臂比较、非有限值和保护区。
观察 Native heads 的独立臂必须与未修改 Native 全状态一致，防止 hook 改变参考结果。
首个 A/B 精度不一致会立即保存报告并拒绝继续；batch 非零立即停止。

## 单输入阶段定位

仅重复 history259 / B4 / T11 / seed62 / layer4，没有恢复完整矩阵。
在另一物理卡复现同样 3 个 heads 和 210 个最终输出差异。

| 阶段 | 不同元素 | 最大 ULP | 证据 |
| --- | --- | --- | --- |
| Native 实际 attention Q 输入 | 0 / 360448 | 0 | 持久观察缓冲与真实调用参数绑定 |
| 六类完整 cache/state、完整 TopK | 0 | 0 | 观察副本与正式版本全状态一致 |
| inverse RoPE 前 heads | 3 / 360448 | 1 | 首次差异已出现在 attention 本体 |
| inverse RoPE 后 heads | 3 / 360448 | 1 | 同三处 NOPE 区元素 |
| 最终输出 | 210 / 180224 | 298 | 小 heads 差异经后续量化放大 |

三个 preinverse 坐标为 `[4,3,151]`、`[5,15,431]`、`[8,53,419]`，均小于 NOPE 维度 448，
不经过 inverse RoPE 旋转。实际 Q 和已写 cache/TopK 都一致，因此本例先定位 attention 本体。
观察副本的完整状态 fidelity、compiled/raw、A/B/A、100 replay/reset 均通过。
此观察本身不能证明 softmax 求和是唯一根因，仍需单因素实验。

## 短压缩序列求和顺序的单因素实验

history259 的有效压缩长度为 K65/66。Native SDK
`softmax_common_nd_reduce.h::NewReduceSumLastNDImpl` 对 `64≤K<128` 使用
首64列复制、tail折叠进对应列、64→8→1 的规约。
正式 v23 在 full-window 条件下对压缩块始终使用 K512 的分组规约，浮点加法结合顺序不同。

独立候选只在 `position≥127 && 64≤K<128` 的压缩块复用已有 two64 求和树。
max、exp、alpha、概率 CAST_ROUND、PV、除法和 BF16 舍入均不改；
K<64、K≥128、raw128 和原短窗口分支不改。这不是完整 K0…512 修复。

同进程保留未修改基线，硬断言复现 heads3 / final210、其余七状态逐 bit，
候选与基线 heads 缓冲地址必须不同，再判断候选是否消除误差。
实际 signature/ABI、lower、orchestration 编译及独立 CPP 审查通过：
两份64列切片保留512物理stride，规约前连续 reshape，临时区不覆盖 exp。

单因素任务 completed、exit0。同任务旧基线稳定复现 heads3 / final210，
候选 A 输入及扰动后的 B 输入均与 Native 全八状态、实际 heads 逐 bit 一致。
compiled/raw、hidden A/B/A、100 replay/reset、保护区、非有限值和真实 Native SK 门禁均通过。

| 同进程同输入 | heads 不同元素 / 最大 ULP | 最终输出不同元素 / 最大 ULP | 六类 cache/state、完整 TopK |
| --- | --- | --- | --- |
| 正式 v23 原始计算 | 3 / 1 | 210 / 298 | 全逐 bit |
| 仅修改 K64…127 求和树 | 0 / 0 | 0 / 0 | 全逐 bit |

这支持本例剩余差异由压缩 K65/66 的 softmax 分母规约顺序引入。
PTO 精度版需要按 Native 的实际 K 分档选择规约树。
同轮 round mean：Native 357.02 μs、候选 634.39 μs、旧基线 640.07 μs，
单个输入及单轮计时不足以证明稳定性能收益。

该修改仅保留在独立诊断副本，未合入正式源码。矩阵保持暂停，余下44项未执行。
K64、K127及128…511等分档边界仍需单独验证，不能从K65/66零误差推广为全部短上下文修复。

## 结果记录与范围

结构化记录：[history_20261010.json](evidence_tnd_precision/history_20261010.json)。
原始报告 SHA、各臂源码 manifest、driver SHA、逐阶段指标和门禁保留在该文件，便于后续 commit 对比。
上一轮历史：[精度版历史](DSV4_FLASH_CSA_TND_PRECISION_20261009.md)。

这是单层合成输入测试，未证明整模型、DP/EP16、EPLB、接收步长或吞吐。
本轮不是性能优化测试；稳定性通过两档的 Native/精度版 round mean 分别为
443.31/735.76 μs 和 416.97/758.06 μs，不外推其它形状。
