# Native精度TND：QK前瞻流水优化（2026-10-26）

## 当前结论

本轮按模块差距优先优化attention，保留Native数值顺序。query内QK前瞻在8K两次独立捕获、128K/B16和混长T126均有稳定收益，Native A/B八边界及ABA逐bit通过；真实整模型8K/B40的warm192＋测量192 token和接受事件也与既有Native完全相同。128K/B24整模型随后通过，同样与Native逐token及接受事件一致。

- 当前分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
- 保留基准：`0a5baa322733afdb1baad6a600f1eabfa22a4419`。默认精度alias及原combined实现保持原选择；本轮仅改显式`tnd_precision_tuned`的attention编排，T<96原路径保留。
- nalinaly远端已同步到`c346573ab3313fcc71cd58756acd129f34755439`，源码另行封存，没有merge其不同数值边界。
- 仍是纯DSL、一个完整HC-pre→HC-post顶层入口、64Tensor ABI、真实TND，不要求每请求补到S6；没有新增Native模型算子、SDK/runtime或生成C++修改。

## 1200us与1800us的口径

前序同环境、同输入的8K/B32受控比较中，Native event均值1257.224us，我方旧组合版1816.296us，nalinaly单SPMD固定SDK兼容版1212.730us。Native采用SK0，该表不是截图中的SK1；nalinaly独立兼容副本只删除固定SDK不支持的`backoff_ticks=150`关键字，不能等同其原工具链速度。详见[原始比较](DSV4_FLASH_CSA_TND_NALINALY_MODULE_GAP_AND_HC_OVERLAP_20261023.md)。

我方之后已优化到约1660us，1800us不是本轮控制。nalinaly最新公开`csa_performance_20261025`在其新SDK上8K/B32为946.46us，SDK为PyPTO d44925c1/PTOAS914863ed；我方本轮固定2dfa7f12/PTOAS0.66。946.46、历史1212.730及本轮结果不是一组计时，不能跨SDK/批次计算受控加速比。

## 模块优先级与实测收益位置

前序同输入nalinaly多task伴随诊断提供可展开的阶段参考；不是其单SPMD内部计时。本轮在我方已验Vector控制与新lookahead间做独立L1/依赖映射。

| 非重叠L1区间，中位数us | 我方本轮控制 | 我方lookahead | nalinaly多task历史参考 |
| --- | ---: | ---: | ---: |
| 入口→TopK就绪 | 687.20 | 690.76 | 565.50 |
| TopK就绪→heads就绪 | 537.60 | 503.84 | 280.29 |
| heads就绪→完整输出 | 347.16 | 347.76 | 218.97 |

每个窗口先求分段，再分别取中位数；中位数本身不要求相加。nalinaly列是前序cafcadec96多task参考，没有冒充本轮c346573ab单SPMD重测。最大剩余参考差距仍在attention。

本轮QK/PV AIV包络407.56→378.42us，AIC405.70→376.90us；merge_norm38.50→38.88us，WO-A168.22→163.56us。不同函数可能重叠，不能相加。QK/PV包含计算、搬运、Scalar控制和同步等待；这些数据确认收益出现在QK/PV，没有把减少量全部解释成纯等待或纯算力变化。

18个无dep计时窗口全部task/core记录与单独4个dep窗口对应，重复映射稳定。CPU/JIT Core SHA一致，编排通过严格的局部identifier一一改名等价门禁；字面量、预处理、常量、调用顺序、配置及compiled metadata保持相同。计时与dep-gen没有同时启用。

## 实现：借鉴流水，保留Native数学

原版Cube逐块执行QK0→PV0→QK1→PV1；新版本在SWA和压缩块均有效时执行QK0→QK1→PV0→PV1。AIV消费SCORE0事件后，先把压缩KV预取到另一GM槽，再计算原SWA softmax；此时Cube可以执行QK1。softmax、PV归并仍严格按SWA→压缩块顺序。

- score由每core一个GM槽改两个，增加3MiB。SCORE0留在GM，预取结束后才load到UB。
- probability/PV仍各一个槽，必须消费PV0并归并后才能发布PROB1。
- L1槽0是scratch，压缩parts2/3在槽1/2跨PV0保留；PV0重新载入SWA到槽0，PV1的parts0/1同样重新载入。
- host防护要求固定两块（SWA与compressed），不符则拒绝；CPU已检查1/2/3/5仅2通过。只在双有效块启用前瞻；其他mask保持原逐块协议。移除旧PROB0后的重复压缩KV预取，sb1保留下一query SWA预取。
- Native K512更新、actualK求和树、sink初态、概率`round`、inverse RoPE前BF16、全8192列共同量化及O-proj/HC-post均不改。
- 实际12类关键浮点赋值AST与控制相同；正式文件经import路径归一化并排除新增host防护后，每个DSL函数AST与设备受测候选相同，不以代码行数证明数值等价。

抽象协议模型检查3个连续query、全部64种mask组合、64000次随机双AIV/Cube交错，约268.8万步，验证事件单次发布/消费和KV/score/probability/PV生命周期。独立生成物检查确认UB最大187072B、L1驻留和每个PV重新初始化累加器。模型、编译通过均不代替NPU精度测试。

## 单层性能与精度

真实第4层权重、合成hidden/history，page32/NZ2、确定性level1/HCCL，完整CSA FULL graph raw replay。每臂12轮×100次event、另9个L1窗口，不剔除样本。降低为`1-新/控制`，越低越好。

| 配置 | 已验Vector控制us | lookahead us | 耗时降低 | 成对领先 | Native精度 |
| --- | ---: | ---: | ---: | ---: | --- |
| 8K/B32/T192 | 1656.683 | 1629.167 | 1.661% | 12/12 | A/B八边界＋ABA逐bit一致 |
| 8K/B32，反向捕获 | 1660.950 | 1629.200 | 1.912% | 12/12 | A/B八边界＋ABA逐bit一致 |
| 128K/B16/T96 | 1411.387 | 1381.201 | 2.139% | 12/12 | A/B八边界＋ABA逐bit一致 |
| 8K/B36，混合1～6行/T126 | 1398.751 | 1366.840 | 2.281% | 12/12 | A/B八边界＋ABA逐bit一致 |

四组L1中位数分别1572.96→1545.62、1573.22→1549.62、1319.30→1314.86、1316.34→1292.84us。收益门槛预先固定：event至少降低0.5%、至少10/12成对领先且L1更低；反向捕获作独立重复。先落盘性能，再作Native精度检查。

八边界为最终x_out、TopK、raw KV、compressed KV、compressor state、indexer两份cache及indexer state；每组有原输入A、扰动输入B和恢复A，不只比较一次输出。guards和只读metadata检查均通过。混长36条请求共126真实行，若每请求补6行则216行，本轮没有那90行padding。该结果不宣称所有不等长/短history组合均已穷尽。

## 真实整模型192 token

沿用固定Native bank，TP1、DP=EP16、DSpark出5验6、FULL、NZ2、确定性开启、EPLB关闭。21层CSA完整PTO，20层HCA及FFN保持Native；不跑Native2。先8K成功才提交128K。

| 配置 | 实际窗口 | warm192＋测量192与Native | Forward均值ms | Cycle均值ms |
| --- | --- | --- | ---: | ---: |
| 8K/B40 | 40活跃、T240/bucket240 | 16rank共245760位置token及事件完全相同，0抢占 | 99.609 | 115.191 |
| 128K/B24 | 24提交、21活跃、真实T126/bucket144 | 16rank共147456位置token及事件完全相同，0抢占 | 88.299 | 102.135 |

8K模型阶段11:08:50～11:20:40，约11分50秒；128K为12:29:18～12:37:52，约8分34秒。两项exit0并释放16卡。位置数是16rank×请求数×192×两轮，不是同等数量的独立请求；16rank使用相同固定bank。每个对齐步骤取16rank最大耗时，统计20个forward和19个cycle；cycle包含draft、采样、调度和CPU观察，forward含未替换HCA/FFN。没有本轮新Native/旧版whole计时臂，不据历史whole均值之差声称受控整模型加速。

审计要求各CSA层T6与大桶实际正式root捕获、FULL replay、计时期间Native分发计数不增加、无JIT重建；没有只凭环境变量或最终token判断替换成功。128K保留24000KV块、21活跃/真实T126/物理bucket144、原2GiB预留及[56,96,176,1]MiB ring/window2048，本轮已在上述ring配置完成设备验收，新增3MiB score容量没有触发容量错误。

各硬件任务调度780秒、内层750秒。8K排队后观察存在空档：最后一次pending观察至11:35收尾之间未满足持续每5分钟观察；该事实保留，不写成持续监控，硬件上限始终保留。

## 否决与未采用的候选

| 单因素 | 本轮8K/B32 event us，原→候选 | 决定 |
| --- | ---: | --- |
| 物理行plan＋MGATHER(row/zero) | 1662.990→1672.533，慢0.574%，0/12领先 | 不验Native精度，不扩128K/whole，不合入 |
| 同物理行plan＋Scalar/TLOAD | 1680.772→1682.509，慢0.103%，2/12领先 | L1同样回退，不合入 |

独立DFX表明MGATHER回退来自QK/PV而非新plan。固定ISA的MGATHER仍逐行DMA并带保守跨管线同步，不能假定它是并行64行gather。

另有统计量UB驻留尝试，仅CPU布局/表达失败，没有本轮NPU任务；复核历史发现同类独立状态方案已于10月23日实测慢1.48%，本轮不再重复占卡。后续不能因nalinaly采用某种存储方式就假定在我方K512/八头布局上必然更快。

直接在QK所属AIV尾部归一化/逆RoPE并发布heads是下一独立候选，尚未算入本表或正式实现。它必须先性能获益，再Native精度及两档整模型验收，不能把本轮lookahead结果套到该候选。

## 复现

固定SDK、模型、bank及Native黄金沿用[上轮证据](evidence_tnd_precision/attention_qa95_20261023/README.md)。本轮原始结果、源码封存、CPU收据、事件模型、L1/dep映射与整模型数据见[本轮证据目录](evidence_tnd_precision/qk_lookahead_20261026/README.md)。复现必须新目录校验manifest、重新编译和资格检查，不能给改过路径/源文件的目录套用旧编译收据。

显式选择`tnd_precision_tuned`；短T<96仍走旧路径。回退可恢复`0a5baa322`，旧精度/default/combined alias本轮未改。当前没有声称追平nalinaly单SPMD，也不将nalinaly已知数值差异当作全部速度来源。
