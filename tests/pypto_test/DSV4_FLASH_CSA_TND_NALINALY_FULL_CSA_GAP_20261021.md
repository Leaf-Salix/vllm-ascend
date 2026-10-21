# 同环境完整 CSA 性能差距定位（2026-10-21）

## 当前结论

已完成同环境、同输入、相同完整 HC-pre→HC-post 图重放边界的四方实测，并补齐双方图内 compact metadata producer。
我方完整精度版与确定性 Native 的输出、TopK、六组 cache/state 在输入 A/B 下逐位一致。
当前最快参考是 nalinaly 多 task 版；修正计时边界后我方差距约 **515 μs**，不能用此前后缀单独快 17% 宣称完整 CSA 达标。

nalinaly 最新动态 SPMD 入口本身已经支持真实 TND；本次为了兼容其旧多 task 绑定器，只测共同的全 S6 子集。
没有使用它的生成 C++ 等待退避、phase clock 或 atomic 改写。默认 TWAIT 的完整图外层延迟不是用户截图的 SPMD L1 口径。

## 范围与复现

- 我方分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`；起始 HEAD `ac80d0e3b44dcacf0e6ea8d62329e4bf48f79a5c`。
- 正式精度源、验收环境和此前全部后缀候选保留，当前只在独立诊断目录增加文件。
- Nal pin：`91c4400e19ec10bd96e8de6314376f29229f6e0e`；SPMD 包 Git tree 与此前 `beaefa6f` 相同。
- DeepSeek-V4-Flash-0731-w8a8 正式第4层权重，合成 hidden/history，TP1，B16，T96，history8191，Native 页32，NZ2。
- 本次是完整单层 CSA，不是192-token整模型，也不代表128K结果。
- 四臂相同 hidden、positions、原始 cache、metadata、compact RoPE；各臂独立 cache/state，四组根NZ权重共享Native已加载存储。
- FULL raw NPUGraph，Native静态装包/DSA多流，SK0；没有证明SK1。Torch确定性开启、NPU deterministic level1、HCCL确定性开启。
- 每臂有编译/原始图回放一致、output/TopK毒化、A/B/A、自重复、readonly、外部与私有workspace保护区检查。
- Nal额外检查72参与者代际一致、12窗每窗+100，以及信号槽padding不变。Nal对Native差异单独记录；它不是我方精度gold。
- 正式计时12窗×100次，四臂轮换位置，保留全部样本；计时、L1 DFX、dep_gen分别运行。
- 单任务内部750秒、队列780秒，不留后台任务。

完整源码、SHA、版本、原始结果与实际生成入口见末尾证据与复现说明。

## 当前已验收完整精度候选：真实整模型8K与128K

目前已完成两档token验收、且这两档观测forward均值最低的完整候选是
`pkg:deepseek_v4_flash_csa_tnd_precision_spmd_overlap`，正式源提交
`48c58ce7c467dfce63d5cb5c41afd225488d6012`。保留旧精度基准及默认路由；此候选需显式选择。
完整HC-pre→HC-post、单顶层入口、纯DSL、真正TND；SPMD用于前缀，后缀仍为多task。
本表来自10月20日已归档整模型任务，本轮没有重跑整模型。

| 整模型配置 | Native历史forward均值 ms | 我方候选forward均值 ms | 比Native历史慢 | 我方cycle均值 ms | 对Native的输出token | 接受事件 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 8K/B40 | 89.4822 | **106.4669** | +18.981% | 122.0472 | **491,520个全一致，零差异** | 105,152个全一致 |
| 128K/B24 | 79.7299 | **90.7129** | +13.775% | 103.0798 | **294,912个全一致，零差异** | 61,568个全一致 |

两档均使用真实完整75分片权重，TP1、DP=EP16、DSpark出5验6、FULL_DECODE_ONLY、确定性level1/HCCL，EPLB关闭；没有固定接收步长3.8。
warmup192 token后测3轮×192 token，token/事件验收计入warmup及三轮。21层CSA实际替换，20层HCA及FFN保留Native；16rank均通过，零抢占。
8K每卡提交且实际活跃40请求；128K每卡提交24请求，测量窗口21活跃、3等待，126真实行使用144行全局graph档位；这不是每请求补S6。

性能为每步先取16rank最大值，再对测量轮次求均值；候选forward60点、cycle57点，保留所有样本。
Native为历史跨任务基准，128K Native有40个forward/38个cycle样本；不是本轮交错复测，微小跨任务变化不能作为稳定收益证明。
此处“最优”仅表示现有已验收候选中的观测均值最低，**仍慢于Native，也没有证明对齐nalinaly速度**。
旧SDK为PyPTO `3e87a843`、Simpler `a54c0509`、实际PTOAS0.66；不能把下文新SDK单层性能或候选验收拼成新的整模型结论。

token/事件一致只覆盖保存的本项目固定bank和实际入场协议，未证明所有中间tensor/logits逐位相同、任意请求或EPLB开启均一致。
128K严格核对Native相位映射和入场；CPU复用缓冲已知0/-1哨兵差异单独保留，不能称所有GPU输入逐位证明。
完整复现、环境、源码快照和原始归档见
[Sinkhorn重叠报告](DSV4_FLASH_CSA_TND_PRECISION_SINKHORN_OVERLAP_20261020.md)，独立逐项复核见
[8K原始复核](evidence_tnd_precision/spmd_sinkhorn_overlap_20261020/WHOLE_8K_INDEPENDENT_RECHECK.json)与
[128K原始复核](evidence_tnd_precision/spmd_sinkhorn_overlap_20261020/WHOLE_128K_INDEPENDENT_RECHECK.json)。

## 最新性能目标：按用户提供的八档完整L1表对齐

以下参考数值转录自用户本轮提供的截图，**不是本轮重新测出的nalinaly结果**。
截图注明BSH基准为单SPMD `a241fa61`，全部S6、完整链路L1；未给出TND候选commit、详细工具链、权重层、Native静态档位和重复统计方法。
不可自动沿用之前10月20日另一张截图的所有设置或统计定义。参考表的BSH与TND均属nalinaly；下列“我方”才指我们的完整精度候选。

| 档位 | Nal BSH μs | Nal TND μs | Nal TND耗时变化 | 我方已有L1 μs | 我方同口径复测 |
| --- | ---: | ---: | ---: | --- | --- |
| 128K/B1 | 363.28 | 357.18 | −1.68% | — | 待测 |
| 128K/B4 | 506.12 | 493.52 | −2.49% | — | 待测 |
| 128K/B8 | 617.16 | 620.40 | +0.53% | — | 待测 |
| 128K/B16 | 789.94 | 777.44 | −1.58% | — | 待测 |
| 128K/B24 | 1017.90 | 1013.46 | −0.44% | — | 待测 |
| 128K/B28 | 1153.56 | 1156.34 | +0.24% | — | 待测 |
| 8K/B16 | 635.20 | 634.16 | −0.16% | 1203.68，仅既有配置参照 | 待统一配置复测 |
| 8K/B32 | 915.52 | 918.32 | +0.31% | — | 待测 |

截图八档等权几何平均TND/BSH为0.99335，耗时降低约0.67%，尚未达到截图的≤0.90目标。
这项0.90是截图中nalinaly的TND/BSH目标，不能宣称我们已通过或将其等同token精度指标。
我方1203.68来自本轮新SDK、合成hidden/history8191、真实第4层权重、完整incore跨度的跨窗median。
它与截图634.16未建立同卡、同输入、同工具链及相同计时提取方式的受控关系，因此不填写正式相对差距或八档几何平均。
本轮自己复测的Nal多task在该相同fixture下L1为671.38、默认SPMD为852.28，另见下节；不能将它们改标成截图结果。

下一轮计时以这八档为目标矩阵：先固定双方实现SHA和原始脚本，明确完整HC-pre→HC-post的L1最早incore至最晚结束、包含内部等待，
计时与dep/PMU分开，记录权重/页布局/NZ2、S6真实行数、Native SK档位与静态资格、工具链和等待策略。
先完成8K/B16同输入控制点再扩八档；我方逐位首差即停。另保留短query/混合1～6行TND门禁，不能靠只跑S6掩盖布局回退。
各任务小于15分钟；完整CSA稳定收益成立后，再按8K/B40→128K/B24顺序复测真实整模型192-token一致性与性能。
本轮只整理与定方案，不把未执行的八档或截图数值登记为我方验收结果。

## 完整图耗时：统一producer口径

单位 μs，median，不删除离群值。Nal两臂在raw FULL捕获中各真实调用2次Native producer，输出与原fixture逐位一致；零拷贝绑定到本次DSL参数，没有增加copy。此处为固定metadata重放，不证明动态metadata更新。

| 实现 | min | mean | median | max | 对 Native 的输出 relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native SK0 | 949.79 | 950.84 | 950.15 | 957.70 | 0.00000000 |
| 我方冻结精度完整 CSA | 1295.17 | 1298.81 | 1298.00 | 1307.74 | 0.00000000 |
| Nal多task | 779.68 | 782.78 | 782.93 | 787.90 | 0.00331159 |
| Nal动态SPMD，默认等待 | 1031.52 | 1034.67 | 1034.61 | 1039.40 | 0.00331159 |

我方比Native慢36.61%，比Nal多task慢65.79%；我方对Native的A/B八类边界全部逐位一致。
Nal两版相对Native所记录的误差指标相同，未直接做两版之间的逐位验收；两版均不与Native一致。

旧v5保留作追溯：Native929.84、我方1310.02、Nal多task704.13、NalSPMD968.08。**旧v5 Nal图外预备compact metadata，不能作为最终共同producer比较表。** 新旧跨任务数值差不能直接叫producer成本。

Nal的TopK数值relative L2不代表选点集合质量，不能据此判断召回率。

## L1与实际执行的模块

本节来自v5范围的独立DFX/dep任务，提取PTO任务的incore覆盖跨度；不是补齐producer后的common-metadata整图计时。
L1只开level1，依赖图在另一任务中收集，所有记录映射到实际执行的函数；编译出的未执行分支不计入耗时。
同名函数的最早开始到最晚结束包含调度间隙；单核持续时间也包含Scalar、搬运和同步等待，不能叫纯计算时间。

| 边界/模块 | 我方 | Nal多task | 解释 |
| --- | ---: | ---: | --- |
| 完整incore跨度median | 1203.68 | 671.38 | 包含所有核的完整跨度，与图外层event分开报告 |
| 实际QK/PV最慢AIV | 237.28 | 128.82 | 最明显的单个核任务差距，尚需PMU和单因素拆解 |
| 实际QK/PV最慢AIC | 235.36 | 125.10 | 含等待，不可与AIV时间相加 |
| checked/pair索引检查跨度 | 46.82 | 未分成同名独立阶段 | 我方保留Native pair次序，不能直接删除 |
| QK plan跨度 | 17.38 | 15.24 | 不代表producer/consumer完整开销 |
| O-A对应函数最慢核 | 50.62 | 30.78 | 组内实际调度不同，仍须看完整输出链 |

我方可见链：TopK完成约537.12 → checked585.46 → plan610.74 → QK/PV863.30 → merge897.48 → native_heads917.34 → HCpost1203.68。
Nal可见链：TopK约359.86 → plan378.18 → QK/PV510.00（包含头向量发布）→ HCpost671.38。
这些是逐函数跨窗中位数的边界示意，不是单次trace重建，也不能将阶段跨度直接求和。

我方前缀把HC、inputNorm、WKV、Indexer权重放在同一SPMD结束边界后再放行后续Q路径。
Nal的Norm、WKV和Indexer权重是较细的独立任务，可以更早放行部分消费者。数学精度复现成本和同步粒度都要分别验证。

## 首个隔离候选：只前移物理寻址

A：保持原checked、Native pair-swap、逻辑bias、K512 actual-K归约、probability round BF16、PV顺序、FFTS及最终inverse RoPE；额外输出物理行scratch，消费者仍逐行gather_row，只删除其重复分页算术。

这版只检验控制放置。它还没有复用checked/pair已有的page读，不能预设减少总查表或带宽。
范围限定pages小于2^24，INT32物理行显式转换；最多增加512KiB GM。浮点与FFTS的53个敏感调用AST保持一致。

A先做完整根CPU编译，再同输入Native/冻结/A逐位对拍和完整层计时。通过后才把64项物理索引DMA+mgather作为B因素。

### 单因素硬件结果

全部为同任务12窗×100次完整图中位数，每组先Native/Frozen/A或B的八边界A/B/A逐位对拍。

| 独立任务 | 冻结 | A：预解码 | B：再加mgather | 结论 |
| --- | ---: | ---: | ---: | --- |
| A首轮 | 1314.51 | 1313.06 | — | A仅快0.11%，7/12窗领先，无稳定收益 |
| A/B同轮 | 1309.79 | 1308.45 | 1338.37 | A仅快0.10%，7/12；B慢冻结2.18%，0/12 |

A/B均与Native在八类外部边界、A/B输入和恢复下逐位一致。B实际完整根CCEC通过，生成核确有 `MGATHER<Coalesce::Row, GatherOOB::Zero>`。
**两版都不接入正式源。** A控制前移没有证明收益；B在当前K512消费者布局下稳定回退，不能用Nal同类源码推断本分支也更快。
没有达到完整层稳定更快的门槛，因此本轮不扩大到真实整模型8K/B40或128K/B24，也没有新增192-token验收结论。冻结原环境的既有token验收保留，不扩展为新SDK/候选的整模型证明。

## 后续更值得验证的路线

1. **合并最终heads发布，数学保持原样。** 我方当前QK/PV后另有merge_norm与native_heads阶段；Nal在attention内发布。可在AIV持有最终m/l/左右oi时，执行原Native除法→BF16→FP32→inverse RoPE→BF16，直接写Native布局，省去packed中间表示与再复制。先只改变存储与任务边界，不改任何归约；UB实际liveness、padding零行与逐位验收仍是必需门禁。尚未实现或测量，不预报收益。
2. **在K512数学上重排搬运/等待。** 我方每块QK→等probability→PV，Nal用提前QK、滞后PV与三槽流水。保留Native的SWA128+compressed512、一个FP32 PV累加器及actual-K求和树，只尝试把独立QK/采KV与前块PV重叠；不得移植Nal K128局部softmax来换速度。需独立事件模型与硬件对拍。
3. **缩小前缀放行粒度。** 当前同一SPMD前缀完成WKV和Indexer权重之后，后续Q才能开始。对照Nal把归一化就绪后的独立分支提前放行，同时保留HF32等价DSL与Native权重/归约顺序；不能把原生数值边界的额外计算成本说成全是同步。

本轮证据支持优先研究Sparse整段和输出链，而不是继续堆叠寻址微调、以task更少推断更快，或把全SPMD当作自动收益。

## PMU能说明什么

四臂独立PipeUtilization采样后状态复核通过。外层Simpler kernel的AIV Scalar占用比：我方49.8%、Nal多task55.5%、Nal默认SPMD71.8%。
这不是纯轮询比例：Scalar含控制和地址计算，且与Vector/MTE可重叠。CANN只看到完整Simpler外壳，**不能据此拆出QK/PV内部计算、MTE2或纯等待的微秒**。
L1单核持续时间同样包含同步。本轮已经定位最大可见阶段，但没有宣称完成其内部精确分账。Profiler报告缺少ACL→NPU flow关联；有效硬件计数与CSV保留，不把缺失flow当代码失败。

## 工具与失败记录

所有NPU任务最终均已结束；最终状态、启动时间和完整日志见证据包。失败包括启动参数校验、Nal Hadamard应从顶层metadata取得、只读Artifact恢复适配及kernel-map AST观察器。均未修改正式数学或SDK。
A首次CPU解析发现INDEX赋INT32不一致，显式限定页范围并cast后完整编译通过；没有占卡绕过编译门禁。SDK Artifact.load会重新编译，DFX等使用load_prebuilt验证后的只读恢复，记录binary/manifest SHA；不冒称v5事后补存了当时不存在的binary收据。

## 复现与证据

详见[证据与复现说明](evidence_tnd_precision/nal_full_compare_20261021/README.md)：工具及固定参考源、raw JSON、逐窗记录、独立L1/dep/PMU CSV、生成码、编译门禁和独立审查。所有结果区分正式精度源、隔离候选和工具版本。

13个设备任务已全部结束；7个exit0，5个启动/绑定/观察器失败，1个主动停止。成功任务实际launcher耗时144～320秒，均未超过15分钟。正式实现没有新增数学改动，原精度基准保持。
