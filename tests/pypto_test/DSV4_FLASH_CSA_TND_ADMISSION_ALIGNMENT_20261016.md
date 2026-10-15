# TND 精度版：128K/B24 入场对齐与首轮模块对拍

测试日期：2026-10-15 至 2026-10-16。
分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
数学源码冻结在 `a9e0ae0d9f5bc60b59fc3b74342ca2b156401045`，本轮没有修改正式数学实现或性能版。
Native 指同一冻结 checkout 的原生 CSA 路径。

## 为什么先对齐入场

[此前的 128K/B24、192 token 测试](DSV4_FLASH_CSA_TND_128K_B24_20261015.md)中，
Native 与 CSA 的观测活跃请求峰值分别为 24 和 23。即使使用相同 bank、相同容量策略且零抢占，
也不能直接认定每步整模型输入相同。因此先固定实际 KV 分配，记录真正执行时的联合输入，
再定位浮点误差；没有继续扩大配置。

## 本轮配置与观测

| 项目 | 口径 |
| --- | --- |
| 模型 | DeepSeek-V4-Flash-0731-w8a8 完整 75 分片权重 |
| 并行与提交 | TP1、DP=EP16，每卡提交前24条不同问题，允许等待入场 |
| bank | 冻结的 fixed131072，实际 history 为131104–131346；与前轮相同 |
| 输出长度 | 64 token，仅用于缩小定位范围 |
| 解码 | greedy、DSpark提出5验6、ignore EOS |
| 图 | FULL_DECODE_ONLY，档位 `[6,144]`，两侧event模式1 |
| 数值开关 | 确定性level1、HCCL true、atomic0、NZ2、EPLB关闭 |
| 实际共同容量 | 24000 blocks，47个KV tensor，含逐tensor对齐余量的计划上界21,955,846,144 bytes（20.448 GiB） |
| 运行边界 | 每项单独提交，上限780秒；SDK、READY及13个官方静态包复用 |

初始100000 blocks因容量估计错误在KV初始化OOM；修正到26000后，实际计划为22.15 GiB，
被分配前21 GiB诊断门禁拒绝。两者均未进入生成，不能当作CSA数学失败。
最终24000的计划在Native与CSA逐项相同，并通过实际free额外预留2 GiB的检查。

记录源为实际 `input_batch` 顺序及设备上的token ID、position、seq_lens，
并包含query bounds、入场/完成/抢占请求、完整padding输入、graph档位及DP token向量。
观测从已有的暂停入队/DP barrier之后开始，包含idle/dummy forward；
审计要求16个rank的轮数和逐轮DP向量一致，再跨运行逐字段比较。
`execute_index`仅作rank本地辅助，不能代替联合forward轮次。
D2H观测会改变时序，因此本轮不提供性能结论。

## Native 独立重复与实际入场

| 检查 | 两次Native结果 |
| --- | --- |
| 联合forward轮数 | 每rank32轮，其中30次FULL144 |
| 实际活跃请求峰值 | 每rank21条 |
| 入场映射 | q00–q20首轮入场；q21–q23在零基第14轮入场 |
| 完整实际设备输入/联合轨迹 | 16rank全部逐字段相同 |
| 输出与接受事件 | 384条副本请求、24,576 token，完整序列和接受事件均相同 |
| 平均接受步长 | 3.2472826087；仅代表本轮64 token口径 |
| 抢占 | 两次均为0 |

该结果证明本轮Native输入轨迹可复现，没有推断其他batch或192 token也具有相同轨迹。

## CSA 首个输入分叉

- 16个rank的零基第0–24轮，所有记录的实际输入及入场轨迹均与Native一致。
- 真正首差是零基第25轮：`input_ids`、`positions`、`seq_lens`及对应padding数组不同。
- 当轮请求仍为q22、q21、q23；请求顺序、query bounds、graph档位及DP向量没有先发生变化。
- 不同输入在进入模型前被拒绝；没有执行该分叉轮次。
- 旧记录器随后又收到已排队的第26轮，拒绝执行但覆盖了STOP文件。
  通过原始trace和异常日志确认首差为25，后续记录器已增加失败锁定和独占STOP写入，
  CPU回归确认不会覆盖首差、追加后续记录或再次执行模型。

这将定位窗口缩小到此前相同输入的计算，但不能把第25轮称为“首次引入浮点误差”。
它也不能直接证明某一个CSA模块是分叉原因；target/draft边界仍需继续检查。

## 同输入首轮模块对拍

在真实forward0的21个有效token上，只对rank0做Native shadow。
原始CSA先执行；保存初始和CSA后写入页，Native前恢复初态，Native使用独立输出，
随后恢复CSA后的页并逐bit复核。没有新增条件性Gloo集体通信。

| 观测边界 | 结果 |
| --- | --- |
| CSA层 | 21层，layer2、4、…、42，记录完整 |
| 输入归一化、Native KV投影发布值、Indexer权重 | 全部逐bit一致 |
| SWA KV、压缩KV、Indexer cache、两类compressor state物理写页 | 全部逐bit一致 |
| TopK、inverse RoPE后heads、O-proj、attention半层输出 | 全部逐bit一致 |
| hidden保护、五类cache恢复、原CSA输出保护 | 全部通过 |
| 每层比较项 | 20项，包含恢复/保护检查；21层共420项 |

这次hook运行后，后续仍在第25轮出现相同输入分叉；新的锁定器保留了真正首差。
本项首轮graph mode为 **NONE**，不能写成“FULL下21层全部逐bit一致”，
也没有检查未导出的每一个内部Q/QR tensor。

## 后续定位门禁

首轮未发现不同，下一阶段定位后续FULL replay。
先在原始Native图返回后记录最终hidden、aux hidden及MTP residual的位级指纹；
随后增加图内边界复制，要求相同实际输入下这些有效输出仍逐bit一致。
只有观测保真和实际replay刷新检查通过，才用同样观测与CSA逐模块比较。
发现首个不同边界后停止后续step，再按Native实现修正。

截至本记录，**FULL模块逐bit精度、128K/B24完整192 token验收均未通过证明**。
原有192 token失败结论仍保留；本轮没有引入新的数学修复或性能优化。

## 环境与证据

CANN9.2 beta2、ATB9.2 ABI1、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1，
PyPTO `3e87a843`、Simpler `a54c0509552b01e13fb0960e23ce409a01b5024f`。
沿用冻结的私有运行包和模型bank，未核验与nalinaly原始bank逐字节相同。

[结构化结果与逐层比较](evidence_tnd_precision/admission_alignment_firststep_20261016.json)
包含源码、SDK、bank清单SHA及21层完整指标。
原始小型证据包包含实际输入、日志、比较器及诊断脚本，没有将大tensor或编译缓存加入Git。

- 联合输入三臂证据SHA256：`4c73a399130490c194980e437e36ad9fa9f8743aa19f609cd92675c1a38a5cde`
- 首轮模块证据SHA256：`642beacbd1934253dc0da63dd7bd32dd867efae147b50ef1973fc95c8d14ab69`

CPU观测回归涵盖真实顺序、空/不完整参考拒绝、实际容量不匹配、idle插入、联合轮次校验、
首差锁定、当前forward上下文及模块回调失败后的停止。无设备环境预检与独立源码复核均通过。
