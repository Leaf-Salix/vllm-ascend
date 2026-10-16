# TND 精度版：128K/B24 入场对齐、FULL首差与Indexer修复

测试日期：2026-10-15 至 2026-10-16。
分支：`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
定位基线冻结在 `a9e0ae0d9f5bc60b59fc3b74342ca2b156401045`。
定位后仅修复精度版 `decode_indexer.py` 的多leaf临时分数与TopK根发布，性能版未改。
Native 指同一冻结 checkout 的原生 CSA 路径；修复候选相对该基线的其余源码逐字节相同。

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

以上是修复前的定位过程，当时完整FULL模块精度尚未通过。
后续FULL首差、修复及64 token回归见下文；原有192 token失败结论保留，尚未复测。

## FULL同输入首差：第24轮、第2层heads

已完成三个顺序独立任务：原始Native仅观测图外输出、增加图内观测的Native、使用相同观测的CSA精度版。
图内观测先通过Native保真门禁，再用于CSA归因。

| 检查 | 结果 |
| --- | --- |
| Native图内观测是否改变实际输入/入场/完整token及接受事件 | 16rank、32轮均未改变，零抢占 |
| Native图内观测是否改变target hidden、aux hidden或MTP residual | rank0全部32轮逐bit相同 |
| 图内观测资格 | 43层、21个CSA，6816份边界摘要及32份原值快照完整；逐字段epoch、poison、positions检查通过 |
| CSA在首差前的输出 | 零基第0–23轮，rank0所有观测边界及图外hidden/aux/MTP均逐bit一致 |
| 同输入资格 | 至第24轮，16rank实际输入记录与Native逐字段相同；原值SHA与摘要交叉核验通过 |
| 最早不同边界 | 第24轮，FULL144，18个有效token，第2层CSA的inverse RoPE后heads |
| 本轮此前边界 | layer0、1全部边界；layer2的输入、positions、normed，共9项逐bit一致 |
| 首差规模 | 589,127 / 589,824个BF16元素位模式不同；relative L2为1.53171959，max abs为4.125，无非有限值 |
| 停止 | 保存首差原值后停止后续model step；没有继续192 token或扩大参数 |

该轮q20完成，活跃请求从4条缩到3条，顺序为q22、q21、q23，每条6个token，padding档位仍为144。
仅凭缩批的时间关联，不能认定mask、页表或原生算子有bug。
CPU对已保存heads的比较排除了“上一轮结果原样残留”和“当前行简单重排”。
该误差量级明显超过舍入差异，后续先核查真实设备metadata、TopK与cache寻址/状态，再细分Q及attention。

观测修复也保留记录：首次Native witness在profile的400行输入复制到144行缓冲时失败，未进入生成。
动态`min`切片候选又被CPU编译门禁拒绝。最终采用只写私有buffer的观测custom op，
CPU回归直接复用同一捕获图处理6/21/144/400行；随后Native保真验证通过。
这些属于探针修复，没有改动CSA数学源码。所有硬件任务上限均为780秒。

## Indexer归因：长档缩批路径没有发布本轮TopK根

继续在真实第24轮的root入口，把设备metadata复制到独立持久缓冲区，并在原始FULL replay后读取
已有的TopK输出。query bounds为`[0,6,...,144]`；前3条seq_lens有效，其余为0；
18个有效token的position和slot有效，padding slot为-1。原始heads的首差SHA与前项完全相同，
没有因增加观测而改变故障。

该轮PyPTO TopK每行512项只有 **47–75个不同索引**。检查对应生产者与消费者发现：

1. 长档至少4组S6请求走`balance_leaves=True`，每leaf发布一个根。
2. 缩到3条请求后走双query的`balance_leaves=False`分支。此前短单leaf修正让分数写到连续query行，
   同时删除half-root发布；长多leaf也复用了这个分支，多个leaf会抢写同一query行。
3. 下游`indexer_topk_query_merge_one`在长档仍按每leaf两个half-root读取`pair_arena`，
   但该分支没有本轮publisher，合并读到了未初始化或旧内容。

这是本分支Indexer的生产/消费契约错误，不能归为Native bug或BF16舍入噪声。
额外Native eager shadow虽然产生512个不同索引，但没有通过独立Native FULL原值门禁，
因此未用作金标准或将其结果注入正式候选。最终比较仍使用已通过保真验证的Native FULL边界。

### 修复范围

仅修改`deepseek_v4_flash_csa_tnd_precision/decode_indexer.py`：

- 单leaf保留原连续query score布局和短档排序路径。
- 多leaf改为worker/query/AIV私有score行，半区内偏移从0开始，避免不同leaf抢写。
- 恢复`query * TOPK_ROWS_PER_QUERY + leaf * 2 + AIV`的half-root发布，匹配merge读址。
- 不完整query组只发布有效行；空半叶显式填负无穷，避免残留进入合并。

没有改变QK计算、缩放、舍入顺序或Native实现。修复文件SHA256为
`7d117c766f7cafeff379e0fbad4fa4176c999c3dd9226b14f86241c2fdc35023`。
独立源码清单核验确认候选只有此文件不同。CPU回归执行实际DSL的地址与发布语句，
覆盖B1/B2/B3、`(1,5,2)`变长query组和空半叶；5项通过，未修复源码的负控制5项全部被拒绝。
该CPU检查不替代排序数学及设备同步验证，后者由下面的真实整模型任务覆盖。

### 修复后同输入回归

| 检查 | 结果 |
| --- | --- |
| 实际入场与联合输入 | 16rank全部32轮与Native逐字段相同，含30轮FULL144 |
| rank0模块边界 | 43层、其中21个CSA，6816份边界摘要全部逐bit一致 |
| rank0图外输出 | 全部32轮target hidden、aux hidden、MTP residual逐bit一致 |
| 完整输出 | 384条副本请求、24,576 token，完整token序列及接受事件与Native一致 |
| 实际并发/接受/抢占 | 峰值21；平均接受3.2472826087；零抢占，与Native一致 |
| 原故障轮TopK | FULL144/T18，每行512个不同索引，全部位于各query有效范围 |
| 独立重复 | 相同源码重复任务通过；16rank均READY命中1次，生成/二进制编译/bypass/损坏/存储错误均0 |

模块边界是明确导出的层输入、positions、归一化结果、heads、O-proj及层输出等，不代表导出了所有内部tensor；
浮点观测范围为rank0，其他rank通过实际输入、完整token及接受事件核验。

| 阶段 | 任务 | 结果 |
| --- | --- | --- |
| 额外metadata/TopK定位 | `task_20261016_012900_59571325267` | 首差仍为第24轮heads；TopK结构异常，按门禁停止 |
| 无设备生成/增量编译 | `task_20261016_015300_117095229502` | 约19.7秒；复用SDK，没有重装框架 |
| 首次修复回归 | `task_20261016_015811_127824410229` | 数值完整通过，约7分42秒；首次真实调用由rank9构建1次，不记为零编译资格通过 |
| 严格READY重复回归 | `task_20261016_021041_160980929476` | 约8分42秒，exit0；同输入精度与全部运行门禁通过 |

第一次无设备预编译仅通过签名级缓存，未保证真实调用已预热；因此保留首次回归的构建记录，
没有放宽READY门禁。独立审计还绑定两侧任务的源码清单、候选实际snapshot、32轮完整witness及逐轮有效token数。

**当前通过范围：本固定bank、128K/B24提交、实际峰值21、64 token生成。**
128K/B24的192 token尚未复测；本轮没有新增长档不等长query矩阵，也没有性能测量。
因此不能由本结果推出所有配置已逐bit一致，或覆盖先前192 token的失败记录。

## 环境与证据

CANN9.2 beta2、ATB9.2 ABI1、Torch2.10.0、Torch-NPU2.10.0.post2、vLLM0.25.1，
PyPTO `3e87a843`、Simpler `a54c0509552b01e13fb0960e23ce409a01b5024f`。
沿用冻结的私有运行包和模型bank，未核验与nalinaly原始bank逐字节相同。

[结构化结果与逐层比较](evidence_tnd_precision/admission_alignment_firststep_20261016.json)
包含源码、SDK、bank清单SHA及21层完整指标。
原始小型证据包包含实际输入、日志、比较器及诊断脚本，没有将大tensor或编译缓存加入Git。

- 联合输入三臂证据SHA256：`4c73a399130490c194980e437e36ad9fa9f8743aa19f609cd92675c1a38a5cde`
- 首轮模块证据SHA256：`642beacbd1934253dc0da63dd7bd32dd867efae147b50ef1973fc95c8d14ab69`
- FULL首差小型证据SHA256：`18be126cae8c0517f96fd5d11afa8499c508651103bbdafea1cc3b45da853070`
- Indexer归因与两次修复回归小型证据SHA256：`75016bd5aac7474da635b7359e43e0c35f2aa40f30623c01bb0482c9a2c325e4`

CPU观测回归涵盖真实顺序、空/不完整参考拒绝、实际容量不匹配、idle插入、联合轮次校验、
首差锁定、当前forward上下文及模块回调失败后的停止。无设备环境预检与独立源码复核均通过。
