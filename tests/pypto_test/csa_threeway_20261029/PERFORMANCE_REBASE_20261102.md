# 真实图容量下重新定位我方与nalinaly差距

## 结论和版本

此前物理cache MGather优化在单层降低约2.5%～3.1%，但整模型8K的
77.652 ms没有优于旧77.079 ms，不能作为整模型优化成功。
默认精度基准`65bebf7f`保留。本轮先重做归因，再测试独立候选。

直接对比的是我方`65bebf7f`和nalinaly`1574ed59`，使用相同SDK和
Python/Simpler入口。nalinaly最新公开`2d05303d`的整网报告另用了
ACLNN和SuperKernel，属于不同后端，不能把其整网收益全部算作核内收益。
见[其最新SK对照报告](https://github.com/nalinaly/vllm-ascend/blob/2d05303d613a901577a58c24c341e07967444fb7/tests/pypto_test/csa_aclnn_host_20261030/SK_AB_RESULTS_20261031.md)。

## 新测量：实际48行、物理240行

既有整模型观测确认：主要FULL解码形状是8个有效请求、48个有效token，
但图容量为240行、40个请求槽位。因此不能直接沿用全有效B32/T192的
attention归因来解释整模型结果。

本轮使用真实第4层权重、合成共同hidden/history，8K历史，8个有效S6请求，
其余32个dummy请求长度为0、slot为-1；物理token容量240，边界覆盖全部240行。
它复现主要张量容量和有效工作量，未声称复现真实模型的全部hidden/cache内容。
双方使用同一fixture、同一物理8卡、确定性环境、NZ2、atomic关闭。

计时为完整FULL graph回放：5次热身后30次交替A/B；cache恢复、同步、
只读输入检查都在Event区间外。每轮从相同cache恢复。

| 实现 | 中位数/μs | 均值/μs |
| --- | ---: | ---: |
| 我方冻结精度TND `65bebf7f` | 1364.03 | 1363.93 |
| nalinaly性能TND `1574ed59` | 994.79 | 993.43 |

30/30个交替样本中我方更慢；中位数差369.24 μs。
两边各自的eager/FULL输出有效行及全部物理cache均逐字节一致。
这不是双方对Native的新增精度验收，不能据此认可nalinaly的数值精度。

## 阶段归因

仅诊断副本增加已授权时钟插桩；原生150-tick退避保持不变。
CPU审计代码、加载二进制身份复核均通过；插桩前后双方各自的
eager、FULL及5轮重复FULL的输出有效行和全部物理cache逐字节一致。
普通计时使用未插桩版本。

下表取5个诊断窗口中各阶段最后完成时刻的中位数，单位μs。
这是累计进度，不是独立子模块纯计算时间；存在跨核/跨阶段重叠，不能相加。

| 完成里程碑 | 我方 | nalinaly | 累计差距 |
| --- | ---: | ---: | ---: |
| INPUT：输入准备和HC打包 | 96.52 | 22.98 | 73.54 |
| HC组合完成 | 228.52 | 84.46 | 144.06 |
| TopK完成 | 643.98 | 419.98 | 224.00 |
| Attention Vector完成 | 843.84 | 572.02 | 271.82 |
| HC-post/OUT完成 | 1166.02 | 821.92 | 344.10 |

在这个形状下，到TopK为止已产生224 μs差距，attention之后累计再增加
约48 μs，后缀再增加约72 μs。与旧全有效B32的归因不同，优先级应先转向
HC/前缀，而不是只优化attention cache搬运。

累计GM等待包含上游等待，不能逐阶段直接相减后当作纯计算。
INPUT之前没有这类GM等待，仍已有约74 μs差距，说明不能仅靠减少轮询解释。
其余耗时仍包含计算、搬运、硬件事件等待和流水气泡。

## 数值边界与可借鉴优化

- nalinaly的attention采用5个K128块及三槽交错，QK、PV、搬运跨工作项重叠。
  我方为保留Native的softmax/PV累计边界，采用SWA128和CMP512两块。
  直接复制五次在线更新会改变数值顺序；可以借鉴缓冲区和生产消费流水，
  但不能直接替换其数学分块。
- 我方HC额外保留Native HF32转换、高低位打包和22段累计顺序。
  其中常量权重每次打包没有必要依赖请求，适合首先尝试移到图外。
- 后续attention候选优先考虑同query的双KV/score缓冲与QK前瞻，
  保留SWA→CMP的softmax、概率舍入和PV累计顺序；先验证缓冲复用依赖，
  不直接套用nalinaly的跨query三槽覆盖。

## HC常量预打包候选

独立目录`hc-prepack-20261102-v2`，不修改默认算子。
完全复用原`hc_pack_weight` DSL，仅在FULL capture前执行一次。
完整CSA根新增packed权重参数，移除每次Vector中的同一打包调用；
48个INPUT发布和Cube等待保留，HC投影、Norm和后续算术均不改。

代价为每个绑定常驻2 MiB；21个CSA层、每层一个绑定约42 MiB，
不同图档绑定不能无条件重复分配并忽略显存代价。
使用版本可追踪的固定权重，检查对象、地址、offset、shape、stride、dtype、
device与版本；图回放前后也在计时外检查。图生命周期内权重必须不变。
图外对完整2 MiB布局做整数HF32转换和BF16边界的CPU参考逐字节检查，
packed加入只读门禁。源级逆变换证明根入口仅增加外部常量、替换原分配。

CPU预编译、完整2 MiB参考布局及packed只读检查通过。
单卡`task_20261102_212359_14433134020`完成并退出。

| 实现 | 中位数/μs | 均值/μs |
| --- | ---: | ---: |
| 同组冻结基准 | 1347.70 | 1348.28 |
| 图外HC权重打包 | 1345.74 | 1348.66 |

中位只降低0.15%，仅13/30对更快，均值略慢；不具备稳定收益。
两边各自eager/FULL输出/cache一致，但按性能优先策略没有继续跨版本/Native
精度验收，不合入。这个负结果也说明不能把INPUT累计差距都归因于权重打包。

## HC输入BF16布局候选

独立目录`hc-input-bf16-20261102-v2`，不组合上一负收益候选。
只在T为16倍数的完整行块中，将原BF16→FP32→布局复制/转置→BF16改成
BF16布局复制/转置。HC平方和、HF32权重、22段投影归约及全部同步保持。
任意非16倍数T仍执行冻结的旧helper，保留尾行清零；不要求请求补S6。

v1在CPU阶段重现已有SDK的BF16 Tile/FP32 Scalar补零类型错误，未上卡。
v2只为可证明没有尾行的形状启用快路，其余T完全保留旧函数，
反向AST检查证明旧路径和其他模块没有改变。CPU构建通过。

任务`task_20261102_213745_17375508012`在物理10卡完成退出：

| 实现 | 中位数/μs | 均值/μs |
| --- | ---: | ---: |
| 同组冻结基准 | 1353.19 | 1354.41 |
| 输入BF16布局快路 | 1340.44 | 1343.96 |

中位降低0.94%，21/30对更快，尚未达到预设≥1%且≥24/30的稳定收益门槛。
未继续Native精度验收，不合入、不与其他候选叠加。
这里只证明普通样本的性能趋势，不声称subnormal、NaN或特殊BF16载荷等价。

## 删除已被覆盖的旧RMS候选

新检查发现`vector_schedule.py`调用链为：

1. `hc_widen_rms_range`完成BF16加宽，同时计算旧32×K512平方和和rsqrt写inv。
2. Cube完成22段Native-order平方partial。
3. 同一AIV的`hc_input_rms_range`重新计算并覆盖inv。
4. `hc_gates_range`和`hc_comb_range`才首次读取inv；Cube没有inv输入。

生产者和消费者都按M8/48条带分工。对于T≤256，Native RMS覆盖到
ceil8(T)，包括最后不足8行时gates的整块读取域；PAD16多出的行不被消费。
独立源审查确认旧RMS数值无消费者。实际生成C++中仍存在该段
`w_inv_inline`的TRSQRT，说明没有被编译器自动删除。

`hc-dead-rms-20261102-v1`只删这遍旧平方、归约、rsqrt；保留原BF16加宽、
尾行scratch与有效行GM store，在原inv写域写FP32零维持Out契约，
随后仍由Native RMS完全覆盖。所有发布/等待和实际Native RMS算术均不改。
该证明只适用于当前SPMD调用链，不能改变旧helper独立使用的数学契约。
CPU构建通过，生成物中TRSQRT从5处降为4处，旧`w_inv_inline`消失。
任务`task_20261102_214853_194431522565`在物理8卡完成退出：

| 实现 | 中位数/μs | 均值/μs |
| --- | ---: | ---: |
| 同组冻结基准 | 1353.19 | 1354.43 |
| 删除旧RMS | 1340.69 | 1343.21 |

中位降低0.92%，25/30对更快；仍未达≥1%的门槛。
不合入，未继续跨版本/Native精度验收。不能用源级dead-work证明代替设备精度资格。

## 尾部dummy平方块：仅CPU，禁止硬件执行

独立`hc-padding-square-20261102-v1`尝试只跳过最后有效请求之后的M16平方块；
部分有效块保持原算，原22段及K32顺序保留。它未组合上述任何候选。

独立审查发现两项边界：

- 单用Indexer Q/L不能证明所有原生metadata组的活动域相同。
  诊断runner已增加五组Q/L逐元素相同的前置条件；这仍不能作为任意生产绑定的资格。
- AIC内的零矩阵填充虽然CPU编译通过，却新增了第三个核角色：
  `cube_schedule_aic.cpp`、`cube_schedule_aiv.cpp`、`vector_schedule.cpp`。
  继续核验发现，`kernel_config.py`只注册了拆分后的Cube AIC/AIV，
  orchestration的顶层`MixedKernels={0,1,1}`没有单独注册原`vector_schedule`。
  新AIV内部是否包含原Vector逻辑及其完成顺序尚未完整审计，
  不能断言确定死锁，也不能将这份新编排视作协议通过。
  PTO的零矩阵确为Vec `TEXPANDS`/store，不能当作合法AIC零写。

因此未提交NPU测试，也没有性能/精度结论。
直接生成物与hash见`evidence/hc-padding-square-role-audit-20261102.json`。
manifest已标记`hardware_blocked_reason`，runner在首次设备执行之前拒绝；
远端对应硬件启动器也停用。CPU成功不计作SPMD协议通过。
无设备任务`task_20261102_221214_237126224574`实际验证了拒绝路径：
`passed=true`、`rejected_before_npu_init=true`，见
`evidence/hardware-block-check-20261102.json`。该测试没有占用NPU。
收尾时rsync连接被重置；这份小收据转录自已通过SSH返回、相互一致的
远端JSON和任务stdout，文件中显式注明来源，不伪称原文件已传输成功。

## 下一步和当前交付状态

本轮只提交测试工具、报告和证据，默认`65bebf7f`不变。
三个已实测候选分别为0.15%/13对、0.94%/21对、0.92%/25对，
均未达预设稳定收益门槛，不包装成已完成的性能提升，也不重复跑整模型。
既有8K token一致性只属于原已验版本，不能替这些新候选作保证。

下一轮优先级：

1. INPUT内部单独测量加宽/旧归约、输入打包、权重打包；
   HC线性投影与22段平方也分开取时。不要再把累计约74/144 μs差距全归给其中一个helper。
2. 在相同Native乘加/归约边界下，优先优化真正占时的打包与矩阵布局，
   对比MTE搬运、循环开销和有效计算，而非继续盲改低占比末端。
3. 若做dummy裁剪，先由全部相关Native metadata证明活动域，
   在现有AIV角色内完成受保护的零写，再保留原AIC完成信号，避免隐式增加核角色。
4. 单因素普通性能稳定胜出后，先冻结版输出及完整cache逐字节、
   再正确的Native满容量捕获/同址更新资格，最后才安排真实8K token验证。
   227仍禁止14和0–7；八卡整模型需另获明确授权，不重用旧例外。

## 复现和证据

环境使用项目公共`env.sh`和`--shared-toolchain` bootstrap；
PyPTO `d44925c`、PTOAS `914863ed`、Simpler `57e499`，CANN9.2 beta2，
Python3.12、Torch2.10、Torch-NPU2.10.0.post2、vLLM0.25.1。
不重装SDK，不修改生产生成代码。诊断仅替换已授权的时钟测量点。

```bash
# 仅列runner核心参数，外层须使用共同环境、bootstrap和task-submit。
python "$r/src/tests/pypto_test/csa_threeway_20261029/paired_single_layer.py" "$r" "$out" \
  --batch 8 --token-capacity 240 --request-capacity 40 \
  --clock --clock-cpu-label 20261102-real-geometry-v1
```

时钟CPU预编译使用`phase_clock_compile.py --batch 40 --reuse-source`；
其policy与物理240/40一致。普通版本先分别`--cpu ours`和`--cpu nalinaly`。
所有CPU构建用ccache和四路并发，不占NPU。

此次padded Native金值不能直接捕获48行Native eager路径来代替。
严格容量精度复核必须使用既有`probe_padding.py`的
“满40/240捕获→同址更新为有效8/48”协议。
因此runner明确禁止`--native-check`与padding容量选项组合，避免误判。

任务`task_20261102_211712_118353519175`完成并退出，仅使用物理8卡；
外层300秒、内层240秒，未占用14或0–7。
首个诊断任务因RoPE代理接口错误在约40秒退出；修复脚本后才获得上述数据。
候选首次CPU编译拒绝新增参数的In/InOut不一致，未提交其硬件任务；
v2按实际SSA返回链声明InOut，packed实际只读仍需设备门禁证明。

已完成的阶段归因和权重预打包证据归档为
`evidence/performance-rebase-20261102-part1.tar.gz`（964268 bytes），
SHA256：`75d8fd9f29482ca019ffab92dde4c5b67c5aacb5b12942fc11583f8dfcfcd849`。
包含普通样本、10份原始阶段时钟及Perfetto轨迹、CPU审计收据、冻结runner和启动器。

其余候选证据为`evidence/performance-rebase-20261102-part2.tar.gz`（28587 bytes），
SHA256：`ae210aa91cb372133212b2fde516cd5e9a72f0cb292deb3a29026b00269b4d6c`。
包含输入BF16布局/死RMS的A/B样本、CPU收据、冻结runner，以及剪枝候选的硬件阻止标记。
成功NPU任务分别约82、61、70、67秒；所有本轮任务已退出，未留下排队任务，
未使用物理14或0–7，也未跑新的整模型或128K。
