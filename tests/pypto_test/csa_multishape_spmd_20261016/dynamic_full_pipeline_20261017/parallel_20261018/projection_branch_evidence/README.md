# 动态CSA：投影分支分核候选

2026-10-18。候选在正式 `ops/pypto/deepseek_v4_flash_csa_spmd` 中开发并验证，
完整源码保存在本目录 `source_candidate`，原版在 `source_reference`。
完整L1确认退步后，正式目录已恢复原版，不保留失败策略入口；本目录供历史复现。

## 已有证据与本次假设

最新同进程128K/B16/S6完整L1：原多task 877.84μs、交错动态候选889.96μs、
静态参考787.30μs。交错方案没有改善，不能靠再次重测改变结论。

历史静态 `b16_row_deps_stage_clock_v1/stage_clock/stages.json` 显示，静态版
20名AIC做主Q_B，部分其他核在这段期间执行query Hadamard/KV，较早推进Indexer
输入链。正式动态版本使用24核串行执行多个独立阶段；AIV也先做query RoPE及
head权重，再处理更早就绪的Indexer压缩。这是可直接核对的调度差别。
历史时钟与当前不是同一次测量，不能把它们的端点相减当作可省耗时。

## 实现与动态合同

候选新文件 `source_candidate/projection_branches.py` 提供完整中文总览和实现：

1. 主压缩投影前移，仍使用原24核owner及K归约。
2. 前4名AIC执行head权重、query/compact Hadamard及KV；后20名AIC做Q_B。
3. query Hadamard每四token用M256，保持每行K128，真实尾行用valid_shape裁剪。
4. AIV先完成已就绪的Indexer pool/commit/norm，再等query投影做RoPE及weights。
5. 原phase布局、全部真实producer发布、共享prefix读完集合及缓存维护保持。
   无业务的阶段也只由本核发布一次空完成；原AIV等待集合包含新的全部producer。

T/B、Native Q/L、position、页表、compact count全部保持动态；没有测试点白名单。
4或8核是可比较的资源分配策略，当前只验证4核候选，不预设最终启用区间。
原多task的输出、Top-K/score、完整cache/state、保护区及图重放仍是精度判据。

## 验证顺序

- `run_cpu.sh NEW_OUTPUT`：无样例shape的完整符号编译，核对单MIX及58动态轴。
- `run_device.sh NEW_OUTPUT --batch 16 --history 131072 --launches 3 --l1-windows 5`：
  必须经 `task-submit --device auto`，先A→B→A/图与保护检查，再做三方完整L1。
  静态参考缺失的规格不能借邻档；B28按之前约定只保留已取得的比较项。

只使用已构建的独立PyPTO内存修复worktree和Simpler参数扩容worktree。
本候选未采用新PTOAS并行赋值补丁，也不修改共享安装。

## 已执行的过程与边界

1. `cpu_v1`：Tensor级matmul结果误用Tile级store，解析时报TensorType不符。
   改为Tensor级assemble，不改变有效行或数学。
2. `cpu_v2`：两个互斥constexpr位置各定义idx_requests，后续使用时SSA报告
   分支局部变量逃逸。将共同的设备shape读取提升到分支外，不新增Host取值。
3. `cpu_v3`：`max(worker-R,0)`经简化后无法通过NZ偏移的非负证明。改为后部
   连续24-R个核对24-R取模的循环置换；覆盖同一逻辑worker集合，原AIV仍等待
   全部实际Q_B生产者。没有关闭NZ合法性检查或改权重布局。
4. `cpu_v4`：完整符号编译/PTOAS通过，单MIX、sync_start、130个运行时Tensor
   ABI项、58个动态轴，不传样例shape。`hadamard_valid_shape_codegen.json`
   保存M256读取、矩阵输出和写回均保留有效行的实际PTO片段。
5. `check_prefix_control.py`执行正式源码的前段控制表达式，450例覆盖T=1..513
   的代表边界、S1/S5/S6构成的请求数、零/单/双compact行、4/8辅助核策略。
   全部发布唯一、等待有来源、依赖无环，NZ Q_B的128列块无遗漏或重复。
   它只检查前段Scalar控制，不执行Tile数学、缓存或FFTS，不能替代NPU。
6. `task_20261018_114949_268532326529`：128K/B16/S6，4辅助核候选。
   A→B→A三次已通过原多task的精确数值、完整state/cache、保护区及阶段协议。
   最终exit0，图重放及五窗完整L1已完成，候选退步，见下表。

失败摘要与完整本地结果的路径/大小在 `CPU_ATTEMPTS.json`。v2的大型重复IR诊断
本地保留，不把它当最小复现或额外代码修改提交；本轮没有修改依赖仓库。

## 完整L1结果：不采用4辅助核候选

| 实现 | 五窗完整L1（μs） | 中位数（μs） |
| --- | --- | --- |
| 原动态分task | 878.50 / 890.20 / 880.06 / 872.78 / 893.20 | 880.06 |
| 动态单SPMD，4辅助核 | 946.46 / 946.96 / 947.16 / 958.24 / 950.28 | 947.16 |
| 历史静态单SPMD | 781.94 / 785.94 / 778.68 / 778.52 / 785.96 | 781.94 |

候选比同轮原分task慢7.62%，默认保持关闭。上述数值由每窗原始记录的
最早incore开始和最晚结束复算，50MHz时钟，不取阶段时钟或最小值代替。
输出、Top-K/score、完整六份cache/state、信号代际、Native/工作区保护区和
基本graph replay均通过。未覆盖S1/S5、混合长度、策略边界及其余六档。
完整结果在 `h131072_b16_branch4_v1/result.json`，摘要在 `FULL_L1_RESULT_V1.json`。

为区分前段改善与后段退步，提交同进程、同根Tensor的两版动态SPMD阶段时钟诊断
`task_20261018_115704_29584807981`，输出 `h131072_b16_branch4_clock_v1`。
双方均使用相同独立PyPTO；A→B→A三轮，不产生完整L1达标结论。
任务最终exit0，三轮两版的数值、cache/state、保护区和图重放通过。结果由
`summarize_phase_clock.py`汇总为 `PHASE_CLOCK_RESULT_V1.json`。

阶段端点取各核最早BEGIN到该phase最晚notify完成的时间，以下为三轮中位数。
这是插桩就绪时刻，不是阶段数学耗时，也不能代替上面的完整L1：

| 阶段就绪 | 原动态SPMD（μs） | 4核候选（μs） | 后移量（μs） |
| --- | --- | --- | --- |
| Indexer compact norm | 195.30 | 148.46 | -46.84 |
| query Hadamard | 175.06 | 225.68 | +50.62 |
| KV投影 | 164.02 | 288.10 | +124.08 |
| 原始KV写回 | 232.02 | 313.38 | +81.36 |
| Indexer query量化 | 245.16 | 325.84 | +80.68 |
| Indexer key/scale就绪 | 275.18 | 350.72 | +75.54 |
| 主Q_B | 245.38 | 245.54 | +0.16 |
| 主压缩投影 | 287.50 | 182.82 | -104.68 |
| 主压缩最终输出 | 330.30 | 403.28 | +72.98 |
| Score Vector结束 | 542.98 | 621.14 | +78.16 |
| 最终输出 | 902.38 | 973.54 | +71.16 |

源码与时钟共同说明：主压缩前移、Indexer norm提前确实生效，但它们没有自动
缩短关键链。候选把KV放在辅助4核的weights→query Hadamard→compact Hadamard
之后；AIV仍按KV后处理→query量化→系数/compact key→主Query/压缩的顺序。
于是较晚的KV阻断了已具备其他输入的Indexer量化，把约75～80μs的延迟传到
Score和后继链。主压缩投影虽然提前约105μs，其AIV消费者仍迟到，收益被抵消。
Q_B就绪中位数基本不变，不支持把退步主要归因20核Q_B负载。

这些是本候选的直接时序证据；编译布局和插桩也可能影响局部耗时，未将全部
71.16μs归为某个单一操作。下一版必须同时处理AIC producer和AIV consumer顺序，
不能只把4辅助核改成8核反复扫参数。当前失败方案完整留档，未推广到其他档位。

## 905.08与880.06的原分task基线来源核对

用户指出历史128K/B16约905μs，而本轮约880μs。只读核对，未新测：

- 旧任务 `task_20261017_044912_40225339550`，历史静态对照：
  原分task三窗888.48/905.08/909.80，中位905.08μs。
- 本次任务五窗如上，中位880.06μs，相差25.02μs（2.76%）。
- 两者均为正式第4层W8A8、H131072/B16/S6、设备0、完整L1 incore首尾；
  原始记录均为50MHz、66个task、973条核记录、72个核，未减少统计范围。
- 冻结参考包与当前包的23个CSA Python文件逐字相同，权重加载记录相同。
  顶层独立hc_pre.py虽有改动，本次分task调用的是CSA子包自己的hc_pre.py。
  编排C++仅归一化源码路径和inline编号后相同。
- 旧PyPTO来自static-fixes worktree，新PyPTO来自memory-reuse worktree；
  保存的67份PTO IR中38份在上述归一化后相同，29份有差异；73份核C++中
  38份相同，35份有差异。不能声称两轮使用完全相同的编译产物。
- 旧流程两角色AB/BA，5次预热加20次Event采样后采3窗L1；新流程三角色轮换，
  3次图预热后采5窗L1。没有锁定或记录足以归因的全部频率、cache及设备状态。

因此当前只能确认测试轮次、编译环境和测量流程不同，尚未分离25.02μs的具体
成因。不能直接归因编译器优化，也不能直接声称普通波动。905.08历史值保留，
880.06只作为本轮基线，性能收益始终使用同轮分母。复核脚本
`audit_multitask_baseline.py`和结果 `MULTITASK_905_VS_880_AUDIT.json`保存证据。
