# 恢复原派发语义：CSA本轮受益，无后继开销仍未达标

本轮只在私有环境修改，公共算子及共享安装未切换。按用户要求保留功能候选供讨论，
在无后继快路径达到基本无开销前，不声明整项性能验收通过，也不继续扩大调度改动。

## 改动与范围

- 删除为交接优先选择busy前驱核的策略，恢复原idle/pending候选集合。
- 删除ACK后的额外派发，保持原有ACK处理流程。
- Set不再隐式开启`allow_early_resolve`。CSA普通版与候选的63处设置逐项一致；
  除内部能力字段及自动变量的一一对应重命名外，两份orchestration代码相同。
- Set只声明尾部等待由executor负责。无合格的已dispatch后继时执行普通`PIPE_ALL`，
  有真实交接时在后继Wait执行同一等待，完成后才FIN。去掉尾部Event7和Set的GM状态写入。
- 保留跨Wait的寄存器依赖，删除逐值memory clobber与常量零伪依赖；原数学、分块、
  任务依赖和sync_start不变。

版本：Simpler `a9edc8455`，PyPTO `d8e8e5935`，PTOAS `d53e3af3`，私有CSA `2cd33a9a`。
三仓补丁在本目录，基点分别为`9d6757269`、`9daa07428`、`4ef2f93c`。
[生成代码范围检查](scope_checks.json)保留完整early-resolve字段列表。

## 单卡128K/B16 CSA

任务`task_20261011_131501_9774708873`，0卡，CANN9.2、TMR、NZ2、atomic关闭、第二层真实权重。
范围为HC_pre+norm+CSA+HC_post；warmup后两版交错各30次，单次图重放，profiler关闭。
同轮同SDK的普通版是对照，不能与其他轮次的绝对时间直接相减作归因。

| 路径 | min μs | mean μs | max μs |
| --- | ---: | ---: | ---: |
| 普通版 | 959.04 | 984.39 | 1011.80 |
| 修正版全部局部边界 | 956.18 | 971.36 | 1000.06 |

均值减少13.03 μs（1.32%）；23/30配对更快，配对差值中位数−12.58 μs。
9项输出/缓存/状态经`torch.equal`精确一致，改变输入后的图重放也一致，两版各31项保护区通过。
这是一次同轮结果，不宣称已经证明长期稳定加速。[全部60次样本及功能结果](csa_results.json)。

## 保证没有后继的小图

任务`task_20261011_131301_90184725429`，0卡。图中只有一个48-block AIV task，
每核恰好一个block，无任何后续task。两版数学一致，仅候选增加局部Wait/Set。
比较1次与64次Vector循环，各两版交错30次；计时是整个小图设备时间，包含runtime开销。

| 循环数 | 路径 | min μs | mean μs | max μs |
| ---: | --- | ---: | ---: | ---: |
| 1 | 普通版 | 81.42 | 89.82 | 119.40 |
| 1 | 修正版 | 83.08 | 93.93 | 109.96 |
| 64 | 普通版 | 97.80 | 110.99 | 118.92 |
| 64 | 修正版 | 106.90 | 111.87 | 116.22 |

短图均值增加4.11 μs（4.57%），21/30配对较慢；较长图增加0.88 μs（0.80%），20/30配对较慢。
最大值虽下降，仍不能据此忽略平均额外开销或称为零成本；也不能将整图差值除以48当作每核成本。
这项检查尚未达标，按用户要求先停下讨论。[全部120次样本](no_successor_results.json)。

## 功能证据与限制

- CPU调度器9项、PyPTO定向codegen 6项、PTOAS正向2项/负向4项通过。
- 基础设备3项通过，包括动态分支、无后继、重复执行、AIV/AIC交接及保护区。
  两次AIV分别有48/48条观察到同核Scalar/流水重叠；AIC为15/24、24/24。
  这仅是基础用例，不能当作CSA交接比例。
- 新诊断的完成时间覆盖全部本核流水，不能直接当成Vector/Cube独立完成时刻。
  本轮没有CSA逐task的真实流水完成trace，未确认任何CSA task为Scalar-bound，也没有用它解释回退。
- 参考CANN9.2 A3同步头文件，没有发现逐Scalar值GM保存或逐值编译屏障；安装CCEC拒绝设备端
  `-S`/`-emit-llvm`，objdump不解码，因此未声称已获得ISA级零指令证明。
- 两个设备任务都退出0并释放卡。没有16卡测试、额外形状矩阵或公共环境切换。

## 复现与证据

私有根目录为工作区下`.worktrees/scalar-overlap-demo-20261009`。
归档脚本应放在该根目录的`artifacts/`后使用，脚本通过自身目录定位隔离环境；
先执行`build_runtime.sh`和`compile_cases.sh`，完成架构预检后用task-submit单卡执行
`run_local.sh`及`run_csa.sh`。PyPTO的`task_boundary_no_successor_demo.py`包含无后继图源码。

[汇总计算](analysis.json)、[调度器检查](scheduler_checks.txt)、[PTOAS检查](ptoas_checks.txt)、
[AIV原始trace](local_aiv_trace.json)、[AIC原始trace](local_aic_trace.json)。
