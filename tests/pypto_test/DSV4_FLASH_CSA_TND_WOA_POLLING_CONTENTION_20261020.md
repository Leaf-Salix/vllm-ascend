# TND精度SPMD后缀：WO-A轮询干扰与任务切点（2026-10-20）

## 本轮结论

上一轮只靠MAC/MTE2比例，不能区分搬运与同步。此次保持三臂AIC实际代码完全相同、AIV不读业务数据，
确认**等待协议并发执行会拖慢WO-A生产者，且强烈依赖形状**：

- T126连续回放，空闲/等8owner/等24owner为226.22/281.91/366.11 μs；后两臂分别慢24.62%/61.84%，各12/12窗同向。
- T126独立L1的AIC跨度为146.56/194.58/267.60 μs。不是只有消费者等待包络变长，生产者也变慢。
- T240连续回放只慢5.22%/10.27%；单次回放方向并不全部稳定，不能推广为所有形状同等受损。
- WO-A后切成两任务，T126后缀降8.09%，但T240回退3.08%；均仍慢于同任务冻结后缀，不通用保留。

本轮保留正式精度源和原基准，没有改SDK或生成C++，没有切换正式路由。
**尚未达到稳定快于冻结后缀的门槛，不进入8K/128K整模型token复验。**

## 身份与范围

分支`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`，报告起点ae9d9cc47；
正式精度运行源48c58ce7c、上轮所有候选不变。
前缀取自phase1 f22d7d8d…；完整后缀直接父版owner2 399dbc67…；新增候选的完整SHA及父链在proof中。
参考nalinaly固定beaefa6f6788a9448d49293238f6818c246f24f5，本轮从本地Git对象读取，不切换其它checkout。

真实第4层WO-A/WO-B权重，合成heads/residual/gates，FULL graph；T为token行数，不等于请求batch。
纯等待诊断金值是同输入完整phase1后缀的有效FP32 accum；完整后缀金值为冻结PTO五边界。
**本轮不是新增Native整层或真实整模型验收**，也不新增非等长TND metadata资格。

| 项目 | 身份 |
| --- | --- |
| PyPTO | 2dfa7f120ba5e577566046d920845aa241bfa7bb |
| Simpler | 57e499238fd4ee3548947ad860be456aaeeb16b5 |
| PTO-ISA | 327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d |
| 实际PTOAS | 隔离wheel0.66，SHA256 d121b5b5…，Git源提交未证明 |
| Python / Torch / Torch-NPU | 3.12.14 / 2.10.0+cpu / 2.10.0.post2 |
| vLLM / CANN | 0.25.1 / 9.2.0-beta.2 |
| 确定性 | level1，HCCL_DETERMINISTIC=true |

调度参数仍为`--ptoas 0.63`，不代表实际编译器；脚本明确PTOAS_ROOT指向隔离0.66。
导入路径、DSO及精确源SHA见每个result的sdk_preflight，不因旧文档profile而推断当前环境。

## 三种纯等待控制

| 控制 | AIC | AIV |
| --- | --- | --- |
| idle | 原WO-A、原publish1 | begin代际及共同cache/fence，无业务读取、不等待 |
| wait8 | 与idle完全相同 | 仅等待实际8个owner，随后共同cache/fence |
| wait24aligned | 与idle完全相同 | 前8个地址/顺序与wait8相同，继续扩展到全部24owner |

三臂相同24 AIC/48 AIV、sync_start、16 Tensor ABI与独立workspace。
AIC源AST及实际PTO/C++去loc/标识符后逐项同构，数值常量、物理地址、布局和控制不忽略。
AIV均不cast、不load/store业务tensor；未用五输出及integer工作区始终保持哨兵。
避免把并发cast的搬运量混进轮询干扰。
共同cache维护数量相同，但执行时刻随等待不同；所以因果结论是**等待协议并发的整体影响**，
不能进一步单独归到TWAIT某条指令、DDR字节或DCache失效。

## 门禁与计时

先CPU完整PTOAS/CCEC、artifact.load、实际注册表/角色和依赖审查，再通过task-submit占卡。
前缀检查有效FP32 accum逐位、eager/FULL A/B/A毒化、padding、输入/工作区/signal guard、只读SHA、独立地址及代际。
A/B改变真实heads并重算完整父版gold，确认输出随输入改变，恢复A后逐位复核。

无插桩任务每臂分别测1次与100次回放，两种批量各12窗，三臂位置轮换/反转；每个窗口后逐位复核，不删离群值。
中位数为窗口平均耗时的中位数，不是单次p50。单次事件测量也含运行期入口，不是纯核内计时。
每项前缀84个逐位指标、final_generation三臂1216；DFX/PMU另任务，各21指标、三臂7。
完整后缀仍五边界160项、双generation1204，首差即停。

内层750秒/调度780秒，均小于15分钟；观察间隔不超过5分钟，不开自动化。
一次本地下载自动审批超时，按工具要求唯一重试后成功；没有因此重交或遗留NPU任务。

## 无插桩前缀对拍

单位μs/WO-A prefix FULL replay。每行三臂同一任务同一卡同输入。

| T | 每窗回放 | idle | wait8 | wait24aligned | wait8较idle | wait24较idle | 慢于idle窗数8/24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 126 | 1 | 231.95 | 296.46 | 375.36 | +27.81% | +61.83% | 12/12、12/12 |
| 126 | 100 | 226.22 | 281.91 | 366.11 | +24.62% | +61.84% | 12/12、12/12 |
| 240 | 1 | 230.90 | 236.29 | 242.58 | +2.33% | +5.06% | 6/12、7/12 |
| 240 | 100 | 203.58 | 214.20 | 224.48 | +5.22% | +10.27% | 12/12、12/12 |

T126/T240走不同K轨迹/流水路径；不能用两行绝对数宣称更大T计算更少，不能把前缀差值直接当完整后缀收益。

## 独立L1：生产者也受影响

每臂3次单回放中位数，单位μs。三臂各72条记录，24AIC/48AIV、50MHz、1个root marker、无dropped。
18个原始窗口已离线重算跨度；不把run_boundaries当成起止对，不与正式事件计时相减。

| T | 角色 | idle | wait8 | wait24aligned |
| --- | --- | ---: | ---: | ---: |
| 126 | AIC跨度 | 146.56 | 194.58 | 267.60 |
| 126 | AIV跨度 | 9.60 | 197.94 | 272.42 |
| 240 | AIC跨度 | 114.96 | 122.86 | 130.72 |
| 240 | AIV跨度 | 9.68 | 126.72 | 134.40 |

同一AIC代码在AIV等待时结束更晚，支持并发等待干扰生产者，而非仅消费者被动挂起。
角色跨度互相重叠，不可加总；没有原位phase时间戳，不能将差值直接称为纯等待时长。

## 独立PMU：计算不变，搬运路径受到影响

T126同任务、每臂三次采样中位数；9份设备CSV均存在，各取同名MIX kernel，排除AICPU。

| 指标 | idle | wait8 | wait24aligned |
| --- | ---: | ---: | ---: |
| kernel Duration μs | 241.56 | 292.30 | 388.64 |
| aicore_time μs | 237.46 | 289.36 | 383.52 |
| AIC MAC busy μs | 27.949 | 27.948 | 27.949 |
| AIC MTE1 busy μs | 26.677 | 26.012 | 25.873 |
| AIC MTE2 busy μs | 95.826 | 139.932 | 165.554 |
| AIV Scalar比例 | 73.1% | 87.4% | 89.8% |

实际WO-A数学及加载相同，MAC忙时也近乎相同，MTE2忙时却随并发等待增加，支持搬运路径被拖长的解释。
没有DDR事务/字节计数，不能说多搬了数据；MTE2 busy也不是纯等待，不能与MAC/MTE1求和或相减为stall。
PMU是角色汇总，L1是跨度，来自独立任务，不计算两者贡献比例。
idle的AIV Scalar仍有73.1%，进一步说明不能把Scalar比例全称为信号等待。
profiler的host-flow/feature告警保留，未伪造kernel关联；本轮PMU正常exit0。

## 与nalinaly的等待及流水差异

| 方面 | 固定nalinaly参考 | 当前实验 | 含义 |
| --- | --- | --- | --- |
| 未就绪轮询 | wait_codegen可显式插入100 syscnt tick退避；所读代码生成audit为wait_backoff=true、device_validated=false | 当前SDK默认TWAIT紧轮询 | 多消费者轮询策略不同，是值得实测的独立因素；不能单凭源码解释全部速度差 |
| 启用条件 | 编译回调显式替换并核数量，仅导入模块不生效 | 未增加C++源码变换或SDK扩展 | 保留纯DSL约束；本轮改任务依赖减少并发轮询 |
| WO-A计算 | 参考递增K256、按组流水 | Native校准cyclic K及必要K16边界 | 不能直接复制K轨迹而声称Native精度不变 |
| WO-A流水 | 按源与实际缓冲判断 | 原L1有双缓冲，部分L0槽被内部K128占用 | 前轮K128有局部收益，不能认定整个prefetch失效 |
| 量化语义 | 参考1024分组scale | 全8192共同scale | 精度链不同，不迁入分组量化来追速度 |

wait_codegen源码、编译回调及选定reference audit从本地Git对象按完整commit归档。
该audit是代码生成记录，device_validated=false，不是本轮nalinaly设备启用或收益的证明；
未声称该audit代表用户图片中每个测点，也没有进行本轮同输入nalinaly全层速度对拍。
其100 tick不是微秒，不引入手写C++等待helper，也不自动授权修改SDK后端。

## 单因素优化一：WO-A后增加任务切点

一个PTO顶层保留两段：MIX1原WO-A、AIV无业务；MIX2原cast/8192量化、WO-B及HC-post。
MIX2读取已完成accum、signals，删除WO-A信号轮询但保留原payload acquire。
current_generation只读本core阶段0，不递增、不notify；整个调用只prefix begin一次。

实际编排是两条独立24block/sync_start MIX submit，共享原accum/signals等InOut；
Simpler默认allow_early_resolve=false，存在真实完成依赖，不以scope退出或cluster名字当同步。
没有增加workspace或根ABI，数学节点及owner2等待逐AST保留，实际prefix AIC与纯等待idle同构。

| T | 冻结PTO μs | owner2父版 μs | 两段候选 μs | 对父版 | 对冻结 | 改善窗 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 126 | 403.62 | 496.48 | 456.31 | -8.09% | +13.05% | 12/12 |
| 240 | 469.76 | 502.69 | 518.17 | +3.08% | +10.31% | 0/12 |

两档五边界逐位均通过。T126有局部收益但未胜冻结；T240回退，因此不作为通用策略。
新增任务调度/同步与并发重叠变化都包含在结果中，不能用前缀节省量推算净收益。
这也说明当前后缀差距不能全部归到WO-A轮询：消除它后仍有差距，且小干扰形状可能得不偿失。

## 单因素优化二：量化后再切一次

父版明确为上节两段候选；新增一处cast/quant→WO-B切点。
实际目标为一个顶层、24MIX→48AIV→24MIX；中段只执行原cast/量化，WO-B开始前依赖其完成，
保留payload acquire，后段原WO-B/dequant/HC-post不改。逻辑worker仍0..47、phase槽24..71，代际只推进一次。

初稿空闲Cube函数因没有Cube算术被推断成AIV，同cluster两个AIV时实际注册只包含首个空闲入口。
虽然CPU编译通过，实际kernel_config未注册业务量化，**设备前已拒绝，没有上卡**。
修正版显式单AIV函数、48个flat block，不使用split subblock映射；独立核当前runtime的AIV-only资源支持全48分派。
实际kernel_config ID2为vector_quant、wrapper仅flat block_idx；三段原InOut强依赖和有限T1..256工作覆盖已核。

| T | 冻结PTO μs | 两段父版 μs | 三段候选 μs | 对父版 | 对冻结 | 改善窗 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 126 | 401.84 | 433.87 | 432.21 | -0.38% | +7.56% | 9/12 |

五边界160项逐位、A/B/A、guard/只读和代际1204通过。净增量只有0.38%，不认定为可靠收益，不扩大T240、不推广。
本行父版与前表父版不同；不能用跨任务绝对值漂移宣称组合减少了多少延迟，也没有将两次切点百分比相加。

本轮8项NPU任务全部exit0并已结束；共711项逐位指标通过，其中纯等待/仪器231项、完整后缀480项。
未发生数值首差；CPU/编排被拦截的历史保留，不把CPU成功冒充设备成功。

## 推广与后续门禁

保留原冻结精度基准、上轮候选及本轮所有实验文件，不将相对较弱父版的收益当作交付收益。
只有稳定快于冻结后缀、再通过完整CSA精度与性能门禁，才先8K/B40再128K/B24做192-token Native一致性复验。
当前没有新增整模型token资格，没有修改正式算子、环境或Native路径。

## 证据与复现

[完整原始结果及复现](evidence_tnd_precision/woa_contention_20261020/README.md)保留源SHA/父链、运行命令、所有task状态、原始L1/PMU、CPU失败及生成码、独立审查。
前轮基线及流水实验见[WO-A流水与owner报告](DSV4_FLASH_CSA_TND_WOA_PIPELINE_OWNER_AB_20261020.md)。
