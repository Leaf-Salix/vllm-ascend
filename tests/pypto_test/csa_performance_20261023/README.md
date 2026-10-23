# CSA八档六列性能，2026-10-23

按[规范快照](SPEC.md)汇总，结果为[八行表](PERFORMANCE.md)。所有耗时单位为μs，
使用完整链路单次跨度的中位数。全量样本、逐格来源及环境见[PERFORMANCE.json](PERFORMANCE.json)，
只读核验收据见[VALIDATION.json](VALIDATION.json)。本次复用46个有效数据格、保留900个
原始样本，另有两个按规范不适用的BSH混长格；没有新增设备测试。

## 固定实现与复用依据

- Native：沿用现有CANN 9.2 beta2的npugraph_ex实际执行记录，保持dynamic=False、
  inplace pass、静态编译、SuperKernel、多stream、NZ2等原配置；逐格保留实际配置
  和生效证据。这是当前可获取版本的实测值，不能代表客户侧最优实现。
- BSH-MPMD与BSH-SPMD：沿用原规范已验证快照；含B1编译参数、150tick等待和
  共享SDK所需的短档临时Tile声明兼容修正，未替换为未验证源码。
- TND-MPMD：固定已采用v22、提交`d86c95f69`。复用完整八档的含AICPU三十窗；
  L1纳入同一冻结源码的全部已发现合格窗口，包括后续消融中的控制组。
  C05/C07/C08分别为50/60/40窗，其余各10窗。因此这三格可能与此前仅列完整
  采集那一轮的v22表略有差别，原因是样本集合补全，不是换了算法或挑快窗口。
- TND-SPMD：选定另一会话已完成设备验证的`output_rectangle_v3`冻结快照。
  C05有完整十窗、16种同址metadata加四步连续状态、service/custom-op与图重放
  门禁；另外七档已有完整生成代码及ABI等价证据，可复用其父版本合格记录。
  证据提交为`bb37e1cce`，但该候选尚未完成默认服务版本整理；本表不宣称默认
  入口已经切换，也不采用该会话未提交工作区作为测量源。

TND-SPMD复用链包括已保留的C04叶scale证明、C05/C06前段就绪顺序、C08混长
Score分组，以及C02实际256K容量支持。最终只选一个冻结版本；逐档不同测量来源
须有代码等价或对应开启域的实际设备证据，不能拼接各次最快候选。
相关版本、等价与门禁收据冻结在[evidence/](evidence/)；状态/CPU记录自身不作为
性能样本，性能只取原始完整L1窗口。没有重新hash源码或SDK目录。

PTO各列统一使用已验证的`csa-shared-20261022-logical-codegen`：
PyPTO `26e94e056`、PTOAS `914863ed3`、Simpler `57e499238`。
新增复用的SPMD任务实际bootstrap绑定和CANN日志摘录见
[TASK_LOG_EVIDENCE.json](TASK_LOG_EVIDENCE.json)。部分原始结果未单列框架版本，
已明确引用相同公共启动器及虚拟环境的规范基线记录，未伪称重新采集环境指纹。

## 口径与边界

固定八档均为单卡TP1、正式第4层W8A8/NZ2权重、BF16激活，输入seed=1024。
范围为`HC_pre → input RMSNorm → CSA → HC_post`，混长实际T56且无业务padding。
256K采用真实H262144、对应cache/页表和容量，未改为128K。

L1包含区间内的调度等待，不能解释成纯计算时间。含AICPU列为相关device流的
完整首尾跨度，不能解释成AICPU自身时长。两列来自同一v22实现；DFX L1与外层
Profiler原本分开采集，L1又纳入了同条件控制组，两列窗口不逐一配对，不宜相减
后声称差值就是调度开销。编译、warmup、测试复位及读回均不计入。

Native是TND路径；BSH与TND两种PTO入口在表内分别标识。PTO各版本的本轮比较
有输出、TopK/score、cache/state、图与保护区证据；不把原有Native/PTO的非bit
差异改标为一致。单层数据不能证明新的整模型token或DSpark接受率结论。

最后一列严格为`100 × (Native中位数 − TND-SPMD中位数) / TND-SPMD中位数`，
使用未四舍五入值计算；它比较Native外层device跨度与SPMD的L1跨度，不代表
同口径的端到端延迟降低率或服务吞吐增幅。

## 复算

```bash
python tests/pypto_test/csa_performance_20261023/collect.py
```

该脚本只读取既有JSON、L1与Profiler CSV，不初始化设备或执行新测试。
大型trace不重复复制，逐格保留可访问路径和task ID。旧表及原优化验收口径不覆盖。
汇总阶段仅按用户指令整理性能；优化Goal保持paused，不新增优化、不运行16卡。
推送按用户后续明确指令执行。

## 八档单次L1下载

按用户追加要求，v22八档L1已单独导出到：

`/data/pyptouser/qinchuanyu/pto-eager/downloads/CSA_TND_MPMD_v22_L1_8cases_20261023/`

同名`.zip`可一次下载。目录内只有八个泳道JSON和README，文件按01～08、历史长度、
B、S6/混长T56及版本命名。每档从本表全部合格窗口中选择最接近中位数的一次真实
执行，保留全部incore实例、实际物理核和任务间隙；没有拼接多窗或合成阶段时间。

Score/Attention标注AIC/AIV，O_A、量化、O_B标注输出组号。C05/C08使用配对任务的
v22控制组，C07使用v16/v22配对中的v22候选；已依据同次生成编排明确区分源码。
具体原始路径、所选单次耗时及版本判据见[download_manifest.json](download_manifest.json)。
用[export_l1_download.py](export_l1_download.py)导出，不占卡；既有目录不会被覆盖。
