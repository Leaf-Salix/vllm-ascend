# dynamic固定关闭、inplace开启的七档对照

任务：`task_20260929_095651_194851727802`，2026-09-29 09:56正常auto提交，同卡完成整个矩阵。
Native七档正式计时与主性能profile已完成，见[NATIVE_RESULTS.md](NATIVE_RESULTS.md)；
完整矩阵现已completed(exit=0)，七档计时、独立核内诊断及28个DFX窗口齐全。
[完整比较](RESULTS.md)、[逐任务核内与pipeline](TASKS.md)、[28份原始JSON下载](download/README.md)。

用户进一步明确固定dynamic=False并开启inplace_pass。旧任务
`task_20260929_093737_25396355997`的inplace为False，已主动停止并确认exit130；
旧轮已完成档位只保留为历史，不与本轮拼接。不是旧轮性能或编译失败。

本轮复制完整已冻结源码及PTO包，算子仍为55b89ee2，性能版已融合Sparse末块发布。
只改私有测试入口：

- Native显式torch.compile(backend="npugraph_ex", dynamic=False, fullgraph=True)，
  inplace_pass=True、static_kernel_compile=True；整体性能SuperKernel开启。
- Native核内诊断保持上述配置，仅关闭SuperKernel；一次重放状态检查加独立profile，
  不产生可混入正式表的性能均值。
- PTO保留生产自定义算子边界，私有vLLM编译包装也开启inplace_pass，记录实际送往后端的选项。
  不改变共享compiler_interface、Native生产流程或算子算术。

Native复用已校准的后端唯一图直接replay，要求无主机更新节点；不会嵌套手工外图。
每档5预热20次无profiler设备事件，保留全部样本/P95/max；两侧独立PyTorch profile，PTO四窗DFX。
静态编译、SuperKernel调用、同图八类状态及保护区通过后才接受结果；不是跨Native/PTO精度验收。

档位128K×B4/B8/B16/B24＋8K×B16/B24/B32，B40不测；长短8:2，各上下文内batch等权。
CANN9.2/mode2/det0、PTO atomic0、EPLB关；第二个CSA（layer4）真实权重及独立合成历史。
范围为HC_pre+norm+CSA+HC_post，不含MoE，不是EP16或整模型forward结果。

冻结包两入口依赖图、生成runner的Python语法/Ruff和shell语法已通过。
收集器同时检查冻结torch.compile参数和报告中的实际inplace配置，
Native主性能完整SuperKernel范围与关SuperKernel的QLI/Sparse核时分别记录。
另存Native实际输入shape、block数和MAC/MTE/FIX/Vector等pipeline计数，
并生成七档TASKS明细；各pipeline可重叠，不能求和或拿不同worker工作量的核时直接相减。
PTO四窗继续使用官方原始join/行数/block检查；短B16/B32系数worker为48，其余为batch。

任务成功结束后依次运行`collect.py`和`bundle.py`。下载包保留28个原始JSON：
每档Native主性能、PTO主性能、固定window_3泳道及Native核内诊断，不修改或拼接事件。

[准备入口](prepare.py)、[冻结来源](source.json)、[解析](parse.json)、[矩阵](run.sh)、
[分侧入口](run_side.sh)、[收集器](collect.py)、[汇集器](bundle.py)、[任务](task.txt)。

按用户后续补充，Native单独收集器`collect_native.py`在最后Native子进程成功返回后立即发布，
不等待PTO采集。以后PTO使用现有已验证路径和数据，不为匹配Native编译选项重测。
