# 2026-10-22 八档六列 CSA 性能

按用户指定的 [规范快照](SPEC.md) 采集。结果只看 [八行表](PERFORMANCE.md)，
全部原始样本、版本、计时边界及来源保存在 [PERFORMANCE.json](PERFORMANCE.json)。
本轮是单卡单层测量，优化 Goal 保持暂停，不包含整模型/16 卡，也不修改生产算法。

## 场景与版本

[cases.json](cases.json) 固定七档 S6 和 T56 混长。128K/B4 和 8K/B16 已从本表移除，
256K/B16 使用真实 H=262144 的 cache、页表和 metadata。没有业务 padding。
权重为正式第 4 层 W8A8/NZ2，激活 BF16，输入 seed=1024；Indexer scale 使用既有
非均匀公式。所有实现沿用同样的历史页映射、请求顺序和状态起点。

[manifest.json](manifest.json) 固定各实现来源和公共 SDK。TND 单 SPMD 采用另一会话
已完成完整设备和服务检查的 combined_v1；不采用仅 CPU 通过的后续组合。
TND 多 task 使用61af469e中的正式实现。BSH单SPMD采用正式B1驻UB版本及已有的
150tick适配、短历史静态类型修正。后者已有公共SDK设备证据
`task_20261021_204807_386597714`，不属于本轮新增优化。

旧 Native 六档各有30个合格窗口，已复核原始CSV、静态编译、SuperKernel、NZ和多流
证据后复用；只补 Native 256K 和混长。旧 PTO 数据因输入、版本或每侧仅5窗不足以
直接填表，详见 [历史审查](history_audit.json) 和 [复用清单](inventory.json)。

## 采样与检查

- 每个 PTO L1 窗口仅重放一次，预热3次后交替测量每侧10次；覆盖完整任务首尾。
- TND 多 task 的 AICPU 列来自同一进程、同一图另采30次 Profiler；保留所有相关流的
  AICPU/AICore活动。DFX L1与Profiler分开采集，以免导出过程相互影响。
- 图外cache恢复、输出填毒和D2H不计入跨度；算法内metadata、同步和搬运保留。
- 检查eager、图重放及采样后输出、TopK/score、cache/state和保护区，沿用明确容差。
  Native真实路径设置`return_value=False`，不导出Indexer分数；其TopK及状态仍检查。
- Native与PTO已有浮点精度差异单列，不宣称跨实现bit一致或整模型token验收。
  表尾比例按用户要求比较Native外层与TND-SPMD L1，不是端到端吞吐提升。

## 采集过程中的排除项

首次Native启动漏带测试模块measure.py，在导入时失败；独立入口补齐后重跑成功，
失败任务保留。BSH B1首次测试绕过正式`csa_compile_constants`，误用
`SINGLE_M16_QA=False`，该MPMD格排除并按正式分支重测。SPMD侧源码和配置未变，
两轮的全部有效窗口合并保存，没有挑快值。

BSH短档首次编译暴露未激活长档临时Tile的3840→4096 reshape错误。已定位另一会话
原始通过快照，复用两处临时shape声明修正后补测8K/B32。所有长档score_tile=1024，
该修正前后的静态形状相同；保留原长档记录。原失败及来源都记入结果JSON。

旧runner的`source.mpmd`字段沿用CLI的candidate路径；BSH实际由family分支调用
冻结的BSH NativeCSACall。汇总证据明确标注这个旧字段的含义，真实路径见实际编译
目录、源码清单和任务日志，不把BSH数据标为TND。

## 使用方式

本地原始产物在本目录`runs/`，算子与测试源码冻结在`sources/`及两个BSH兼容目录。
它们不在有任务执行时改写；不复制SDK、权重或KV历史进入Git。
完整任务命令与终态见 [jobs.json](jobs.json)，大trace保留可访问的本地路径。

```bash
# 仅从现有原始记录重汇总，不上卡。
python tests/pypto_test/csa_performance_20261022/collect.py

# 必须换新的attempt，不覆盖既有结果；仅在确需补测时运行。
task-submit --device auto --max-time 1500 \
  'bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_performance_20261022/launch.sh tnd h131072_b16_s6 new_attempt'
```

准备脚本与补测脚本只服务本性能规范，不修改原优化目标的八档场景或分母。
