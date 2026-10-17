# 完整动态单 SPMD CSA：当前验证入口

本目录保存正式动态源码的整链测试、结果及失败尝试。当前入口在
`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/entry.py::decode_csa_spmd`，
由 `binding.py::PreparedDynamicCSA` 绑定 Native 输入和捕获前工作区。
源码保留 `pl.dynamic`、运行时维度和 device Q/L/position/slot/page table；
历史形状实值化版本只在 tests 中保留。禁止重新执行旧 `assemble_source.py`
覆盖后续手工维护的正式源码；该脚本及 `.txt` 模板仅是组装过程档案。

优化过程持续追加到 `vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`。
本文件只提供运行入口与结果索引，不能取代主记录。

## 已通过的范围

- 七档固定 S6：128K B4/8/16/24/28，8K B16/32；完整输出、Top-K/score、
  cache/state、保护区、图重放和信号代际检查通过。
- 128K/B16、8K/B32：同图同地址六次历史/位置/页表/紧凑metadata变化，
  以及 B→B-1→1→B 的整请求 padding 重放通过。Native metadata producer
  在图外执行；不将其图捕获、mixed-query、空rank或连续状态演进写成通过。
- `check_cpu.py` 无样例形状的符号编译通过；CPU模型与小型device probe
  只证明各自覆盖的合同，不能替代完整CSA场景。

## 性能口径与最新证据

只比较正式权重、同卡同进程、同轮完整 L1 incore 最早开始到最晚结束的中位数。
全部结果含内部等待与阶段空隙；目前仍启用测试编译hook的100tick轮询退避，
正式binding尚未默认启用该扩展。七档均未达到相对原动态分task降低30%的目标。

- [最新已选七档](HC_ROWS_PLAN_COLUMNS_SEVEN.md) / `hc_rows_plan_columns_seven.json`：
  默认HC行依赖、Attention有效列修正及输出组流水；这仍是最新完整七档测量。
- `dynamic_scenarios_v1.json`：上述两档的metadata与padding重放证据。
- `query_metadata_v1/result_summary.json`：复用Attention mask第5/6列中的owner/position，
  两档同进程单轴对照均功能通过；128K/B16关闭896.20、开启911.06μs，
  8K/B32关闭1036.34、开启1022.52μs。默认关闭，没有新建策略阈值。
  两侧原动态分task分别为873.20、978.82μs，均未达到30%目标。

## 复现入口

CPU符号编译：`bash reproduce_cpu.sh <全新输出目录> --attention-query-metadata on`。
依赖隔离PyPTO/Simpler worktree路径见脚本；不会更改共享安装。

设备任务必须通过任务队列分配卡，在本目录执行：

```bash
task-submit --device auto --max-time 1200 'bash run_device.sh /绝对路径/全新输出目录 --batch 16 --history 131072 --launches 1 --wait-backoff --attention-query-metadata on --attention-query-matrix --l1-windows 5 --plan-probe --metadata-replay --padding-replay'
```

`run_device.py` 先检查完整功能和精度，再捕获图及测性能；`output_policy_matrix.py`
只改变指定策略轴，以交替顺序测原动态分task与候选。`padding_replay.py` 不规定
dummy输出值，但检查dummy不得写cache/state及所有非允许写区。

大编译产物、原始完整trace与权重留在本地；提交清单和本地路径见上级阶段归档清单。
本轮不重复已经完成的七档测量，后续默认策略变化才触发相应回归。

用户最新约束：KV历史长度与batch为主要策略维度，本轮输入优先S6并补S5。
S5/S6允许分别特化，实际请求边界仍来自device metadata。本目录当前设备脚本
固定S6，以上数据不包含S5；下一阶段先参数化测试驱动并验证S5，再决定query
长度分档，不将CPU符号编译或helper测试当作完整S5通过。
