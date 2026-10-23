# 优化前验收证据

本目录固定2026-10-22至23日共同三方实测和阶段定位的机器结果。
算子数学基准为6eabea0ec；本目录不启用新候选。

- `stage_metrics.json`：同输入模块对拍；含Native eager/FULL控制、不同元素数及relative L2。
- `stage_cpu_recheck.json`：本机从原始张量重新核验的19个边界。
- `threeway_recheck.json`：三方单层完整图和独立L1原始窗口重算。
- `whole_summary.json`：真实整模型范围和token结论；nalinaly完整64明确未完成。
- `executed_stage_manifest.json`：实际设备执行的三个Host观测脚本SHA。
- `stage_manifest.json`：仓库格式化后三个Host脚本SHA，用于自校验。
- `tools/`：v3观测脚本的仓库格式版本，仅规范引号、换行及导入顺序、删除未使用导入；计算和控制流AST不变。

## 原始观测脚本的复现条件

脚本保留受测时的路径和身份门禁，依赖原共同实验的`capture-v2`资格目录、
固定source manifest、SDK收据、13份官方静态包、模型权重和独立fixture。
这些资产不包含在本Git目录，迁移机器必须先准备相同资产，再显式改路径、重新封存并完成CPU资格；
不能直接删除哈希/静态资格断言来启动。

已有原资产时，将tools及`stage_manifest.json`复制到一个新的自有实验目录，
将manifest命名为`manifest.json`。任务入口为：

```bash
task-submit --device auto --device-num 1 --max-time 780 \
  --env PYTHONNOUSERSITE=1 "bash <实验目录>/run.sh st-stage8k-v4"
```

使用新label防止覆盖。入口内限750秒，只有全部三方FULL/raw/ABA门禁通过后才做观测。
Native额外eager必须将owner cache绑定回自身fixture，结束后恢复；
否则会误读最后一个fixture的cache。该观测问题的失败材料被保留，未计作算子误差。

结果中`stage_diagnostic_only=true`和`timing_executed=false`，本任务不授予新性能收益。
`stages.json`的`complete=true`只证明该单层诊断完成，不代表整模型验收。
TopK逐位一致仅限最终索引；HC-post端点重构及attention组合的范围见主报告。

原始96MiB阶段张量SHA为
`a9154905d777098c90c7b66d174a4a7650773ba791ed7fd8d0830f17a5d7ac5a`。
机器结果和独立CPU重算通过，原始张量保留在实验归档，不写入Git。
