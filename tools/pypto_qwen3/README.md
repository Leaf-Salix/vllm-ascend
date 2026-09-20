# Qwen3 PyPTO 整网泳道采集

功能、性能和 ACLGraph 验证结果见 [TEST_REPORT.md](TEST_REPORT.md)。

当前实验分支可在不额外重跑请求的前提下，采集 Qwen3 PyPTO kernel 模式的真实 eager
model forward 或 ACLGraph replay。先保持原有 Qwen3 启动配置，再增加：

```bash
export VLLM_ASCEND_PYPTO_QWEN3_MODE=partial  # 或 attention_block / full
export VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_LEVEL=4
export VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_DIR="$PWD/qwen3_swimlane"
export VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_MAX_CAPTURES=1
```

| 开关 | 默认值 | 含义 |
| --- | --- | --- |
| `VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_LEVEL` | `0` | `0` 关闭；`1..4` 指定泳道采集级别，并同时采集依赖边 |
| `VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_DIR` | 空 | 开启时必填的产物根目录 |
| `VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_MAX_CAPTURES` | `1` | 每个 worker 最多采集的 live 窗口数，必须为正整数 |

强制 eager 时，每种 model-forward 入参形状的第一次真实调用只用于预热，后续真实调用才进入
`begin_dfx()` / `end_dfx()`；不会为采集额外执行一次 forward。图模式下 eager warmup 和
capture 都不消耗预算，采集窗口只包围实际 `NPUGraph.replay()`。v2 full-graph 的参数更新也在
窗口内完成，因为 `end_dfx()` 会 drain 当前流。

每个 worker 在输出根目录下建立独立的
`worker_<pid>_device<id>_<unique>/`。第一次采集直接写入该目录，后续采集使用
`window_1/`、`window_2/` 等子目录：

- `chip_swimlane_records.json`：设备原始泳道记录；
- `deps.json`：依赖图；
- `merged_swimlane.json`：可用 Perfetto 打开的合并结果；
- `capture.json`：eager/replay 边界、进程和采集级别等元数据。

若 replay 没有记录到 PyPTO AICore task，只保留原始记录和带
`"status": "no_pypto_tasks"` 的 `capture.json`，不会生成一个看似有效的空合并图。

采集会同步并 drain 当前流，且依赖生成本身有额外开销，因此采集窗口内的端到端延迟和吞吐
不能当作正常性能数据。本能力只同步泳道采集策略；复杂问题的 Qwen 输出精度与吐字检查需在
后续整网测试中单独验证。
