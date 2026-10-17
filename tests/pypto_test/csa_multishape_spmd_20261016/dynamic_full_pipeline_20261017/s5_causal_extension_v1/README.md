# S5 的原分 task PTO 因果前缀精确参照

日期：2026-10-17～18。此结果接续 `../s5_native_validation_v1/README.md`，
用于核实尾组修复后剩余 Native 差异是否由动态融合引入。

原分 task 本身仍要求 S6。`../causal_extension.py` 让它实际执行 S6，
将每请求前五行输入、位置、物理页表及全部初始 allocation 与真实 S5 对齐；
额外第六行分别取 X/-X。先检查这两个 S6 结果的前五行和共同缓存精确一致，
再与真实 S5 单 SPMD 比较输出、Top-K、score 和共同 allocation 字节。

| 场景 | 任务 | 结果 |
| --- | --- | --- |
| 128K/B16/S5 | `task_20261017_235222_220641110995` | X/-X 两轮全部逐 bit 一致 |
| 8K/B32/S5 | `task_20261017_235222_22062647905` | X/-X 两轮全部逐 bit 一致 |

五份 allocation 的比较只排除 S6 独有合法写槽，全部 S5 合法写入均纳入；
两边保护区各自检查。没有修改 DECODE_SEQ、伪造 S6 metadata 或改变舍入门槛。
S5 自身 eager/graph、信号代际及保护区也继续通过。

在这两组输入上，S5 剩余 Native 数值差异同样由原 PTO 数学产生，未发现动态
融合新增误差。但 Native 冻结门槛及整模型 token 尚未通过，不能把该结论扩展
为所有混合请求、连续轨迹或 Native 精度完成。原始任务仍返回 1，原因是保留
了 Native 零容差报告；`causal_extension.passed=true` 单独表示此项精确参照。

原始报告见 `../native_h131072_b16_s5_v5/`、`../native_h8192_b32_s5_v5/`；
任务索引和精简结果在本目录。复现使用 `run_native_query_probe.sh`，添加
`--query-tokens 5 --causal-extension-reference`，保持原有正式权重与运行环境。
本轮未测 S5 性能。
