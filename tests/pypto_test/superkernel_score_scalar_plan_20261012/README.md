# 过程071：复用 Score 最大长度，单独评估 Scalar 规划前移

2026-10-12。保留私有 `score_scalar` 算子候选：复用调度入口已有的最大长度，去掉各 Score 成员重复的长度扫描。
`score_early` 进一步把规划移到局部 Wait 前，两轮都慢于 `score_scalar`，不采用。
扫描消除是独立算子优化，不能归功于 early-start。公共 CSA 默认没有修改。

## 修改及安全依据

私有 vLLM 驱动 `ac56df93`，PyPTO `9e2865861`、Simpler `6174ec5db`、PTOAS `3b352f70`、
PTO-ISA `327cd5869` 不变。继续使用片上填充通知和过程070的 Wait 同步修复。
局部 `wait_pre_task_end()` 手工标识及 chip swimlane/PMU/参数 dump 回退按用户要求保留，
不修改派发、ACK、核选择、`allow_early_resolve` 或 MIX 组完成语义。

- 入口 `_read_device_max_seq_len()` 原本就扫描 `kv_seq_lens`，用于长短策略选择。
  `indexer_score_topk_forest()` 已有 `max_topk_cache_len = host_max_seq_len // COMPRESS_RATIO`。
  将此值传入四个互斥的 Native-Cube Score 策略调用，不增加入口扫描。
- 原核内扫描取 `max(0, floor(length / R))`；这里 `R=4`，扫描同一组固定六 query 的请求时，
  等价于 `max(0, floor(max(length) / R))`。当前设备证据为128K/B16，每请求verify6。
  不据此宣称任意变长打包、其他 batch 或其他实现均已验证。
- 降低后的 IR 显示长度是根输入，所有相关调用均只读；运行 fixture 在图捕获前检查真实指针范围，
  仅允许与同样只读的 `cmp_seq_lens` 重叠。通用 ABI 允许完整区间别名，不能只凭 `In` 方向证明全局不可写。
- `score_scalar` 保留原 Wait 位置；`score_early` 只将 Scalar 叶片规划放到 Wait 前。
  后者不含 GM 读取，FFTS/FIFO/流水工作仍在 Wait 后，没有放宽编译器对任意 GM 前缀读取的拒绝。
- 三条 early-start 路径各73个生成成员检查通过，未发现 Wait 前流水初始化；
  实际长档 Score 的 AIC/AIV 跨 Wait 保留值从1个变成4个。这是生成代码计数，不是节省的周期数。

## 单卡完整 CSA

初测 `task_20261012_015901_1080531630`，确认 `task_20261012_020205_2339611852`；均卡0、退出0并释放设备。
第2层权重、合成128K/B16历史、HC_pre+norm+CSA+HC_post，CANN9.2、TMR、NZ2、atomic0、deterministic0、无诊断。
每轮预热5次，四条路径各30次轮换单次 NPUGraph replay，外部 NPU Event 计时，reset 在区间外。
普通 PTO 复用过程068同 SDK 产物；`current` 复用过程070修复后的原边界产物。

| 轮次/路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 初测普通 PTO | 947.06 | 987.97 | 1186.14 |
| 初测 current | 967.80 | 990.81 | 1010.88 |
| 初测 score_scalar | 961.28 | 979.34 | 998.82 |
| 初测 score_early | 966.80 | 991.42 | 1018.84 |
| 确认普通 PTO | 965.08 | 982.52 | 996.56 |
| 确认 current | 974.00 | 988.37 | 1006.04 |
| 确认 score_scalar | 952.06 | 981.58 | 1014.10 |
| 确认 score_early | 967.32 | 984.32 | 1003.36 |

`score_scalar` 对 current 的两轮 mean 分别减少11.47/6.78μs，配对中位差为−11.70/−9.57μs，
24/30和20/30配对更快；保留为后续私有算子对照基础。max 分别减少12.06、增加8.06μs，不能宣称长尾改善。
`score_early` 比 score_scalar 的 mean 多12.08/2.74μs，配对中位差+14.47/+7.02μs，
仅6/30和13/30配对更快，因此本轮不扩大此处前缀。

普通 PTO 尚未应用扫描消除；它与 score_scalar 的差同时包含算子和交接两项变化。
确认轮仅少0.94μs、15/30配对更快，不能据此宣布 early-start 已经获得稳定净收益。
初测普通1186.14μs长尾完整保留，不将 max 差归因于单个改动。
两轮四路各31项保护区检查通过；三条 early-start 路径的9项输出/cache/状态与普通 PTO 逐 bit 一致，
改变 hidden 输入后的图重放也逐 bit 一致。没有新增 Native、七档、16卡或整模型 token 验收。

## 规划前移为何没加速：已知和未知

额外做了6次 CPU 编译检查，使用正常编译所解析的固定 PTO-ISA、相同 CANN/CCEC flags，并加 `-fstack-usage`。
current/score_scalar/score_early 的 AIC 静态栈都为240B，AIV都为320B，尾回调都为8B。
score_scalar→score_early 的链接代码 AIC 9992→9888B、AIV 11296→11228B。
这些结果不支持“静态栈增大导致回退”的说法，也不能证明寄存器调度或流水时序没有变化。
未新增扰动设备时序的插桩，故仍不能定性为 Scalar-bound 或确认具体回退原因；不继续小参数搜索。

## 文件和复现

- `timing/`、`confirmation/`：完整结果、两轮共240条样本、运行日志及配对统计。
- `source/`、`vllm-ascend-incremental.patch`：私有驱动；零上下文补丁使用 `git apply --unidiff-zero`。
- `compiler/*_operator.diff`：两个冻结算子包相对 mixed 包的完整差异。
- `compiler/generated_prefix_audit.json`、`generated/`：同步边界、入口最大长度及 Score 前缀。
- `compiler/readonly_metadata_inspection.txt`：降级后的只读参数流；物理别名检查见运行日志。
- `compiler/stack_usage.json`、`compiler/stack_usage/`：6个对象的编译命令、静态栈及链接代码大小。
- `repro/`：编译、两轮排队测试、CPU 静态栈检查脚本。脚本按 `artifacts/..` 找到私有根目录，
  需放回原私有环境的 `artifacts/`，依赖已有冻结包、先前产物和权重，不是新 clone 可独立运行的发布包。

设备命令（当前无需再跑）：

```bash
task-submit --timeout 900 --max-time 900 --device auto --device-num 1 --run 'bash /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/artifacts/run_score_scalar_plan_20261012.sh'
```

当前仍有一般只读 GM 前缀证明和通知事件资源覆盖的编译限制。手工 Wait 和诊断回退是确认保留的范围，
不能当作下一轮待移除项；当前优化也没有自动识别 Scalar 前缀。
