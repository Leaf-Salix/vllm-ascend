# 区间入口独立复核

本目录仅保存 CPU 策略与生成验证；未调用 NPU、未运行 lowering、未测性能。
检查复用了 `../range_source_audit_v1/binding_enum.json`，没有重新执行 192 次 binder。

## 策略及成本匹配

首次记录为 `policy_checks.json` / `policy_checks.log`：

- 192 个已记录源码绑定点，与区间入口的长度族、query 分组、投影族全部一致。
- 6 个长度端点 × B1～64，共 384 次分类检查通过。
- 七个回归点的调度、QK 索引缓存、行依赖及等待参数与 `selected_shape_policies.json` 一致。
- 9 个无效类型、batch、token 数、query S 和长度输入均拒绝。
- 七个成本文件准确命中；逐个修改全部 56 个 root 的形状，共 392 个变异均不命中。
  另外 14 个增删 root、7 个历史错配均不命中。没有把邻近档位样本称为当前 shape 实测。
- 80 个非默认手工实验策略均拒绝。

## 发现并修复的 CLI 漏洞

首次检查发现：仅比较 `args` 与 parser 默认值无法识别显式默认参数。
例如 `--range-policy --schedule-policy critical_path` 被接受后会静默改为
`earliest_ready`，显式 `--wait-backoff-ticks 0` 会静默改为 100。
原始 `policy_checks.json` 因此保留 `passed=false`，这是当时真实结果。

Root 已在 `range_policy.configure_device_arguments` 中增加 `argv` 参数，
通过 parser 识别显式 action；不传时使用 `sys.argv[1:]`。
修复后只复测相关反例，见 `cli_fix_recheck.json` / `.log`：
四个原反例、两种 `--option=value` 写法，以及实际默认 `sys.argv` 路径均正确拒绝；
普通区间入口仍通过。没有重复完整成本与分类检查。
`validate_policy.py` 随后适配了新 argv 接口供未来复测，历史首次结果未覆盖。

## 非七点 B12/H65536 完整生成

`prepare_b12.py` 调用真实 `build_range_variant`，约 15.49 秒完成：

- B12、S6、T72、历史 H65536、本轮长度 L65542。
- 策略为 long_cube / request_s6，earliest_ready；QK 索引缓存启用，行依赖细化关闭。
- 借用同构参考 h131072_b8，但实际 block、参数、视图及依赖重新生成。
- 没有完全相同 shape/历史的成本样本，`cost_profile=null`，未冒用邻近实测。
- 292 个 Tensor 视图有界，工作区互不重叠，等待表重建通过。
- 3 个种子 × 3 个代际的进度逻辑模拟全部通过；依赖偏序保持，新增全局屏障为 0。

摘要为 `b12_generation_result.json`，完整源码及参数在
`b12_h65536/variants/h65536_b12/`；调度绑定在对应 `bindings/`。

该输入是 CPU 描述符 fixture，保留 B16 长历史的物理页分配，调整批维与 compact 容量。
它不是已经构造并验证的真实设备 batch，也没有验证连续 decode、Scalar 缓存一致性、
硬件同步、精度或性能。整个区间的设备验证状态仍明确为 false。
