# 全阶段独立编译与现选等待策略的组合诊断

2026-10-17。复用`stage_codegen.py`和`signal_polling.py`，没有修改依赖或生产入口。
外部仍为一个SPMD任务；这里的74个函数是同任务内部device函数，不是新增74个运行时任务。

## 为什么增加这一对照

早期全部阶段独立编译/noinline的1044.54μs，发生在默认等待时期。
当前约807μs的版本已采用cached100、QK索引缓存、行依赖，不能用旧轮次直接判断该组合性能。
此次只评估现有数学和同步语义下的编译边界影响，没有叠加本轮未获收益的批量RoPE。

## CPU证据与执行前门禁

`compile_outlined_cached_wait.py`从现选`b16_cached_row_deps_v1`的源码/布局开始，
在全部编译扩展生效后解析JIT artifact。CPU结果在`outlined_cached_wait_cpu_v1/`：

- 74个阶段完整独立编译，全部noinline，内部只传必要的参数和外层计算的同一代际。
- 仅在所有阶段已分离时清理外层原数学事件，且原set/wait列表必须严格配对。
  helper自身的流水和FFTS事件保持，业务数据维护保持。
- 实际交给CCEC的AIC/AIV源码各含74处替换等待；DCCI后volatile读、100tick退避正确，
  无原TWAIT残留。外层无遗留流水事件，各阶段正好调用一次。
- 审计最初没识别后端正规化后的`static __aicore__`拼写；修正识别规则后通过，
  没有修改候选C++或放宽覆盖条件。

`outlined_cached_audit.py`在真正设备执行前再次检查上述最终源码。
driver只开放已核对的B16/128K、全阶段、compact参数、noinline、QK Vector索引、
行依赖、earliest_ready、cached100组合；部分阶段或缺少noinline仍拒绝。
CPU参数门禁记录在`outlined_cached_argument_guard_audit.json`。

## 任务记录

- `task_20261017_131724_17808923318` / `b16_row_deps_outlined_cached_v1`：
  被另一处旧行依赖组合保护拒绝，exit1，尚未执行算子；不记功能或性能结果。
- 补齐同一窄范围组合的保护后，提交`task_20261017_131834_1981138300` /
  `b16_row_deps_outlined_cached_v2`。先完整exact、六状态、图、30代与保护区，再采三窗L1。
  截至本记录等待结果。

这是C++函数边界归因实验，`eligible_for_700us_target=false`。
即使更快，也不能直接宣布正式PyPTO接口接入或700μs目标完成。


## 设备结果：完整PASS，未采用

task_20261017_131834_1981138300 exit0。现选数学/行依赖与全部阶段独立noinline+cached100的组合完整通过：eager/graph/final输出、TopK、六状态零差异，30代通知与所有保护区PASS，实际74helper/事件闭合/100tick门禁PASS。完整L1 807.02/805.34/810.66，中位807.02；同轮动态886.36/906.28/902.84，中位902.84。与现选约807持平，不采用、不扩档、不再重复该组合。该归因诊断eligible_for_700us_target=false。不能把静态屏障数减少直接当收益，也不能据此断言同步完全没有成本：函数调用/独立规划可能抵消其变化。

详见`outlined_cached_result_v2.json`及原结果目录。40秒观察等待曾超时，但任务正常继续并exit0；未取消、未把观察超时当kernel故障。
