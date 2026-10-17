# O_A→O_B 量化：复用第一次 FP32 加载

本候选完整 symbolic CPU 编译与生成审计均通过。只删除同一 qt 的第二次
FP32 GM load，原值在 abs/row_max 期间保持完整，量化数学和动态有效行不变。
**两档设备完整检查通过，但完整 L1 无收益；候选不采用。最终数据见文末及 DEVICE_RESULTS.md。**

## 源码与范围

基线为正式 `d253d627dde3646aa916879a481d3e57322f5c33`。
`source_before/` 冻结当时全部 33 个正式 Python 文件，`source_candidate/`
为独立候选。冻结时间 UTC `2026-10-17T19:15:04.385180+00:00`，本地约 03:15。
除 `output_projection_a_stages.py` 外，其他 32 文件逐字不变。

唯一执行逻辑变化位于 `o_projection_groups_quant_stage`：

```python
# 原来再次读取同一 o_r_pad / qt / col_g / rows。
oc_q = oc_amax
```

它复用第一次 `pl.load` 的带符号原值，**不是复用 abs 后的值**。
保留原 abs→row_max→floor→127/amax、amax*(1/127)、row_expand_mul、
INT32 rint→FP16 round→INT8 trunc 全部顺序；不改 tile、owner、两级流水、
尾零、scale 未写区、notify/wait、ABI 或依赖。
`source_audit.json` 将唯一赋值归一化后，整个模块 AST 与原文件完全相同。
`candidate.patch` 保存改动，`make_candidate.py` 可在全新目录复现并拒绝覆盖。

历史排重及选择依据在相邻目录 `../dense_projection_d253_audit/`。
原分 task 也有此重复加载；本轮不把它说成动态融合独有错误，不重复 O_A
双输出/K512/N256/HCstream 或主 Q_B Left 槽的旧实验。

## 唯一一次完整 CPU

`cpu_v1/result.json`：完整代码生成 PASS，58 个动态轴、single MIX，未提供
实际 shape，未执行设备。区间控制为 G6、QK64、score1024、OA128、OB96；
attention query metadata 关闭，其他控制沿用该入口默认值。
`controls.json` 仅为 `{"OA_COLS":128,"OB_ROWS":96}`。

这是长 B16 使用的策略，但编译仍是动态 T/B 和设备 metadata，不冻结成 B16。
完整 CPU 仅执行一次，没有为审计补编或扩大 tile 扫描。

## 实际生成审计

基线生成物采用与冻结源 **33 文件全部一致**的既有真实 B16 产物：

`parallel_20261018/hc_active_notifications_h131072_b16_v1/build_output/decode_csa_spmd_3kuwaqat`

审计程序 `audit_generated.py` 只读上述产物和本次 `cpu_v1/kernel`：

- 原量化 PTO 有 6 处 TLOAD，候选只有 3 处：两级展开的两个 qt 各 1 次，
  再加尾迭代 1 次。partition_view 从 13 变 10。
- abs、row_max、div、乘法、三段转换、7 处 store、31 处 validshape 的静态
  操作计数全部一致。没增加复制算子来替代第二次 load。
- 三处原值都在 abs 与 row_max 期间保留独立物理区间；量化乘法仍读取这些
  原值的同址别名。后续缩放可原地覆盖原值，因为原值此后已经完成最后使用。

| 分支 | 原值地址 | abs 输出 | row_max 临时 |
| --- | ---: | ---: | ---: |
| 第一 qt | 65536 | 32768 | 0 |
| 第二 qt | 98304 | 0 | 65536 |
| 尾 qt | 65536 | 0 | 98304 |

每个大块均为 `[8,1024] FP32`，32 KiB。第二 qt 临时区复用第一 qt 已结束的
槽，不覆盖第二 qt 的原值；仍由原自动流水依赖管理复用顺序。
`generated_audit.json` 保存全部区间、原 PTO 行号和非重叠断言。

真实完整分配的地址末端上界：

| 空间 | 原版字节 | 候选字节 |
| --- | ---: | ---: |
| AIV Vec/UB | 163872 | **155872** |
| AIC Mat | 475136 | 475136 |
| AIC Left | 65536 | 65536 |
| AIC Right | 65536 | 65536 |
| AIC Acc | 131072 | 131072 |
| AIC Vec | 16384 | 16384 |

AIV 总上界下降 **8000 B**，不是预估的 32 KiB；其他阶段仍贡献全函数上界。
PTO Tile 几何计算与 `38_after_AllocateMemoryAddr.py` 的真实 MemRef 容量复核
一致。不把只读规划的 128 KiB 活跃量当作本次完整分配值。
全局 UB 地址可能重新排列，不宣称所有其他 helper 地址不变。

生成 C++ `kernels/aiv/vector_schedule.cpp` 的首个展开片段也已保留：
`candidate_quant_first.cpp.txt`（60022–60163 行）。其中只有当前/下一块的
两次 TLOAD（60048/60075），TABS 60099、TROWMAX 60106、TROWEXPANDMUL
60163；没有额外 TMOV 复制。地址别名的数值证据见对应 PTO 摘录。

## 复现及设备验证边界

```bash
E=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-compact-priority-20261018/tests/pypto_test/ultra_oa_quant_reuse_20261018
D=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017

# 只读已有产物，可重复执行，不编译。
python "$E/audit_generated.py"

# 已执行的 CPU 命令；若复现应换全新输出目录，勿覆盖证据。
bash "$D/reproduce_cpu.sh" "$E/cpu_v1" \
  --attention-query-metadata off \
  --spmd-source "$E/source_candidate" \
  --controls-json "$E/controls.json"
```

根代理已提交以下完整同进程三方矩阵，本子任务没有重复提交；提交后源保持
冻结，结果由根代理收取：

- 128K B16：`task_20261018_031725_241975130621`。
- 8K B32：`task_20261018_031725_241981318600`。

参考源为本目录 `source_before/`。CPU 审计不代替两档设备精度、图重放、
信号代际、保护区和完整 L1 性能结果。正式源码与依赖未改，没有提交或 push。

## 后续设备结果：功能通过，完整 L1 无收益，不采用

根代理已收取上述两任务结果。两档完整精度、图重放、状态、信号及保护检查
均 PASS；下面是同进程三方五窗的完整 L1 中位数，不是子阶段时间：

| 档位 | 原分 task | d253 参考 | 候选 | 候选相对参考 |
| --- | ---: | ---: | ---: | ---: |
| 128K B16 | 876.46 μs | 877.40 μs | 877.86 μs | 慢 0.052% |
| 8K B32 | 982.24 μs | 989.88 μs | 997.74 μs | 慢 0.794% |

长 B16 五个配对窗口中，2 个更快、3 个更慢；短 B32 为 0 个更快、5 个更慢。
长档窗口计数已按根代理脚本更正，不能沿用最初口算的 1/5 更快。
因此减少 TLOAD 和 UB 占用没有转成完整 CSA 收益，**否决本候选，不合入正式、
不扩七档、不与其他候选组合**。源码、CPU 生成证据和设备任务 ID 均保留。
