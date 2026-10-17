# Attention 有限容量 Vector 物理行 plan 候选（2026-10-18）

当前仅为独立 worktree 中默认关闭的候选，不提交、不改主仓和依赖。完整 CSA 七档相对原动态分 task 的 L1 降低至少 30% 目标没有因本实验被改变；不能拿 plan 的指令减少推断整层收益。

## 假设与范围

原 plan 输出 512 个逻辑压缩 token 索引，随后两路 AIV gather 对每个候选重复做除 32、页列检查、Scalar 查表与物理行推导。本候选在既有 plan worker 的 M8 行内用临时 Vector Tile 完成一次地址合法性筛选与物理行映射，消费者直接读取有效物理行。原始 TopK 输出不改，只有内部 indices 工作区含义在同一谓词下改变。无新增根 Tensor、GM 工作区或 phase。

启用条件：`PLAN_PHYSICAL_ROWS` 为真，压缩页表列数 `1 <= C <= 2048`，物理页数 `1 <= P <= 2**24`；其余容量整次退回原逻辑含义。生产和消费共用 `physical_plan_supported`。页号合法性比较可用 FP32，但最终地址只用 INT32 位移和加法；合法最大物理行小于 2**29。当前 cache 合同为完整 32 行物理页。

页表先加载到最大 2048 元素临时 Tile，用实际 C 设置 valid_shape；每行 gather 512 页号。不跨 tick 保存 UB Tile。负页、越界页、越界逻辑列、causal 无效、dummy 与陈旧 position 均得到 -1，且维持原 bias/block mask。保持 geometry 重读、SWA、全部 FFTS READY/ACK/drain、softmax 和输出归约数学，便于只归因于压缩分支的地址工作迁移。

与已失败的消费者 UB 地址缓存不同：这里在 plan 产生地址，删除消费者每 half 的重复页表推导；代价是少数活跃 plan worker 额外执行 Vector gather 与 mask。不能假设一定有收益。实际长 B16 描述符为 table[16,1026]、cache[525344,512]；短 B32 为 table[32,66]、cache[67616,512]，两者容量都可走候选。

## 已完成 CPU 证据

- `cpu_v1/result.json`：首轮完整动态符号编译通过，单 MIX、58 动态轴、130 根 Tensor ABI；没有根据失败反复改编译器表达，没有修改依赖。
- `check_oracle.py` / `oracle.json`：实际候选源码 AST 用 CPU DSL 解释执行，共 132 轮（开关各两态，A→B→A 页表，容量 0/1/32/66/255/1025/1026/2048/2049，以及物理页容量边界、odd/mixed/dummy/stale/T0）。独立逐候选 Python 整数期望、indices/bias/mask、pad poison、单写者覆盖、输入不变均通过。
- 源码 AST 审计确认原 causal/clamp/bias、softmax/归约赋值及全部 `pl.system` 调用不变。
- `source_freeze.json` 记录 2026-10-17 16:57:42 UTC 生产源码冻结；其 AST 与首轮 CPU 保存的源码相同。没有 hash。

CPU oracle **不覆盖 A3 Tile 布局、valid_shape 及真实指令**，不能称设备精度通过。

## 设备验证与消息交错

第一次提交 `task_20261018_005742_348545224475` 后，收到需先补真实设备地址 oracle 的要求；任务当时正在候选编译，已主动终止。尚无完整 CSA 精度、性能结果，不把该终止视为数学或编译失败，也不重复统计成验证轮次。原输出保留在主验证目录 `parallel_20261018/physical_plan_h131072_b16_v1`。

新增 `device_plan_probe.py` 明确隔离导入本 worktree 候选 helper，显式 incore 边界，测试 C=1/66/1026/2048/2049、odd/mixed/dummy、负页/OOB 页/陈旧 position、固定输入输出地址 A→B→A；C=2049 验证原逻辑回退。TopK 保持原值，内部物理行独立期望、bias/mask、pad poison 与源输入逐项核对。不是 canonical 默认关闭路径的 plan probe。

`run_device.sh` 在同一次自动分配卡的任务里先执行私有地址 probe，全部通过后调用共享只读测试驱动，对 128K/B16 完整 CSA 做原动态 task、593c5d25 旧单 SPMD、候选三方比较，精度/状态/保护/协议/graph 通过后才收集 5 个完整 L1 窗口；额外沿用 metadata replay 页表交换。源码继续冻结，脚本修改仅发生在旧任务确认终止之后。

复现：

```bash
task-submit --device auto --max-time 1200 "bash /data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-physical-plan-20261018/tests/pypto_test/ultra_physical_plan_20261018/run_device.sh"
```

设备结果待执行后追加；此时没有性能结论。

## 真机私有地址 probe 结果

任务 `task_20261018_010403_360472313010` 于 01:04 提交；同任务第一阶段 `device_probe_v1/result.json` 的 15 项真实 A3 地址验证全部 PASS。包含四种物理路径容量与 C=2049 逻辑回退，每种 A→B→A、poison/源输入/地址固定检查。随后已进入 128K/B16 整层三方流程，此处暂不预记整层 PASS 或性能。

## 最终整层结果与取舍

任务 `task_20261018_010403_360472313010` completed(exit=0)。正式第4层权重，同进程、同一批根 Tensor 地址：候选完整 A→B→A 三次精确对比、TopK/scores、五份 Native cache/state 完整 allocation、所有输入与 scratch guard、phase 协议通过；metadata 改变与页表交换的同一图 6 次重放通过；三方矩阵 7 项检查通过。私有地址 probe 已独立覆盖改变的 indices 内部语义。

完整 L1 incore 首尾耗时（μs），每侧 5 窗口：

| 实现 | 五次采样 | 中位数 |
|---|---|---:|
| 原动态分 task PTO | 887.82 / 880.32 / 858.44 / 894.06 / 874.94 | 880.32 |
| 593c5d25 旧单 SPMD | 913.06 / 902.06 / 904.24 / 907.82 / 904.78 | 904.78 |
| 物理行 plan 候选 | 906.10 / 893.26 / 897.44 / 889.88 / 901.88 | 897.44 |

相对旧单 SPMD 的名义下降仅 **0.811%（7.34 μs）**，相对原动态分 task **慢 1.945%**；没有达到降低 30% 目标。五窗口不是独立统计显著性证明，不能把 Vector 地址解析的局部理论收益当完整层收益。该实现没有明显整层回退，但小幅收益不足以在本轮继续扩大验证；保留默认关闭，不提交、不合入、不再发任务。其它六档没有实测，不作外推。

完整结果：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/physical_plan_h131072_b16_v2/result.json`；同目录 `source_matrix/result.json` 含三方样本与各泳道路径。私有 `summary.json` 汇总本轮真实状态、范围和局限。原取消任务、全部源码与 CPU/设备产物均保留；不做hash。
