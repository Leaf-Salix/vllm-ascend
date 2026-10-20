# 动态单 SPMD CSA：roofline 驱动的新一轮优化

本轮由用户重新设置的目标启动，覆盖此前“本轮后停止”的边界；完整目标见
`GOAL.md`。最终仍须五个 128K 档相对同轮多 task 完整 L1 降低至少 30%，
不能以候选结束或重点档净省 20 μs 代替目标完成。所有仓库未经授权不 push。

## 源码与测量边界

- 工作树起点 `f20efd87`；正式单 SPMD 目录没有未提交修改，内容仍为已验收
  `13259f92`。其它会话已经更新多 task，不能用早先表格作当前验收基线。
- 本轮冻结基线、候选、输入和独立工具链，同轮交替测完整 incore 首尾。
- 复用 `csa_critical_path_20261020` 的测试器和原冻结检查；每个新增候选的
  精度及性能结论都必须来自它自己的结果，不能借以前的 PASS。

## 首个候选：精确阈值筛选

目标是缩短长历史 Score 的 AIV 排序/归并服务时间，减轻 Cube 因消费者
处理不完而产生的等待。已有资源模型给出 B16/B28 排序最低服务量约
80/148 μs，但这不是可直接节省的完整延迟；Cube/FIX 也可能限制整体推进。

使用前一段 Top-512 的第 512 个 score 作阈值，只剔除严格低于阈值的项。
等于阈值的候选必须保留。首段没有可信阈值，保持原排序；阈值未能筛掉足够
候选时应回到原路径。所有 score、因果边界、页保护与动态数据保持。

新增成本包括比较掩码、保序压缩、取计数、按数量选择排序长度以及 UB 临时区。
先核实 Sort32/MrgSort 的同分顺序与压缩的关系，不能默认“集合相同”即可。
CPU 版本的 Sort32 按 index 解决同分，不能直接代替 A3 硬件行为。
`sort_tie_probe.py` 是小型设备语义检查，不测完整 CSA 性能。

本候选与旧 Score 转置、N128 面板、wave 交错不同：不改 Cube 数学或
物理页访问、不增加阶段交错，而是尝试减少实际参与排序的项数。
编译需确认实际排序长度缩短、计数保持设备端、没有逐元素 Scalar 扫描；
设备需确认同分顺序与所有冻结检查。若控制/压缩成本抵消收益，或一重点档
不能净省 20 μs、另一档明显退步，则停止该候选，不扩档。

## 接口调查起点及后续组件结论

当前 `pl.tile.gather_mask` 只接受固定硬件 pattern；`gather_compare` 的 A3
实现另有逐元素 Scalar 构造 index 和阈值类型约束，不能未经生成代码核对
就当成低成本浮点阈值压缩。优先查清现有接口/底层能力，必要时在独立
依赖 worktree 补齐；确认的组件 bug 按目标提交上游 issue，不提 PR。

后续已完成小型复现，提交 PyPTO #3079、PTOAS #1647、PTO-ISA #364；
issue 正文、链接和本地建议补丁均在本目录。PyPTO 新独立构建的 39 项 UT
通过；新 PTOAS 完整构建进行中。GT/GE 人工拼接 C++ 的 CCEC 成功仅是
底层可编译性预检，不能当成真实后端或设备压缩通过。

## Attention 独立数值候选

用户已授权独立验证统一 softmax 尺度后在 Cube 跨五块累加。设计、代价、
舍入差异及所有 UB 失败尝试详见 `ATTENTION_GLOBAL_SCALE_DESIGN.md`。
当前 v5 仅通过完整符号编译；B16 真机任务正在验证，没有性能收益结论。
正式算子未替换。公共源码快照入口 `run_frozen_full.py` 继续执行原检查，
不会为数值候选隐藏逐 bit 差异。

## 参考证据

- 当前 `13259f92` 全 CSA PipeUtilization：
  `../csa_critical_path_20261020/current_pipe_pmu/PIPE_COUNTERS.json`。
- 72 角色阶段端点与等待：
  `../csa_critical_path_20261020/CURRENT_SCORE_PRESSURE_ANALYSIS.json`。
- 七档 Cube/Vector/内存模型：
  `../csa_five_way_latest_20261020/theory_roofline/FULL_ROOFLINE.md`。
- 历史负例：`../csa_critical_path_20261020/CURRENT_CHECKPOINT.md` 以及
  主优化日志。旧 wave/转置/共享 Left/SDMA 等不原样重跑。
