# 当前交接：目标 active，Attention 已按用户要求放弃

本轮设计、尝试、组件问题及结果均在此目录；正式算子未替换，没有push。
上一阶段本地提交64934359。过程不是目标完成证明，仍未达到五档128K的30%。

## 已完成的真实证据

- 新PyPTO独立树pypto-csa-topk-compact-20261020：计数物理容量对齐；构建完成，39项UT通过。
- 新PTOAS独立树ptoas-csa-topk-compact-20261020：全量构建及随后增量构建完成。
  第一次完整构建期间修改了被PTO.cpp包含的验证源，旧对象仍在首个二进制内；
  增量重新编译PTO.cpp后GE生效。不是另换旧安装，也没有复制构建产物。
- 三项针对性PTOAS RUN/FileCheck回归通过：ptoas_regressions_v2/result.json。
  lit本机初始化和FileCheck路径的失败记录保留，不冒充编译器bug。
- compact_device_v1：task_20261020_193245_43650423493，59组×3次同地址图重放，
  最终512对score/index逐bit相同，输入及保护区通过。尚需独立计数probe排除回退掩盖。
- TopK完整B16/B28任务task_20261020_193448_6087139012和task_20261020_193448_60878422533
  均完成严格精度、状态、保护、图和代际检查。五窗完整L1为：B16参考798.14、候选798.60；
  B28参考1163.24、候选1154.14μs。净省-0.46/9.10，未达20μs门槛，不扩档不采用。
- Attention两重点档已按用户追加授权完成独立性能诊断；输出数值FAIL保留，
  自身eager/graph、状态和协议通过。B16 792.44→842.80，B28 1169.48→1244.00μs。
  用户随后明确“放弃attention，干别的”。不再做该方向的调测或分析，保留历史负结果。
- 数值、窗口、决定统一见FOCUS_COMPARISON.json/md；没有用PMU时长替代L1。

## 当前待取的具体任务

1. task_20261020_194255_125154822107：gather_direct_device_v1。
   独立核对FP32 GE真实count和保序索引、非零offset，防止完整排序回退掩盖错误。
2. task_20261020_195203_1895914849：memory_profile_b28_v1。
   只补当前正式单SPMD/冻结多task的MemoryAccess与L2计数，每侧三次；
   没有新算子候选，不更新L1性能表。当前队列被整机CI占用，任务pending。

检查具体task状态后继续，不能因为等待时间长重新提交或取消他人作业。

## 下一步边界

- 仍优先128K/B16、B28。TopK v1已经不足门槛，不盲目扩档或重测。
- 已验证当前权重cache_policy=1是BYPASS，但PTOAS无offset时仍生成普通TLOAD。
  不能据IR属性声称缓存策略已生效；不要猜offset或给整个可写KV/state根套BYPASS。
- PTOAS已有相关运行时offset问题1537；没有重复创建issue。
- PyPTO #3079、PTOAS #1647、PTO-ISA #364均已提交，建议与复现保留。
- 所有组件在独立worktree；pypto-lib只读，共享安装不改。
- 后续阶段性中文本地commit继续，未经再次明确授权禁止push。
