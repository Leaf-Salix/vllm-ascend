# 上卡前检查

## 候选

- 正式三份基础源与 `ed2b2e5c3` 的 SHA 一致，正式工作树无修改。
- A仅改变 merge 输出接口和两次 BF16 store 的坐标，删除独立 heads 复制。
- B全部11处 heads slice 改为分组布局，显式使用真实 tokens；packed 缓冲在根级创建。
- 独立审查将改动逆向归一化后，完整数学 AST 与原函数相同。
- 独立审查枚举139264个 packed 地址病例、T1～256发布映射，未发现跨组或尾界错误。
- 两个候选外部根仍为64Tensor；MMA形状/packet起点/累加、WO-A BF16、8192全行量化与WO-B不变。

## 实际编译

独立无设备任务 direct / packed 均 exit0，完整根、内核和编排编译通过，未初始化NPU。
实际生成编排保存在 `orchestration-direct/`、`orchestration-packed/`。

- 两者均为根级 allocation 的父缓冲。
- merge 的 `add_output(heads/o_packed_heads)` 与后续 WO-A 的 `add_input` 指向同一描述符。
- merge 保留真实 QK 与 rope producer 依赖。
- WO-A 位于后续兄弟 scope；packed 在子 scope 返回别名被忽略时，仍通过父描述符生产/消费。
- B实际 rows 使用 `T_DYN`，未从 packed 的容量反推真实行数。
- 实际编排没有独立 `native_heads` 复制任务。

## 助手

独立审查确认计时先于Native/PTO数值比较。12组event使用六种排列各两次，restore与检查在计时外；
5组L1另行begin/end DFX，dep-gen关闭。每组结束、下次restore前检查有限输出、guard和readonly，
避免restore掩盖越界。执行有效性检查不代表精度通过。

筛选门槛在任务前固定：完整图event中位数至少快1%、12组至少10组领先，且L1中位数也更快。
达到门槛后才做Native A/B八边界及ABA；无收益候选不进行精度验收。
此门槛为候选筛选条件，稳定收益还需要第二个短任务复现，不能据一轮筛选宣称完成优化。

## 启动问题记录

前两个无设备启动分别因source覆盖shell变量 `job`、编译助手缺少torch_npu导入而退出。
均在占卡前修正，原基准、SDK和NPU环境没有改动；任务ID与状态保留在CPU_TASKS.json。
