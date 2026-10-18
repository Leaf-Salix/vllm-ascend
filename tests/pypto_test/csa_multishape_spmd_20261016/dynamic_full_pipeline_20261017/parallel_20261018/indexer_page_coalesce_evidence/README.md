# Indexer 连续物理页的合并搬运候选

目标仍为七档完整动态 CSA 单 SPMD 各自较同轮原多 task L1 降低至少30%。
最近几个用户交互完成性能表格和展示公式修正，没有新增优化成绩；正式版本
仍为 f222a917，七档尚未达标。本目录先验证新的搬运路线，不复测旧方案。

现有长历史 G6/N64 路径每面板分别读取两份 Native Key 页，每页4096B Key
后至少64B scale。连续物理页可用多 ND 的 ND→NZ 指令跳过页尾并一次拼入
原 Mat 面板。本地 PTO-ISA common TLoad 的3维实现支持此能力；仍须核对
PyPTO 实际 pinned include、描述符 stride、生成物及真实 A3 数值，不能仅据
共享源码的接口存在就宣称当前完整链可用。

第一步 probe.py：共享 Native storage 的正 stride [P,32,128] 视图，无业务
数据复制、无 device→Host 回读。页起点来自 device Tensor；同一图依次读取
偶页、奇页、最后合法两页再恢复。检查完整 INT8 点积、输入及输出 poison。
不测试性能、不把单独搬运 PASS 当作完整 CSA 功能或性能结果。

后续完整候选必须保留非连续页和尾页的原 gather 回退，不改变数学、score
顺序、READY/ACK 或缓存发布规则。只有设备页表实值满足相邻条件时才合并，
不能假定服务页表总是连续；收益也应注明测量输入的连续页比例。

本轮未修改依赖库和共享安装。结果、失败与取舍逐项追加，不做 hash 校验。

## 已完成的接口核对

cpu_v1符号编译通过；实际pinned include也支持A2/A3 multi-ND，但PyPTO当前
只给声明生成canonical连续视图，不能以这个CPU通过证明带间隙页可用。
第一任务因测试脚本漏收队列的`--device`退出2，未调用算子；修正参数并核对
TASK_DEVICE后，第二任务`task_20261018_220253_21721932796`退出1，入口明确
拒绝非连续stride：`Parameter 'native_keys' requires a contiguous strided tensor; no copy is made`。
保护正常生效，未执行候选搬运，不能称设备数值失败或静默误算。

上游#1964曾明确延后stride参数；本轮查main确认仍未暴露，作为功能需求提交
[PyPTO #3038](https://github.com/hw-native-sys/pypto/issues/3038)，包含实际场景、
复现、接口/alias/边界/flatten建议及验证范围。不创建PR、不加标签。
本路线暂不接入CSA，继续独立推进O_A权重复用，正式数据流保持原逐页gather。
