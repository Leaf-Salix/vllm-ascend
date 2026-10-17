# HC Sinkhorn 标量候选：静态审计，尚未接入

2026-10-17。此目录只提供独立候选，不修改生产 `ops/pypto`、共享生成器或依赖仓库。
没有执行 PyPTO parse/lowering、CCEC 编译或 NPU 测试，没有性能或精度通过结论。

## 改动范围

`transform.py::transform(source)` 接收冻结核体源码，返回新源码和审计字典。
目标是 `comb_sinkhorn` 中首轮稳定 softmax / 列归一化之后的 19 轮迭代。
首轮、函数参数、返回值、全部 GM load/store、完整行/尾行写回分支、其它函数的 AST 均不变。
不增加工作区、跨核信号或同步，也不改变调度器的依赖和成员分配。

原成员处理 8 个 token，候选按 `pl.range(valid_rows)` 逐 token 执行；每个 token 在
16 个设备 FP32 标量中完成 19 轮。支持原成员映射范围内任意 T 的尾块，未添加七档白名单。
只处理真实行，原尾部 scratch 中无效行的数值不承诺与旧算法相同；原 scratch 写回范围不变，
有效输出的读取仍由原 `valid_rows` 限制。实际 scratch/保护区门禁仍须在设备验证。

参考了 `vllm_ascend/ops/pypto/hc_pre.py` 的设备标量思路，但没有照搬其倒数乘法。
候选每个元素仍执行 `numerator / denominator`，先行归一化，再列归一化，eps 位置不变。
列和显式保持 `((row0 + row1) + (row2 + row3)) + eps`。

## 不能静态保证逐 bit 的原因

1. 冻结 A3 向量除法使用 `vdiv`，候选是 Scalar FP32 `/`。同为 FP32 不表示舍入结果相同；
   此项目此前 QR 已遇到这两种除法的舍入差异。保持除法顺序只能避免额外改动数学表达式，
   不能证明当前精度门槛通过。
2. 冻结行归约来自 `pl.tile.row_sum`。当前本地 PTO-ISA 的
   `include/pto/npu/a2a3/TRowSum.hpp` 第 32～36 行将 FP32 归约落到 `vcadd`；
   `TRowReduceOps.hpp` 第 293～305 行对本核 `[8,8]` tile 选择单 repeat 路径，
   第 210～235 行设置 mask 后调用该指令。包装源码没有展开 `vcadd` 的内部加法树。
   候选明确使用 `((col0 + col1) + (col2 + col3)) + eps`，不能仅凭包装代码断言
   与硬件归约树逐 bit 一致。参考 HC_pre 的标量实现使用同样相邻分组，也不能替代本候选验证。
3. 首轮仍完全使用原向量数学，未引入 Scalar exp 或改变稳定 softmax。

上述是候选的真实未证实项；禁止降低精度门槛、静默修改基线或把 CPU 审计称为精度通过。

## 性能风险与后续决策

HC_pre 小档标量路径每核通常只处理一两个矩阵；这里保留原 8 行成员。
候选最多串行执行 `8 × 19 × 32 = 4864` 次 Scalar 除法。
即使省掉短向量指令，8 行串行化也可能明显变慢；comb_sinkhorn 还不一定处于完整 CSA 的关键路径。
当前不接入口、不提交设备任务。若后续决定继续，应先看完整 CSA 的依赖和关键链证据，
再独立决定是否改变每成员行数，并重新证明覆盖、尾块存储和调度依赖。

## 本次审计

执行：

```bash
python tests/pypto_test/csa_multishape_spmd_20261016/hc_sinkhorn_scalar_v1/audit_source.py
```

结果见 `source_audit.json`；可直接审阅 `candidate_comb_sinkhorn.py.txt`。
审计检查 AST 修改边界、GM 访问不变、16 次 tile 读写、每轮 32 个直接除法，
并枚举 T=1..384 的原 8 行成员划分，确认真实 token 恰好覆盖一次。
这只是 Python/AST 与逻辑覆盖检查，不是 PyPTO 解析、编译、设备精度、调度正确性或性能证据。
