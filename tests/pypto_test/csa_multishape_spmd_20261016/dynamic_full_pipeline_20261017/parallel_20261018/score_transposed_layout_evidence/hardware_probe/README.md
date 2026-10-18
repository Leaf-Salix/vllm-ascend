# Key-major Score回退的硬件诊断

完整CSA首版已精确通过，但完整L1为1533.96μs（同轮正式880.26）。
本任务只定位Score自身的pipe成本：复用候选同一helper、完整CSA实际中间Tensor，
先核对完整pair/score allocation，再采三步PipeUtilization。

任务task_20261018_133558_38898924592；仅一个A输入，不再做三方L1/七档重测。
输入准备仍经过原多task精确参考；没有合成理想GEMM替代实际Score。
当前在运行，结论待PMU结果。不得将完整L1回退先验归于TTRANS、Scalar或某条DMA。

## 诊断结果（已完成）

任务exit0，完整Score pair/score在eager、graph及profiling后三次逐bit一致。
三步PMU中位：AIC MTE1 63.471μs、矩阵112.801μs、FIX876.487μs、
MTE2 549.857μs；AIV Vector108.386μs、MTE2 707.563μs。独立Score
图外Event1167.450μs。原路径独立诊断MTE1约155、矩阵104、FIX133、
Vector99μs，属于不同轮次，仅定位数量级变化，不当配对端到端收益。

结果支持优先核对FIX/GM交接，不能把大幅回退主要归给新增TTRANS数学。
下一版只改变两次WS store的valid列6→16和GM行stride8→16；前六列
仍为同一结果，其余本就计算出的列写入padding。AIV仍只读取前8列，
转置、TopK及事件不变。这是根据PMU提出的窄列写回假设，待验证。
