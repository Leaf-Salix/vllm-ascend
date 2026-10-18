# Key-major Score回退的硬件诊断

完整CSA首版已精确通过，但完整L1为1533.96μs（同轮正式880.26）。
本任务只定位Score自身的pipe成本：复用候选同一helper、完整CSA实际中间Tensor，
先核对完整pair/score allocation，再采三步PipeUtilization。

任务task_20261018_133558_38898924592；仅一个A输入，不再做三方L1/七档重测。
输入准备仍经过原多task精确参考；没有合成理想GEMM替代实际Score。
当前在运行，结论待PMU结果。不得将完整L1回退先验归于TTRANS、Scalar或某条DMA。
