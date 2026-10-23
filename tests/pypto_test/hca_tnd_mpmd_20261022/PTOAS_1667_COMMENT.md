补充2026-10-22的真实A3 AscendC编译证据（没有设备执行结果）：

HCA TND输出量化候选在本核Scalar合并8个FP32最大值时命中本问题。
公共PTOAS源码914863ed3d0eb3e92a4433a4a706a86cabf4c91f；PyPTO26e94e05627eb47710dcc214bae1774b8829df8a；CANN9.2.0-beta.2，a2a3/AIV。

PTOAS翻译及编排C++编译成功，但最终AscendC编译报：

```text
hca_native_token_quant.cpp:278:22: error: no matching function for call to 'ptoas_bitcast'
  uint32_t v83 = ptoas_bitcast<uint32_t>(v74);
hca_native_token_quant.cpp:62:18: note: candidate function not viable:
  call to __host__ function from __aicore__ function
  static inline To ptoas_bitcast(From from) {
```

同文件第325行第二处调用也失败。说明问题不只是检查生成文本，已实际阻断A3设备代码的编译/链接；本次失败发生在新算子运行之前。

未修改共享SDK。本地算子候选改用Vector分段row_max、复制后col_max产生全行amax，规避Scalar max；这不是建议上游回避修复。仍建议为helper补正确设备限定，并在回归中实际编译生成的C++，保留NaN和signed-zero语义。没有提修复PR或新增重复issue。
