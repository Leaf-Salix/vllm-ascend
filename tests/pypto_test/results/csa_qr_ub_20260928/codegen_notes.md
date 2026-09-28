# QR输入驻留的CPU生成代码证据

编译入口为compile.py，完整lowering、PTOAS、CCE、AICPU链接通过；尚无设备收益结论。
生成文件：`compiled/kernels/aiv/qr_rms_norm_quant.cpp`。

- gamma在worker外层循环之前TLOAD一次，随后转换为FP32常驻tile。
- 每8行只有一个8×1024 FP32输入TLOAD，驻留于`qr_input_ub`。
- 最终使用显式`tile.extract`：第一个256列块生成`TEXTRACT(v118, v104, ...)`，
  平方为`TMUL(v122, v118, v118)`，后续gamma相乘复用v118；不会为两个乘法操作数重做TEXTRACT。
- 两阶段仍按256列迭代；TROWSUM/TROWMAX、累加次序、三操作数高精度TRSQRT、
  RINT→INT32→FP16→INT8 TRUNC的量化次序保持。
- padded输出及valid_rows保护沿用原范围，任务数/worker上限/early_resolve保持原配置。

首版view式tile.slice在三个使用点分别生成TEXTRACT；它对应的设备任务
`task_20260928_143632_39768493767`在pending期间取消，没有执行或产生性能样本。
这项调整来自生成代码证据，不是用反复设备试验寻找更好结果。
最终候选单卡任务为`task_20260928_143910_419318230381`。
