# PyPTO FIXPIPE epilogue 移植

上游：hw-native-sys/pypto b9240c1820a81c0e762642a34ddd284378cda8ea（#2838）。
本地起点88297437，保留feat/kernel-mode-integration-test，整份上游补丁干净应用。
目的：直接采用pypto-lib的N768/FP16双缓冲Score写法，需要store(pre_quant, pre_relu)。
Simpler a54c05095、PTOAS0.66、PTO-ISA327cd586均不变。

本地提交 `2a4e09ff`（中文说明，Signed-off-by）。按机器配置使用2个编译job构建，
随后安装当前checkout生成的PyPTO扩展到固定venv；未借用其他工作树产物。

- 构建与安装通过。
- FIXPIPE verifier/parser/codegen/frontend 定向CPU用例：35 passed。
- A3 `acc_to_gm_dequant_relu`：1 passed，任务 `task_20260927_111354_37988349724`。
  [设备日志](device.log)、[CPU 测试日志](unit_tests.log)。
- O projection 候选完整CSA编译通过；本记录不代表新的CSA精度或整模型验收。

v3起使用新工具链，旧基线数字属于移植前工具链。
