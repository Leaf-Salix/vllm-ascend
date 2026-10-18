# CCEC实际对象与资源证据边界

为核对融合大函数是否受寄存器/栈影响，CPU重编同轮已验证的正式动态AIC wrapper，
复用KernelCompiler的真实平台、优化及include，仅加save-temps=obj。没有执行NPU，
没有改共享编译器或算子。`dynamic_aic_v1/result.json`完成PASS，保留.o与.elf。

本次实物为elf64-hiipu：.text为50200B，kernel_entry为2048B，cube_schedule主体
为48152B。对象中的.ascend.stack.size.record是32768；它与命令行0x8000预算
相同，不能当作实际spill或占用量。保存C++名称/字节数仍不能证明延迟根因。

save-temps未得到.s/.ll；额外尝试编译器寄存器分配转储时，CCEC明确拒绝
`-print-after`，动态/静态两次CPU命令均退出1，没有设备执行。
`-###`同样被驱动拒绝；SDK llvm-objdump的disassemble-aicore只打印对象类型，
没有指令助记符。本轮不从这些缺失数据推断寄存器溢出，也不继续扫未公开选项。

保留命令、日志、对象路径及失败结果。二进制仅本地，不计算散列；正式性能值未改。
下一步通过真实Score函数的独立MIX硬件流水指标区分供数/计算限制，见相邻
`indexer_score_hardware_evidence`。那也是归因用例，不能替代完整CSA验收。
