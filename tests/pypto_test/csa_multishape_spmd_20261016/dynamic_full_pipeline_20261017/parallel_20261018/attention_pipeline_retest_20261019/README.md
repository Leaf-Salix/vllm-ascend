# 2026-10-19：在FIX范围修正后重新评估Attention流水

用户要求旧负例不能永久排除。本轮的重验依据是：先前Acc分离消除了QK→PV
的错误依赖，但GM分片仍有额外FIX barrier；当前受限PTOAS已移除本CSA四处
FIX barrier、保留MTE3，需重新验证Acc分离是否能够释放新的重叠机会。

两个候选均使用当前正式动态源码为参考，T/B/metadata、AIV、FFTS、GM等待
与100tick退避不变。原多task及当前单SPMD由冻结旧PTOAS编译；候选由
E17/toolchain_fp32编译。同轮三方五窗，只比较完整CSA L1，先过冻结精度门槛。
本轮是算子与编译器组合收益，不把全部变化单独归给某个组件。

## 候选与编译验证

- source_acc_isolated：完整复用E14的96KiB Acc池。QK独占32KiB，PV两槽
  各32KiB。不改旧算法，只更换经过验证的受限编译器。ACC_ONLY_CODEGEN确认
  数学C++不变，FIX静态barrier减4，MTE3不变。
- source_acc_right_v2：在此基础上，PV预取下一片Right。每片128×128 BF16
  为32KiB，总L0B64KiB。当前片被MMA消费前下一片必须已定义，从生存期上
  让两片保持不同地址；下一片MTE1与当前片MMA可重叠。
- 第一版source_acc_right_pingpong把分支内Tile用于后续另一分支，在ConvertToSSA
  因定义域失败。改为固定N四片的序言、两片稳态、末片排空后CPU通过；
  没有修改编译器，没有固化任何B/T。保留原源码、日志与精简错误报告。
- cpu_acc_right_v2完整无样例shape编译通过，58动态轴。RIGHT_LOAD_SEQUENCE
  直接核对四次TEXTRACT的Right地址为0/32768/0/32768；Acc为0/32768/65536。
  RIGHT_PIPELINE_CALLS确认首两片各有独立MTE1事件，当前MMA不等待下一片。
  这些生成物证明双槽形成，不能代替真机性能结论。

两种候选各提交128K/B16、8K/B32/S6，任务和命令见DEVICE_SUBMISSIONS。
尚待完整精度、图重放和性能结果，不预先采用。大构建/完整IR错误/轨迹只
本地保留，提交小型证据及路径，不计算hash。

## 报告字段更正

复用的E17驱动原先是“相同算子、不同编译器”测试，硬编码same_operator_source=True。
E18实际上只改attention_stream_stages.py，见HARNESS_METADATA_NOTE。
该描述字段不适用，但加载的候选/参考源码及计算、比较流程均正确；原始
结果保留，不能拿该字段声称E18同源。后续驱动已逐文件比较真实bytes，
同时列出差异文件，不计算hash。

## 四个真机任务终态

四任务均completed、exit=0，完整精确输出/Top-K/score/cache/state/保护/代际/图通过。
各行均为各自同轮五窗中位，不能跨行直接相减归因Right。

| 候选 | 场景 | 原多task μs | 当前单SPMD μs | 候选 μs | 较当前耗时降低 |
|---|---|---:|---:|---:|---:|
| source_acc_isolated | 131072_b16 | 875.46 | 836.46 | 838.94 | -0.296% |
| source_acc_isolated | 8192_b32 | 993.78 | 940.84 | 951.66 | -1.150% |
| source_acc_right_v2 | 131072_b16 | 868.62 | 821.96 | 838.00 | -1.951% |
| source_acc_right_v2 | 8192_b32 | 987.56 | 944.24 | 942.06 | 0.231% |

未获得稳定实质收益，因此不采用、不扩七档。双槽已经真实形成，仍不能把
可并行就当作必然加速。正式Attention恢复参考，继续检查较大的Score供数链；
本轮不改生产GM等待或缓存协议。
