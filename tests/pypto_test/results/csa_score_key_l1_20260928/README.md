# 长档Score：Key L1与Score L1分开

从2d2f9ca0独立工作树`.cache/csa-score-key-l1-2d2f9ca0`验证，生产源码未改。
前版仅预取Key到L0B，两档状态通过，但128K/B16核内+14.87%、完整CSA七三+2.917%，不保留。
[上一版设备证据](../csa_score_key_prefetch_20260928/README.md)继续有效，本项不原样重测该失败版本。

依据最新ops-transformer b5b33e14的QLI V2 `InitBuffers/ProcessQk`：
Key L1与FIXPIPE写回的Score L1是分开的双槽buffer。
上一PTO候选的Key/Score复用同一96KiB L1空间，每个N1024块出现15处MTE1→FIX保护。
这证明存在源码/生成依赖差异，但没有逐指令stall，不能把上一版全部回退归给该等待。

本项只在原长档S6分派保留Key L0B预发，并增加一个跨panel存活的16KiB Key L1池。
两个N64×D128 INT8面板通过`gather_row`按0/64行填入，再由转置视图TEXTRACT到L0B。
Native paged cache的0/64B视图、物理页映射、tail clamp、读写总量保持；没有入口拆分或新增GM复制。
FP16量化/两次Cube规约、M384/N64、24个MIX任务、固定规约与调度标志均沿用基线。
与pypto-lib仍有原生分页cache和query成组的既有区别；本项借鉴AscendC的缓冲寿命，
不将上游连续cache的数据准备成本混入此核内试验，也不改变精度版。

## CPU编译与实际地址

[候选补丁](candidate.patch)、[完整编译入口](compile.py)、[日志](compile.log)。
完整lowering、PTOAS、CCE、AICPU链接及load通过；编译结束后才提交设备任务。
不修改PyPTO/Simpler/PTOAS/PTO-ISA，未占卡进行CPU编译。

[生成代码提取](codegen.py)、[逐操作地址证据](codegen_evidence.json)：

- Key L1池在49152起，占16KiB；两个Score L1在0和65536起，各48KiB，Mat最大末端112KiB。
- Key L0B首块在16384，稳态0/8192交替，WS占16384起48KiB；L0B合计64KiB。
- 每个N1024块仍16次Key搬入、16次QK、16次WS；Key TEXTRACT的列偏移0/64交替。
- 原15处MTE1→FIX等待降为0。其他必要生产者/消费者同步仍保留，不宣称无stall。
- 对未启用预取的四组Score AIC/AIV及长S6 AIV定向比较，9份生成二进制一致，
  [文件及大小](unchanged_kernels.json)保留；不做仓库/权重hash扫描。

## 设备合同

[命令](run_layer.sh)已提交`task_20260928_175014_22857933418`。
长短B16同一2d2f9ca0基线，layer4真实权重/合成历史、反向物理页与逐物理行scale，mode2/atomic0/det0。
每侧5预热、20次无profiler图计时、4个独立DFX窗口，长短交换基线/候选顺序；保留Native控制。
八类状态零容差、metadata/保护区、自重放、计时图状态及候选A→B→A检查。
先看128K Score核内，并分别报告完整CSA、P95及长短变化率7:3；不能由均值覆盖状态错误或长尾。
[CPU汇总入口](summarize.sh)只复用已有收集器及证据提取，不增加设备执行。
原任务退出1：两档计时/状态报告和长档每侧四个DFX已完成，随后8K基线DFX在`SetDevice`报507033，
设备子进程启动超时，尚未执行CSA。失败报告/日志保留在`h8192_b16/swimlane/baseline_device_open_failure/`，
[错误与已完成范围](device_open_failure.json)单列，不把设备启动故障归为算子正确性失败。
仅通过[补采入口](run_missing_swimlane.sh)提交`task_20260928_180158_266927127827`，采缺失的短档基线/候选四窗口；
没有重跑已完成计时或状态。最终汇总等待补齐，不改原任务退出码，也未合入生产或扩测EP16。
