# Score 的 L1 提前两块、L0 提前一块

2026-10-18。在2db59be4的正式动态CSA上修改 `indexer_score_stages.py`，
其它数学和调度不变。目标是Score供数关键链，首轮只验证128K/B16/S6，
不预先扩测七档。默认PREFETCH仍为原值；实验使用策略覆盖prefetch=2。

## 当前证据和假设

已有真实Score的PMU显示MTE1活跃较高，不能用活跃时间相减推算可省时长。
再次查看当前生成代码，QK之后已经单独发出M→FIX事件，后面的WS有自己的
事件；因此“FIX必须等待WS，挪一行即可解决”的猜测不成立，没有实施。

当前Key的GM→L1和L1→Right都领先一块。候选把两种预取解耦：第j块计算
前只预取Key(j+2)到L1；当前QK结束后才把它送到旧Key(j)的Right槽。
Key(j+1)已经在另一个Right槽。起始预备Key0/1，末端禁止越过panel数。
这样保留两份Key Right容量，以当前QK/WS尝试覆盖更远的GM搬运延迟。
这是供数顺序假设，尚无设备收益结论。

区别于历史 `INDEXER_KEY_PAIRS_EXPERIMENT.md`：旧版每对先送第二Key到L1，
没有在原“下一Key已到Right”之外再前瞻一块；本次保留下一Key的Right常驻，
另把后下一Key送入空出的L1槽。区别于旧Key池扩大：本轮没有扩大池，且真实
搬运提交顺序改变。原QK/WS交错并非本轮新增成果。

## 动态性、数学和同步

- T/B/Q/L/位置/页表均仍读实际device输入；prefetch=2是编译策略，不是B16白名单。
- QK M384/N64/K128和WS M16/N64/K384、FP16缩放/ReLU边界保持原顺序。
- Native页安全映射和64B错位别名选择原样保留；总共仍加载16个Key panel。
- L1仍为双槽16KiB；稳定态Right为WS48KiB加Key8KiB×2，合计64KiB。
- L1覆盖等待旧MTE1读取结束，Right覆盖等待旧MMA结束；保留编译器生成的保护。
- 两AIV的完整函数AST不变，score双槽READY/ACK、末尾排空和阶段信号不变。

## CPU结果

`cpu_long_v1/result.json` 完整符号编译通过：130根、58动态轴、一个MIX，
没有传入B16或其他真实shape。生成Right WS地址0、后续Key地址49152/57344；
初始Key可以临时使用0，因为此时WS还未装入。L1池地址163840、容量16KiB。
第一批Key2的TLOAD在QK0前提交，Key2的TEXTRACT在QK0完成后；后续同类
顺序保留。资源与事件依据见 `CODEGEN_AUDIT.json`，不是硬件重叠时间证明。

源代码在正式ops目录实施，参考与候选完整冻结为 `source_reference/`、
`source_candidate_v1/`；驱动复用既有三方同进程、同根地址完整精确和L1检查。
设备经task-submit，只在输出/Top-K/score/cache/state/图/保护/代际通过后
采五窗口。不得将CPU通过或者后端指令顺序直接记成精度PASS/性能提升。

完整入口使用正式W8A8第4层权重，比较原多task PTO、此前动态SPMD与候选。
30%逐档目标仍按同轮原多task完整L1判断。所有失败与原始窗口保留。

## 真机终态：精确通过，完整耗时没有实质改善

任务 `task_20261018_232714_11062101501` 已完成、exit=0。输出、Top-K及其
score、cache/state、保护区、A→B→A图重放及代际检查全部通过。五窗完整L1
中位数：原多task 886.80 μs、正式动态SPMD 833.22 μs、候选832.74 μs。
候选只减少0.48 μs（0.0576%），逐对五窗仅两窗更快，不能证明稳定收益。
不扩测七档，也不默认采用。原始窗口和终态在 `TERMINAL_RESULTS.json`。

候选完整源码仍保留在 `source_candidate_v1/`。正式文件先与本轮冻结候选
逐字核对，再恢复已有正式版本，凭据在 `FORMAL_RESTORE.json`；没有覆盖
其他会话的修改。随后按用户要求转向完整72核incore流水，不继续扫预取参数。
