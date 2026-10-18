# 2026-10-19：同一GM的独立列分片导致多余barrier

## 结论及范围

现有流水中的Attention PV每次写回在生成C++里都有FIX barrier。
源码定位：PTOAS把partition_view当父GM根的别名，丢掉不相交列的范围。
最小IR在0.66/0.67复现；直接检查上游默认分支master仍有相同别名传播。
这是保守同步导致的优化缺失，不是已发现的数值正确性bug。
上游issue https://github.com/hw-native-sys/PTOAS/issues/1624 已附完整复现和
建议patch，没有PR/标签。ISSUE_POSTED.json保存远端回读。

局部修复只识别直接二维ND view、同基指针/类型、相同静态2次幂行距、
单位内stride、64B对齐且行内互斥的固定列窗口。行起点仍可动态。其它布局、
动态或不同stride、未知指针偏移、动态列、真实重叠全部保留原保守分析。
未知地址标记优先保留，不靠删除生成C++的barrier获得加速。

## 验证证据

12项CPU对照通过，加上跨循环同列复用共13项FileCheck通过。
循环复用仍保留FIX barrier；正反例见EXPECTED/FIXED_RESULTS及LOOP_REUSE_RESULTS。
指针偏移用例最初两次IR语法写错，在解析阶段退出，修正语法后通过；旧日志
保留，不能当作PTOAS bug。远端首次误查main得到404，确认默认master后
正确读取；UPSTREAM_TRANSLATOR_EXCERPT保存相关片段。

同一份已验证完整CSA PTO IR在新旧编译器生成C++：FIX barrier81→77、
MTE3 barrier112→110；1675对set/wait与全部数学、搬运及FFTS调用数不变。
滤去set_flag/wait_flag/pipe_barrier语句后，其余C++逐行完全一致。
四处FIX删除对应PV列写出；两处MTE3删除对应其它可证明列分片。
这只是静态生成物证据，不是运行次数、正确性或性能结论。

## 工具链隔离与复现

实现位于独立PTOAS worktree的MemoryDependentAnalyzer.cpp；此前#1623
的Acc列窗口修正及原并行赋值修正仍保留。只重建PTOTransforms静态archive，
根据Ninja链接参数把新bin另行链接到本目录toolchain/bin/ptoas。
没有覆盖原build目录的bin/ptoas，排队中的PV行块实验仍用旧版。
原可执行文件另拷贝到reference_toolchain/bin/ptoas，两侧均冻结。
LINK_COMMAND/BINARY_ISOLATION记录操作与路径，不计算hash；二进制不入Git。

同进程三方设备驱动保持原fixture、权重、精度、graph及L1流程。只给候选
PreparedDynamicCSA的调用绑定新PTOAS_ROOT；原多task和参考仍用冻结旧版。
调用后恢复环境，graph capture时也使用对应选择，避免工具链cache身份漂移。
COMPILER_INVOCATIONS.jsonl将记录实际编译路径，不能只凭命令行变量宣称生效。
两个source目录逐字相同，均为已验证正式动态算法，不叠加PV重分块或L1三槽。
没有修改生产接口，compiler_context.py仅为本次测试隔离。

两个代表档128K/B16、8K/B32/S6已提交，命令和task id在DEVICE_SUBMISSIONS。
首轮在归档脚本时因仍引用不存在的run_device.sh退出，尚未加载权重或执行
CSA；不是组件失败。修正为实际run_device_compiler_compare.sh，并核对
归档tuple的全部输入路径后提交v2，当前待设备结果，不能预记PASS或加速。先验精度/图/保护/协议，之后
按同进程同根三方五窗完整CSA L1决定是否采用；数学源码不改也须设备验收。

本目录Git只收小型IR/C++、patch、脚本、说明和结果；完整CSA C++、本地
编译器及后续泳道图仅保留路径。正式ops暂时恢复已验证版本，候选待验收。

第三版前再次补齐接口清单：v2已加载权重，但因缺少source_interface.json
仍在任何CSA调用前退出。补上原清单的相对引用，并扫描整个driver的HERE
字面路径，全部存在且JSON可解析。两次harness遗漏均已保留失败报告，
不能写作编译器/算子失败。v3已经进入原多task编译执行，实际调用日志确认
该侧使用冻结reference_toolchain；候选/参考后续仍须逐条核对。

PTOAS源级修改已本地中文提交a5e0a766，包含#1623/#1624修正和20项回归；
没有提交PR或push。共享安装和原build/bin仍保持，新的可执行文件另行链接。

## v3真机终态

两个任务均completed、exit=0，全部精确输出/Top-K/score/cache/state、保护、
代际和图重放门槛通过。实际PTOAS调用日志已核对：只有一个候选完整CSA用
新bin，完整CSA参考和所有原多task仍用冻结旧bin；没有把两侧都换成修正版。

| 场景 | 多task μs | 原单SPMD μs | GM修正版 μs | 相对原单SPMD耗时降低 |
|---|---:|---:|---:|---:|
| 128K/B16/S6 | 871.18 | 840.80 | 821.58 | 2.286% |
| 8K/B32/S6 | 979.76 | 946.58 | 960.54 | -1.475% |

长档五窗均更快，短档四窗更慢，结果不能称为统一收益。数学源码相同、
静态同步少了不代表所有区间必然更快。当前不整体采用，不修改共享安装或
正式算子策略。保留结果，下一步先区分四个FIX与两个MTE3删除的影响，
再在确有理由时组合重验Acc等历史候选。七档30%目标仍未达成。
全部窗口、实际编译器路径和任务原始报告在ALL_RESULTS及各v3输出目录。

## 下一步：只移除本CSA四处FIX的受限候选

在a5e0a766之上加本地FP32类型限制，另行链接toolchain_fp32/bin/ptoas。
这不是声称FP32访问都属于FIX；用完整CSA IR确认本例只减少4个FIX，
MTE3数目保持112，非同步C++仍逐行相同（FP32_ONLY_CODEGEN）。
前两份工具链不覆盖。两个同根三方五窗任务已提交，尚待结果，不提前采用。
`fp32_only_experiment.diff`是本地消融差异，不替代上游通用建议patch。
