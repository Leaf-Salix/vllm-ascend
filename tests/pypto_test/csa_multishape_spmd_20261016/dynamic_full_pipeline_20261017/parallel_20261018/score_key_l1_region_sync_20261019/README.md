# Indexer key L1预取范围的同步分析

从完整CSA生成物观察到：TEXTRACT读取L1双槽中的第一个槽后，
MTE2写另一个槽前仍有MTE1→MTE2等待。本目录从最小复现和
物理分形布局证明开始，现已完成独立编译器修正及七档真机验证。

原key池Mat INT8[128,128]为ColMajor/RowMajor boxed512，
以同地址RowMajor/ColMajor转置视图供TEXTRACT读取[128,64]。
写另槽的Subview为原视图的行64..95，读取的是原行0..63。
不能仅凭逻辑不相交删同步：需证明真实物理分段，并保留重叠、
动态offset及未知布局回退，另查循环跨迭代复用的真实依赖。

## CPU与首轮真机证据

PTOAS #1625已提交，含最小复现、物理范围、建议patch；#1620已补
显式未知基址的保守标记建议。218组字节窗口枚举通过，9项Key范围与
循环FileCheck、7项既有Acc范围回归通过；显式A3输出和默认输出相同。
新编译器保留旧GM分析，没叠加未采用的#1624优化，原两个bin的大小及
mtime保持。独立工作树里的GM FP32实验源码也未改动。

同一完整CSA IR只有同步语句变化；所有计算、搬运、FFTS及PIPE_ALL
保持。set/wait静态数1706→1707，不把条数增加/减少当作收益依据；
其key依赖已从立即前驱改为每个槽对应的旧读取，使另一槽可以预取。

首轮同进程五窗全精确/状态/保护/协议/图均PASS，参考→候选：
128K/B28 1244.46→1224.18us（降低1.630%，五窗都更快）；
8K/B32 933.30→926.38us（降低0.741%，幅度小）。多task分别
1309.96/976.56us。其余五个验收点已提交，不能据此宣称七档达标。

## 七档完成与后续工具链

七档均exit0，全冻结精确/状态/保护/协议/图通过，B16 metadata/padding
通过。完整窗口见ALL_RESULTS；六档中位下降，B8慢0.20us基本持平。
相对原多task的30%目标仍0/7，不能把对旧SPMD的增益替代总目标。

v3增加常量cast的保守穿透约束。原完整流水对两个cast控制本已正确，
这不是新复现的设备错误；记录见NARROWING_CAST_NOTE。11项范围/循环/
cast FileCheck和7项Acc回归通过。七档实际PTO重生成的完整C++与实测v2
逐文件相同，包含同步，见V3_SEVEN_CPP_IDENTITY，故未重复设备测量。

后续CSA实验选toolchain_v3；run_device_selected.sh令原多task、参考
SPMD和候选全部用同一工具链，未来原多task基线需在此条件下重新计量。
run_device.sh维持本轮旧/新编译器隔离对照，历史记录不改。共享env和
共享安装不变。CSA_SPMD_CURRENT_TOOLCHAIN.json记录新基线入口。
PTOAS工作树的本次源码提交为`b7b4a82c`，包含Translator和11项回归；
构建另用旧GM分析对象覆盖工作树已存在的未采用实验，不能直接把任意
该工作树全量构建当作本次实测二进制。重建命令与源码分别见LINK_COMMAND_v3、
REFERENCE_GM_COMPILE_COMMAND、source/，大二进制仅本地保留。
