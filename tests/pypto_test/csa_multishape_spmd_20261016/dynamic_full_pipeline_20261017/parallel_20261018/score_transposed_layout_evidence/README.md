# 动态Score的Key-major计算布局候选

load2d替换仅让MTE1活跃时间减少约1.8μs，独立Score整体没有明确改善。
本候选进一步改变片上数据方向：两次矩阵同时转置，使大量FP16中间值
从Left非转置读取，保留Key双槽预取、QK/WS交错及原双槽READY/ACK。

| 运算 | 原布局 | 候选布局 |
| --- | --- | --- |
| INT8 QK | [G×64,128] × [128,N] | [N,128] × [128,G×64] |
| FP16 head归约WS | [16,G×64] × [G×64,N] | [N,G×64] × [G×64,16] |

原query/系数常驻Left，WS中间值走Right；候选query/系数常驻Right，
WS走Left。整数点积、FIX 2^-10/ReLU/FP16边界、WS的K次序均保持。
不能仅凭代数等价宣称逐bit相同，设备原PTO精确检查仍是必要门槛。

## 动态输入及交接

新增路径覆盖既有G6/N64/1024/prefetch1/toL0/balanced数学策略，非B16
白名单；其它策略保留原helper。B、T、Q/L、位置、页表、有效leaf与循环
次数仍由动态Tensor/device metadata决定。S6只作为当前设备验证点。

每槽由query-major[6,1024]改为key-major[1024,8]，AIC仅写前6列；
双AIV各读一次32KiB，片上转置后取原来的3条query，再执行原scale、
sort/merge。空尾lane仍消费事件和ACK。score/pair最终布局及所有原
根发布协议不变。GM transfer增加393216B，没有新增根参数或Host回读。

## 已完成的CPU检查

- 独立AIV[1024,8]→[8,1024]及取3行探针编译通过，仅说明接口可生成。
- 完整v1在Left空Tile初始化缺TileView、extract含row_major的类型重绑处
  失败，无设备运行。v2去掉wrapper已排除的prefetch分支，直接以真实
  primed Key作为初值；备用Key初值是别名，参与数学前必须经过预取。
- 完整v2符号编译通过：58动态轴、130根、单MIX/sync_start，不供应shape。
- 实际Right Query48KiB@0、系数12KiB@49152；Left WS48KiB@0，
  两Key8KiB@49152/57344。首QK Key临时在0，消费后才能被WS复用，
  后续两Key与WS不重叠。生成物路径/地址见ALLOCATION_REVIEW.json。
- AIV TLOAD→TTRANS→TEXTRACT→MTE2 ACK保持顺序，当前未专门调整
  ACK发射位置；源码先后不代表Vector与MTE2硬件必须完全串行。

## 真机验证

任务task_20261018_132551_345883327524：正式权重128K/B16/S6，原分task、
原正式SPMD和候选同进程；全部精确数值/完整state-cache/保护/图代际
通过后，各取5窗完整CSA L1，按原流程轮换次序。当前尚无最终设备结论。

prepare.py保留从正式动态源码制作候选的完整变换，source_reference与
source_candidate为完整动态源码，不调用静态shape排程生成器。修改仅
indexer_score_stages.py、binding.py；依赖和共享安装均未修改。

## 第一版设备结果：精确通过，性能大幅回退，未采用

任务task_20261018_132551_345883327524 exit0，完整精确/图/状态/保护/代际
全部通过。三方五窗完整L1：原分task874.56、原正式SPMD880.26、候选
1533.96μs，候选回退74.262%。全部样本保留在DEVICE_RESULT_V1.json。

当前不能从整层回退直接断言是AIC的M/N方向或AIV转置造成。只读真实
TTRANS实现确认[1024,8]满足向量快路径对齐条件（源stride8、目的stride1024、
中间stride1024），没有证据把它说成逐元素Scalar转置回退。下一步仅对该
候选的相同Score helper采一次PMU，定位大幅回退来自哪条pipeline，再决定
是否值得修改。正式算子保持原版，不合入该候选、不扩七档。

## 整行写回修正版（待设备结果）

硬件诊断见hardware_probe/PMU_SUMMARY.json：MTE1降到63.471μs，但
FIX升至876.487μs，AIV Vector仅108.386μs。修正版只将两次WS store
valid列6→16和内部GM stride8→16，完整写出已经计算的16列，每行64B；
前6列有效数学不变，AIV仍读取前8列。apply_aligned_store.py可在prepare.py
生成的首版上复现，拒绝重复应用。

aligned_cpu_v1完整符号编译PASS，130根、58动态轴、单MIX/sync_start。
AIV生成C++去注释后与首版逐字符相同，未混入新的向量优化；差异见
aligned_store_v2.diff和ALIGNED_CODEGEN_REVIEW.json。
设备任务task_20261018_134422_286630303，继续完整精确/图/保护检查及
三方五窗L1；当前仍在运行，不预记收益。首次失败版1533.96μs完整保留，
性能目标分母仍是同轮原多task，不用失败版作优化验收基线。

## 整行修正结果：全部精确通过，仍未超过正式版

任务exit0，完整精度/状态/图/保护/代际通过。五窗L1中位原分task880.58、
原正式SPMD863.32、整行写回候选1072.02μs，仍回退24.174%。
相比首版1533.96属于不同轮次，不能用差额冒充对正式版本的收益；本轮候选
仍明显落后，因此不采用。保存全部样本，未重测七档。

后续仅做Acc输出集合的CPU可行性检查：WS每片64×16结果4KiB，8片
占32KiB，与QK的96KiB可在容量上共存；尝试保留每片相同矩阵数学，
在Acc中汇集后减少FIX写回次数。当前只是接口/别名探针，不代表完整
CSA可用、数值通过或性能改善。

## WS输出集合：小探针通过，完整编译遇资源复用限制

小探针实际生成8次TMATMUL、1次TSTORE、无TINSERT，子片落在同一
32KiB Acc的0/4096/…/28672偏移。完整pooled_cpu_v1却在Acc分配失败：
131072+32768=163840B>131072B。MemoryReuse转储中，QK实际视图
仅98304B，但它复用的公共slot被前面Q_B最大tile扩为131072B；WS pool
独立分配32768B。理论同时活跃96+32=128KiB与实际规划不同。

这需要进一步最小复现及生命周期/别名分析，不能直接放宽硬件上限或手工
覆盖地址。当前pool完整CSA没有成功编译、更没有新设备结果。修正前
已完整验证的整行版保存为source_aligned_v2，可用run_device.sh追加
`--spmd-source <本目录/source_aligned_v2>`复现；source_candidate现为
未通过完整分配的pool实验。
