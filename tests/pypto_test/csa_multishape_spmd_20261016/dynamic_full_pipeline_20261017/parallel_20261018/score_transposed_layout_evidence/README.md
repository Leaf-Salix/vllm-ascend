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
