# OA两版B4：实际生成物只读对照

**没有发现两处阈值以外的生成布局、数学、局部事件或代码体积变化。** 两侧raw bin各仅1个字节不同。该结果不能解释异轮性能差异，也不改变小T候选不采用、不补测的结论。

只读两轮已完成的H131072/B4/S6/T24 candidate；角色由各自`source_matrix/result.json.kernel_configs.spmd_candidate`确定。未编译、上卡、改候选/依赖或计算校验摘要。

主PAR为`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018`。

| 版本 | 实际candidate build_output路径 |
| --- | --- |
| 原T≤96版 | oa_pair_manual_h131072_b4_v1/build_output/decode_csa_spmd_p88j021j |
| 新OB32且T≤32版 | oa_pair_small_t_h131072_b4_v1/build_output/decode_csa_spmd_yueosr70 |

两侧policy完全相同：G6/N64/1024、PREFETCH1、BALANCED/ROOTS1、OA128/OB32、QA_M16=true，输出组流水和HC行依赖开启。两侧codegen probe均47个wait替换位置、100tick backoff、无phase clock。实际T24满足两版分支。

## 完整对照结果

| 项目 | AIC cube_schedule | AIV vector_schedule |
| --- | ---: | ---: |
| raw bin大小，两侧各自相同 | 45508 B | 147016 B |
| 直接逐字节比较的不同字节数 | 1 | 1 |
| 不同字节的零起始偏移 | 38688 | 134920 |
| 生成Tile分配条目数，两侧相同 | 585 | 1812 |
| 分配address/space/dtype/shape/size及顺序 | 全同 | 全同 |
| set_flag/wait_flag/pipe_barrier条目及event id顺序 | 1730条，全同 | 3090条，全同 |
| 其中PIPE_ALL静态位置 | 20，两侧相同 | 203，两侧相同 |
| 归一化C++函数的实质差异 | 仅一处`<=96`变`<=32` | 仅一处`<=96`变`<=32` |

AIC全部空间最高地址终点保持Mat475136、Left65536、Right65536、Acc131072、Vec16384 B；AIV Vec终点保持163872 B。终点统计来自实际PTO分配，不是额外并发使用量。

完整PTO保留运算、常量值/类型、valid/layout、物理地址、控制流和同步，只消去source loc、SSA名及常量声明顺序后，恰好两条`arith.cmpi sle`上界96→32；没有其它差异。对应`whole_pto.diff`。

原始C++大面积文本变化来自常量96的声明位置移动，后续v编号随之移动。逐函数内联同值同类型常量并统一临时编号后，完整函数分别只剩一个阈值差异，见`aic_cpp.diff`、`aiv_cpp.diff`。没有借归一化删除数学、事件、地址或分支。实际源码快照33文件去docstring后、仅恢复cube/vector各一个谓词，全部AST一致；新增OB_ROWS条件在本次OB32 constexpr下消去。

raw bin路径为各build的`prebuilt/kernel_0.bin`、`prebuilt/kernel_1.bin`。只确认了精确长度及单字节差异，不重新反汇编，不把偏移擅自标为已解码ISA立即数字段。生成PTO/C++差异与raw比较相互支持“未发现额外布局/同步改变”，不宣称两二进制完全相同。

## 性能解释限制

原版B4同轮ref555.80→candidate527.14 μs，降低5.157%，5/5窗口更快；新版同轮ref546.38→candidate543.34 μs，降低0.556%，3/5。两组是不同作业，不由candidate绝对值的变化认定回退原因，也不能把效果变化归因于片上布局或同步——本次对照没有这些差异。没有当前硬件计数/频率/运行时取指地址证据，不继续猜测原因。

原五窗完整保留如下（μs），包括新版reference的626.46高值；未剔除或重算截尾统计。每个image只有一个字节不同，与阈值变化一致，但不能据此证明任何性能差的根因。

| 版本 / 角色 | 五窗原样 | 中位数 |
| --- | --- | ---: |
| 原版reference | 553.66 / 538.28 / 555.80 / 562.42 / 567.98 | 555.80 |
| 原版candidate | 516.58 / 528.48 / 511.96 / 527.14 / 532.18 | 527.14 |
| 新版reference | 546.38 / 536.22 / 541.98 / 626.46 / 562.38 | 546.38 |
| 新版candidate | 539.22 / 548.74 / 543.34 / 546.58 / 541.64 | 543.34 |

`audit_existing.py`仅使用标准库和已有只读PTO归一化函数，读取现成文件后写本目录diff及`evidence.json`；它不导入设备库或触发编译。`evidence.json`保存每条角色路径、33文件AST检查、资源/事件/字节对照和两轮原五窗数值。`LOCAL_ARTIFACTS.json`列本目录及输入大文件的路径和大小，可整体归档本目录；原产物不复制、不改写。
