# HC_post残差常驻UB

任务`task_20260929_101809_3278473829`于2026-09-29 10:18正常auto提交；结果待完成。
基线55b89ee2，独立于首PV候选，不把两个未验证改动叠加。

参考最新ops-transformer 28f40354：
`mhc/mhc_post/op_kernel/arch22/mhc_post_arch22.h`的`ComputeCopyOutAllX`，
在UB容量允许时用`USE_PERMANENT_X`保留四行残差，不为每个输出通道重读。
当前release Native的`HcPostKernelDSplit`也先整组读残差再计算。

当前PTO共享HC_post沿用pypto-lib2164563的循环组织：每个输出通道都读取四行残差。
上游残差原本FP32；接入Native后残差为BF16，因此当前PTO还重复做BF16→FP32转换。
生成码确认重复存在，未被CSE消除。候选把四行显式加载并转换为FP32，随后四个输出复用。
仍按post*x后in_h=0/1/2/3依次mul/add，输出BF16 RINT；没有采用上游AscendC的Axpy算术。
token分块、每worker最多4行、outer pipeline stage2和任务依赖均保持。

CPU首版被Tensor/Tile混用检查拒绝，尚未上卡；将x读取和y写出统一为显式Tile load/store后，
完整PTOAS/CCE/链接/load及两根依赖图通过。无需修改PyPTO/PTOAS/PTO-ISA。
最终生成码静态调用点（包含pipeline展开的三份正文）如下：

| 指令 | 基线 | 候选 |
| --- | ---: | ---: |
| TLOAD | 51 | 15 |
| TCVT | 63 | 27 |
| TMULS | 60 | 60 |
| TADD | 48 | 48 |
| TSTORE | 12 | 12 |

折合每token残差读取/转换16→4次；上述是生成代码证据，不是运行时指令计数或性能收益。
Native核内profile与PTO不能只按单worker均值作比较：例如B4 Native使用24个Vector block，
PTO为6个worker，每worker循环4个token。比较候选时保持分工不变，消除这项干扰。

只修改私有性能包的hc_post.py，不改公共精度实现或生产文件。
两侧CANN9.2/mode2/atomic0/det0、测试包装inplace=True，固定128K/B16与8K/B24。
5预热20次正式设备事件、各四窗DFX、八类完整状态零容差，核内/CSA/P95分别判断，长短8:2。
有明确核内收益且必要功能通过后才采用，不因内存访问减少就先宣称收益。

[准备](prepare.py)、[补丁](candidate.patch)、[来源](source.json)、[编译](compile_candidate.json)、
[生成码计数](lowering.json)、[两档入口](run.sh)、[复用收集器](collect.py)、[任务](task.txt)。
