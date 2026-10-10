# 原 task 局部 Set/Wait 的单卡证据

本目录对应执行方案第5.8节。不是 AscendC 静态合组或旧整体函数属性接口。

- [完整结论与逐项验收](../DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#38-过程-036局部语句路径单卡功能通过csa性能单独记录)。
- [结果与全部60个CSA计时样本](results.json)。
- [原NZ O-A task的两行改动](decode_o_proj.patch)，基线为vLLM-Ascend `fc6d201b`。

私有根目录：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/`。
PyPTO `b32ee10ff`、PTOAS `75615d31`、Simpler `a1a25dc33`、私有CSA `46597692`。
PTOAS独立CLI构建补丁为`1696b3e6`；不更改公共安装。

原始文件均在该根目录的`artifacts/`：

- `local_boundary_device_v1.xml`：三个设备用例结果。
- `local_boundary_final_v1/results.json`：同一二进制的动态B/C选择、无后继和改输入重复执行。
- `local_boundary_final_v1/local_{aiv,aic}/ready_results.json`：原始物理核号与设备tick，包含前驱返回、真实完成和后继Scalar前缀。
- `csa_local_boundaries_device_v1/results.json`：逐项状态/metadata/保护区结果及完整配置。
- `csa_local_boundaries_compile_v2/{baseline,local}`：预编译的完整CSA，两侧同为73个kernel。
- `run_local_task_boundaries_v1.sh`：本次单卡队列入口，先简单PyPTO，再完整CSA。

源码用例在私有PyPTO的`tests/st/runtime/scheduling/test_task_boundary_statements.py`
和`test_local_task_boundary_overlap.py`。只重跑需要核实的项目；硬件运行必须使用
`task-submit --device auto --device-num 1`，将队列给出的`TASK_DEVICE`传给pytest或CSA脚本。
重新采集时使用新的产物目录，保留本轮原始证据。

验收边界：3个基础设备用例通过，观察到同核Vector/Cube与后继Scalar重叠；
完整CSA仅验证单卡128K/B16、第二层真实权重，普通/局部语句路径逐bit一致。
30次交替单次重放的均值为967.64/970.21 us，候选慢0.27%，不宣称稳定加速。
不扩展为所有CSA task、其他档位、HBG、16卡或自动选择插入点的验证。
