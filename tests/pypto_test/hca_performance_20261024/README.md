# HCA 八档三列性能复测，2026-10-24

依据 `vllm_ascend/ops/pypto/CSA_PERFORMANCE_TEST_SPEC.md`，测完整第4层
（`layer_index=3`）`HC_pre → input RMSNorm → HCA/C128 → HC_post`。
八个case和固定混长列表见 `cases.json`，种子1024，正式ModelSlim W8A8权重，
NZ2、BF16激活、TP1单卡。没有HCA SPMD列，也不以CSA数据填HCA表。

用户本轮明确区分两侧执行方式：

- Native：`torch.compile(npugraph_ex)`，静态编译、SuperKernel、NZ、ACLGraph全部启用；
  逐项核实实际调用成功、静态包安装、多stream及权重格式29，取30个设备全流首尾跨度的中位数。
- HCA-MPMD-AICPU：从真实服务的trace中定位完整PTO根调用，不捕获或重放ACLGraph。
  保留根调用内全部stream的AICPU、AICore、Event、DMA和等待。每窗新forward context，
  额外采到的服务前导compact producer及派发间隙单独记录，不能把服务层9～14ms改称AICPU耗时。
- HCA-MPMD-L1：同一服务调用的整层PTO全部incore首尾，不累加各task；
  L1之外的Native compact producer属于外部列。两种采集器独立运行，源码、输入、状态与调用入口相同。

该边界与CSA完整算子基准一致：Native compact metadata是PTO根入口的既有device Tensor输入；
PTO内的Q/L、slot、压缩有效块等推导仍全部纳入，未移动任何生产计算。
`pto_interval.py`逐窗要求唯一根op和已经完成的唯一Native前导，完整保留后续异步设备活动；
CPU标记只定位归属，时间仍由设备首尾计算。完整服务跨度保留在JSON中，不删原始慢窗口。

PTO每列10窗、每窗一次完整直接调用、每项预热3次。每次采样后在计时范围外检查
输出和全部cache/state对该实现eager结果逐bit一致及保护区未修改；保留所有样本，不挑最快。
Native按原已验证采集器执行，测试复位/填毒在图外，层内必要copy、依赖和空隙保留。
Native额外Event数据仅为原采集器的辅助记录，不用于本表。

百分比为 `100 × (median_native - median_hca_mpmd_l1) / median_hca_mpmd_l1`。
这是用户指定的不同测量边界之比，不是服务端到端吞吐提升。

## 版本和数值门禁

算子冻结在 `manifest.json` 的operator路径，生产HCA源码不改，仍为
`37bbd0f3c` 引入的K128/K512及Q_B/O_B区间策略。
公共SDK为 `csa-shared-20261024-group-dims`，使用公共bootstrap的`--shared-toolchain`。
原工具链历史记录不可改标为本轮实测，因此补采八档。

冻结正式控制组与本轮算子使用同一公共SDK，核对输出/cache/state逐bit一致；
控制组仅做数值校验，不重复测其L1性能。Native输出仍用既定atol/rtol=0.01。
旧有Native cache/state严格1e-5差异保留并单列，不能将其写成全部Native精度通过。
HCA没有Indexer，TopK/score不适用。此任务不修改数学、策略或编译器，不恢复性能优化目标。

## 复现和归档

`submit.py`提交所选case，经task-submit分卡，任务号和完整命令保存于`tasks.json`。
已提交key不重提。新attempt显式增加`--revision`，保留旧失败记录。

```bash
python tests/pypto_test/hca_performance_20261024/submit.py cpu
python tests/pypto_test/hca_performance_20261024/submit.py native C04 --revision 2
python tests/pypto_test/hca_performance_20261024/submit.py pto C04
```

Native初次提交在进入编译前因历史启动器缺少新配置字段`static_kernel`失败；
v2补入`static_kernel=True`，未关闭优化选项、未更改算子。完整失败日志与后续结果均归档。
v2随后遇到CANN把长绝对路径编码进shape_info文件名，超过文件系统NAME_MAX；
v3仅使用私有短编译工作目录，保持所有优化选项。实际路径写入每个结果目录的`compile_workdir.txt`。
大型trace、静态包和编译产物留在本地`runs/`，Git只保存小型报告及路径/大小清单，不做hash扫描。
后续本地提交使用详细中文说明，未经用户指示不push。
