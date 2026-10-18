# Score 与 Attention 跨请求交错：并行审查与实现门槛

2026-10-18。三路独立只读审查基于正式动态单 SPMD `66882634`，没有把旧静态 entry 作为优化对象，也没有新增性能测量。Score 服务诊断另见 `../score_service_clock_evidence/DEVICE_RESULTS.md`；该诊断不是本方案的收益证据。

## 当前决定

保留两套流水状态，尝试 AIC 执行当前 Score step 时让同组双 AIV 推进已完成 TopK/plan 的前波 Attention tick。它不同于旧的整 helper 排空后切换。先准备可维护的动态源码，尚不修改正式路径、不提交设备任务。

优先研究的内存路径是 **只把 AIC 内部 KV Mat 改为两槽，先做旧 PV，再做新 QK；所有 GM ring、mi/li 仍保持三槽**。它避开 boxed Tile 的异 dtype 别名限制。Score Query 保留 Mat、每次切回重新 TMOV 48 KiB，系数保留 Left。保守 Mat 496 KiB，额外计入初始化系数暂态为 508 KiB；这些只是现成 Tile 大小核对，尚无联合后端分配结果。新增搬运及损失的原 QK/PV 重叠都必须由完整 L1 实测结算。

## 三路结论

- [同步审查](sync/README.md)：给出有条件的无环顺序、真实 MTE1/MTE2 复用边界、各组独立 cursor、跨波 root→TopK→plan 的发布/获取、空 mask 与尾拍排空。
- [资源审查](resources/README.md)、[两槽替代](resources/ALTERNATIVE_TWO_L1_SLOTS.md)：原 helper 直接同驻留不成立；给出 Left 重装、Mat/UB 状态与表达前提。
- [状态与最小切分](state/README.md)：完整 Score step/Attention tick 才是暂停边界，至少涉及两个 schedule、两个计算体、TopK 与 M8 plan，以及波进度容量。

B16 的连续四个 group 并不自动铺满24个 Score 核：保留原 leaf-major owner 时只占12核。不能用回归点的总任务数替代动态分核证明，也不能默认混合 S 或尾 query 恰好8行对齐。分波必须读取真实 group/query 描述符，明确处理 M8 截断和无内部切点回退。

## 实现前仍须解决

1. 核对当前 JIT 能否用带 Tile 参数的 helper 保持跨步状态与同一 allocation；底层 IR 支持不代表 JIT 特化层支持。不用冻结巨型生成体规避可维护性要求。
2. 按波隔离的发布值/槽与容量：原整调用 phase generation 不可原值重复写来代表下一波；每核512B独占和真实业务 release/acquire 保持。
3. 两槽源级改法、联合内存规划、回边与硬件事件必须审查；局部容量及拓扑结论不是编译/精度通过。
4. 只有联合实现完成后，才按原程序做一次代表性 CPU lowering 和有理由的同轮设备对比。原七档逐档降低至少30%的目标不变，当前本方案没有性能数字。

`PROVENANCE.json` 保存三份原始审查的路径和大小。此目录的尝试均为动态方案设计，旧静态结果保持历史归档。

## Tile helper 的实际CPU复现

独立Python源码overlay已复现 `ParserTypeError: Parameter 'tile' missing type annotation`。输入Tensor保留动态行轴，incore内部加载固定数学Tile后传入inline并原样返回；失败发生在specialize解析，尚未codegen或设备执行。原始证据暂在独立工作区 `jit_tile_parameter_overlay_20261018/before_v1/`，修复仅处理Tile注解保留。不能将底层IR支持Tile函数误记为现有JIT已支持；尚未把新overlay用于CSA编译。

## 已准备的源码与接口证据

- [Score AIC step](../score_attention_cube_step_source/README.md)
- [Score AIV step与叶状态](../score_attention_vector_step_source/README.md)
- [Attention init/tick](../score_attention_tick_source/README.md)
- [device wave边界](../score_attention_wave_boundaries/README.md)
- [wave进度槽合同](wave_progress/README.md)
- [JIT最小补丁与CPU证据](../jit_tile_parameter_evidence/README.md)

普通Tile原样返回及实际full-valid Vec状态的多返回/循环传递已完成最小CPU编译；动态valid循环、同helper constexpr维度尚未解决，InOut只特化通过。联合CSA、Mat/Left helper、波间索引/信号接线尚待实现及验证，暂无新NPU性能数据。
