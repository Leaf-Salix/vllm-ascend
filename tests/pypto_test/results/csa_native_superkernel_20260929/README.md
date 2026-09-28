# Native npugraph_ex / superkernel 对照

按用户要求使用显式 `torch.compile(module, backend="npugraph_ex")`，
不以 vLLM 的 support_torch_compile 包装替代这个入口。两侧 static_kernel 均开启，
同 CANN 9.2、mode2、det0、正式 layer4 权重和合成独立历史。
固定 fullgraph=True，禁止 graph break 或编译失败后静默回到 eager。

先只测128K/B16、8K/B24；长档 off→on，短档 on→off；5次预热、20次图外事件计时，
保留均值/P95/max、独立 PyTorch profile、八类输出状态和保护区结果。
不把新入口与此前 vLLM 包装入口直接混为同一基线。
新入口走 npugraph_ex 自身 passes，不会因为 EngineArgs 中有 fuse_norm_quant 就自动执行
vLLM Ascend 的 FX pass manager；报告明确标记这个差异，不把旧模板融合配置冒充新入口生效证据。

本实验使用 force_eager=False，由 npugraph_ex 管理捕获和重放，不嵌套手工外层图。
superkernel 两个必要步骤都记录：静态编译实际收到 super_kernel_optimize=True，
以及后端实际调用 NPUGraph.super_kernel_optimize() 成功。profile 用于检查实际图变化。
开关开启不等于已证明 kernel 融合，更不等于已证明性能收益。

Native 的 vllm_ascend.utils.npu_stream_switch 只是 torch.npu.stream 包装，
DSA 沿用 record_event/wait_event/wait_stream 显式跨流依赖，符合
[npugraph_ex 官方多流用法](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/multi_stream.md)。
保留这些同步和临时 Tensor 生命周期，不改成 GE 的同名接口。
DSA 是自定义算子边界，流切换在图捕获时实际执行；不能把 FX 没展开内部流标成多流失效，
也不能只看配置标成 overlap 生效，必须结合实际 profile 的 stream 和执行区间。
共享专家多流配置保留，但单层 CSA 不含 MoE，不宣称验收了共享专家 overlap。
GitCode 通过 `curl --noproxy '*'` 直连读取，多流页明确同一个 event 不可跨 graph break；
本实验 fullgraph=True，失败直接报错。superkernel 文档页当前返回访问频次限制，未取得正文；
开关和图优化的执行位置以本环境安装的 npugraph_ex 源码核实，不冒称已读到该页内容。

prepare.py 从已冻结源码建立独立副本，仅修改测试 runner 和图优化钩子，
不改当前七档任务或生产代码。源路径写入 source.json；入队后不再编辑该副本。
有收益且输出检查通过再纳入 Native 主配置，并重取受影响的对比基线。
用户明确限定：本轮长短代表档A/B若没有明确收益，保留关闭状态并结束superkernel方向，
后续不再调参、扩档或重复测试该方向；测量噪声范围内的微小差异不作为采用依据。
需要区分入口失败与有效负收益，失败结果不能冒充完成的性能对照。

2026-09-29 07:48 已正常 auto 单卡入队，任务见[task.txt](task.txt)。
HCA先运行，不终止或绕过其设备分配；当前尚无本实验设备结论。
