# 同编译配置的 Native / PTO 单层对照

对齐 `run_dp_template.sh` 的 npugraph_ex、static_kernel、fuse_norm_quant 等配置。
使用 d8627207 的冻结整包基线，包含 stream2048 与 UB 中间根，**不包含**尚未验证的长档小 batch S6 候选。
两侧相同 CANN 9.2、NZ mode2、det0；PTO atomic0、ring=[256,128,256,32]MiB / task_window4096。

Native 使用可追踪的 HC_pre + norm + attention + HC_post，保留真实 dsa_forward 边界；
PTO 使用生产 `dsv4_csa_forward` 自定义算子边界，强制检查实际进入 PTO，禁止静默回退。
两者经过相同 `support_torch_compile` 包装。Native 必须实际静态编译并安装算子包；
PTO 内部由 PyPTO 编译，其不透明自定义算子可能没有可交给 CANN 静态编译的描述符，
因此分别记录 wrapper、static_compile 调用和 PTO 调用证据，不把配置开启等同于发生静态编译。
每个进程新建独立空 static_kernel 的可写 OPP，避免前一个档位的静态包影响后一个。

使用第二个 CSA 层 layer4 的正式权重和合成独立历史，逐物理行不同的 Indexer scale；
PTO 复用本 step 已生成的 compact metadata，与第二层口径一致；Native 保留自己的 metadata 调用。
图外设备事件计时，5 次预热 / 20 次重放，另采一次 PyTorch profile；记录均值 / P95 / max、
状态误差、Top-K 合法性与保护区。PTO 与自身 eager 必须逐 bit 一致，Native det0 差异单列。

这是 attention 半层设备区间，未包含 MoE、Worker CPU 绑定和 EP16 通信效果，
不能作为整模型 forward、逐 token 或 DSpark 接受统计的验收。

单个任务依次测 PTO 长档、Native 长档、Native 短档、PTO 短档，共用队列分配的一张卡。
首个 PTO 128K/B16 用于验证真实编译入口，失败立即停止，不重复已经成功的计时。
未完成前不引用旧 PTO 手工图结果与新 Native 编译结果计算收益。

入口：[compiled_case.py](compiled_case.py)、[run_case.sh](run_case.sh)、[prepare_opp.py](prepare_opp.py)。
