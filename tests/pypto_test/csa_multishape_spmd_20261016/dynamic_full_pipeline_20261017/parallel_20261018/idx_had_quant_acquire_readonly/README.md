# HAD → Quant 的整 DCache 获取：只读定界

结论是**现有本地证据不足以证明可删**。能确认 Quant 的业务 payload 只有 DMA/TLOAD 读取，未见通过 Scalar 解引用 HAD 数据；这仅排除了“消费者 Scalar 读取旧 HAD payload”这一直接用途，不等于整段获取或其中一条 ENTIRE DCCI 已被证明冗余。当前不提出删除实现，不编译、不上卡、不改协议。

源码参照 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/idx_had_owner_evidence/source_reference`（66882634），并读取同轮 long B16 reference 的实际 `build_output/decode_csa_spmd_kyg6wy7u/kernels/aiv/vector_schedule.cpp`。只读范围如下。

- `indexer_query_stages.py:201` 的 HAD 一次生成一个 token 的 64×128 FP32；AIC 按本核所有 token 循环结束后发布。Quant 在 `:221` 开始，唯一外部 payload load 是该 token 的 `qh_acc_gm[o0:o0+64,0:128]`，每 token 32768 B；后续是 UB 上的量化及 INT8/scale 写回。没有 Scalar payload load。
- 实际 AIV C++ `:45209` 等待当前 HAD generation，`:45215–45219` 为 PIPE_ALL、DSB、ENTIRE DCCI、PIPE_ALL、DSB；`:45222–45228` 还有 MTE3/V→MTE2 局部事件及 PIPE_ALL；`:45263` 首次 TLOAD，然后 MTE2→V event，`:45270` 获取后才做 TMULS。不能因一个业务 GM load 是 DMA 就顺带删除本核 drain 或 UB 回收事件。
- 本地 `pto-isa-src/include/pto/npu/a2a3/TLoad.hpp:23` 将四字节 Vec 数据搬运落到 `copy_gm_to_ubuf_align_b32`。这是 GM→UB 搬运接口，不能称为 Scalar DCache 读，也不能称为每次直达 HBM；`docs/isa/TLOAD.md` 单独描述 A2/A3 的 L2 allocate/non-allocate 路径。
- 本地 `ptoas-src/lib/PTO/Transforms/PTOToEmitC/ScalarMisc/ScalarMiscSupport.cpp:428` 将整 GM cacheinvalid 生成为两参数 `dcci((__gm__ void*)0, ENTIRE_DATA_CACHE)`；`ScalarMiscOps.cpp:320` 是独立 CMO op 的 lowering。CANN 本机 `aarch64-linux/asc/impl/basic_api/kernel_reg.h:132/148` 与 `impl/c_api/instr_impl/npu_arch_2201/cache_ctrl_impl/asc_dcci_impl.h:29` 可确认 DCCI 指令及 ALL/UB/OUT 等接口，但这些 wrapper 本身没有给出足以证明本次两参数 DCCI 可被 DMA acquire 完全替代的内存排序合同。未拿 A5 的接口说明代替 A3。

四类职责必须分开。信号可见性由每次检查的 SINGLE DCCI+volatile 读和发布侧维护承担；无 Scalar HAD load 不允许删除信号维护。GM payload 可见性仍依赖 AIC 完成 FIX 写回、原 release/notify 和消费者确认当前 generation；不能以“信号读到了”替代未证明的 DMA 排序合同。本核未完流水与 UB 地址复用由 PIPE_ALL/DSB 和生成局部事件承担，不能靠 Scalar 缓存维护解释这些事件。跨调用复用则依赖顺序独占工作区和完整 invocation 收尾，当前 phase 标记不是消费者 ACK；本次没有改变这些前提，也没有把少做 cacheinvalid 当作允许提前覆盖的依据。

一个有利但不充分的事实是：在正式 `vector_schedule.py:694`，Had wait 紧跟 ORIGINAL_CACHE 的原 publish，该 publish 已有 drain/cache 维护；两者之间只进行信号轮询，不包含新的 Quant payload 读取。这让“只移除本次 ENTIRE DCCI、保留全部 fence/DSB/局部事件”看起来比删除整段获取更窄。但现有旧生成物不能证明移除 CMO SyncOp 后的编译器排序仍相同；已读本地 A3 wrapper 也不足以补齐该证明。因此本报告不将它列成已安全候选，也不把不确定性写成必然存在硬件错误。

本次没有重做已知 release_sync_v1 泛化删 fence 负例，没有运行新的模型或测试。后续若继续这个方向，缺的首先是精确 A3 acquire/CMO 语义与该窄变更的生成排序证据；当前有限审计到此收口。

