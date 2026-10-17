# Attention 本核四 Scalar 几何缓存：既有 CPU 产物独立审阅

结论：**未发现阻塞设备验证的生成代码问题，ready。** 本次只读取主会话唯一 cpu_v1，未重新编译、未上卡、未改候选或正式源。CPU 通过不代表设备精度或性能收益。

候选为 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_scalar_geometry_evidence/source_candidate`，产物为同级 `cpu_v1/kernel`。参考是 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_h131072_b16_v1/build_output/decode_csa_spmd_jp7o4g2r`，按该作业 `source_matrix/result.json` 的 spmd_reference 角色核对，33 个参考源逐字相同；控制均为长 G6、panel64、Score1024、prefetch1、roots1、OA128/OB96、QA false、Attention query metadata false。ABI 为 130 根、58 动态轴，single_mix/sync_start/full_codegen 全通过，未提供样例 shape。

| 核侧 | 物理 Tile 分配 | 局部 set/wait/barrier | 分配上界（B） |
| --- | --- | --- | --- |
| AIC | 558 项地址/空间/类型/行列一致 | 1711 项平序列和 event ID 一致 | Mat 475136；Left/Right 65536；Acc 131072；Vec 16384 |
| AIV | 1812 项地址/空间/类型/行列一致 | 3088 项平序列和 event ID 一致 | Vec 163872 |

AIC 整个 `cube_schedule` 函数 PTO 在去 loc、按函数归一 SSA/同值同类型常量后相同；整个 C++ 数学函数去注释、归一变量名，并把参考已有退避 hook 调用还原为 TWAIT 后也相同。这里不是声称辅助头、完整混合文件或机器码逐字一致。

AIV 的源码循环有原四个数学 Tile 与四个新 Scalar，共八个源码循环值。物理 PTO 仍沿用原数学 Tile 分配/搬运；新增循环 iter_args 为三项 index 和一项 i32，没有新增 GM、UB Tile、phase 或根参数。`dynamic_csa_full.pto:12632–12654` 可见初值 tag/owner=-1、position/active=0，且新增查询 tag 比较位于 `tick<block_count` 和 `mask>0` 两层原 guard 内。

实际 `vector_schedule.cpp:57563–57567` 每次进入 stage 初始化四个缓存变量 `v10874..v10877`。前三项 int64_t，最后一项 int32_t；position 未经 FP32。`:57633` 比较当前 qk_t 与缓存 tag，cache miss 才进入原 request/Q/L/causal 检查，hit 直接复用三个值。几何计算分支不包含 FFTS 或局部事件；KV 零初始化仍在原有效块路径内，原 active guard 仍控制 gather，零/失活输出与 ring store 合同保持。

无效 mask 分支在 `:58443–58455` 透传四值；drain 分支在 `:57594–57597` 透传四值，不读取可能已越过 T 的 Q/L/position。有效 mask 的 active=0 仍把 tag 更新为当前 query，后续同 query 不重复扫描；其 owner=-1 不会越过原 active guard。因刷新条件是 tag 变化而非 sb==0，首个块无效、后续块有效时也能首次计算。两 lane 是各自的函数局部状态，不共享缓存。

四值只在 stage 初始化和 tick 回边写入（每变量两处 C++ 赋值）。`:59967–59970` 从当前 gather 的 after 值回传；较晚三拍的 PV/final publish/reset 仅重置原数学状态，没有写入这四项缓存。所有 query 空、末尾三拍以及下一次图重放的入口重置由此保持，不把上一 invocation 的 Q/L/position 沿用到下一次。

数学/地址/事件的审阅边界：源级逆 AST 审计已证明变化仅为四 Scalar 的缓存控制；本审阅独立确认实际物理分配、AIC 全函数、AIV 三层 guard/phi/末端 yield 和局部事件列表。AIV 完整 PTO/C++ 因新控制流而不同，本报告不称其全文等价，也不以事件平列表单独证明路径。原 SWA/compact gather 地址保护、mask/FFTS 谓词、softmax/PV 算术与最终 drain 没有发现额外变化。所有 metadata 在一次 stage 中只读仍是必要前提；按调用重置允许后续调用替换 metadata。

可复核文件是 `audit_generated.py`、`review.json`、`scalar_guard_excerpt.txt`、`INPUT_ARTIFACTS.json` 和 stdout/stderr。脚本只做文本解析；参考解析函数从既有脚本的 AST 取出，不运行其 main。原大 PTO/C++ 仅登记路径和大小，不复制、不 hash。CPU 不能确认 CCEC 最终寄存器压力、Scalar 栈 spill 或真实时延，这些不从“只新增四 Scalar”推断。

