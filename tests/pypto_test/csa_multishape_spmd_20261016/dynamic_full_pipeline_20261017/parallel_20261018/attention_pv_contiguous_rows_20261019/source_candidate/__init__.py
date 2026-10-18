# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 Flash CSA 动态单 SPMD：目录总览与源码阅读顺序。

正式数学入口必须保留 pl.dynamic 维度、按实际输入计算的核内范围以及
覆盖区间的分块策略。entry.py现已接通完整Cube Score路径的源级动态入口，
program/kernel完整CPU编译通过；当前七档固定S6的整链设备精度、图与保护检查
已通过，性能均未达到相对原分task降低30%的目标。长/短历史各一档另通过
同址metadata变化与整请求padding重放，mixed-query、空rank及连续状态仍待
验证。不回退到历史静态展开器，也尚未切换真实服务路由。

已撤回的形状实值化生成链完整保存在 tests/pypto_test 的历史目录。
整体思路：保留原 CSA 的数学阶段和必要依赖，将 task 间调度改为一个
SPMD 内的分工和通知。当前源码数据流如下，设备验收范围另见优化日志：

    Native hidden[T,4,4096] → HC 前处理 → x_norm[T,4096]
      x_norm → QA/QR → 主 query 投影/RoPE
               QR   → Indexer query/Hadamard/量化
      x_norm → Indexer head 权重
      x_norm → 原始 KV 投影/RMS/RoPE → Native KV cache
      x_norm → 主压缩投影/池化/RMS/RoPE → Native 压缩 cache
      x_norm → Indexer 压缩投影/池化/norm/Hadamard/量化 → Native key/scale

    Indexer query + head 权重 + key/scale → score/Top-K
    主 query + Top-K + 两种 KV cache → sparse attention
    attention 结果 → O 投影 → HC 后处理

两条压缩分支各自直接读取、更新 Native state。HC 的 post/comb 和原始
hidden 保留到 HC 后处理；上图未画出这三条跨阶段长依赖，不能提前复用其
存储。当前所有阶段已接通；继续按完整测量优化区间策略并补动态场景边界。

先理解三个边界，再阅读具体数学
    1. T 是输入 Tensor 的实际 token 行数，B 是请求 metadata 的请求数；
       Q=query_start_loc、L=seq_lens 均是已有 device Tensor。每请求有效
       query 区间为 [Q[r],Q[r+1])，历史起点 H=L[r]-(Q[r+1]-Q[r])。
       T 可以包含 dummy，活动性还须检查 Q/L；不能用 token//固定S 反推
       请求，也不能把形状容量当成有效工作量。每请求本轮长度可以不同。
    2. pad8/pad16/pad128 等表示中间工作区容量；真实读写边界仍由 T、Q/L
       和 position/slot 决定。C4 紧凑行数由设备扫描得到：每请求贡献
       floor(L/4)-floor(H/4)，不能假定每请求恰好输出两行。主/Indexer
       metadata 及各自工作区分别传递，不因偶然 shape 相同而合并合同。
    3. AIC 负责投影/Hadamard等矩阵计算；AIV 负责归约、归一化、RoPE、
       量化、排序及 cache/state 写回。helper 的 worker 是外层传入的
       逻辑编号；range/pipeline 只描述本核工作，inline 不会自动建立
       跨核依赖。外层统一安排 spmd/split_aiv，并计算真正需要等待的核。

同步和工作区应沿着数据流阅读
    投影 partial、量化 scale、压缩 pooled 等 GM 中间量有明确的生产者
    和消费者。生产者完成写回并发布，消费者等待后维护缓存再读取；
    Scalar 信号可见不等于所有业务数据自动可见。普通 phase 完成通知
    没有消费 ACK，不能直接代替 QK/PV 等环形缓冲的 READY/ACK 握手。
    压缩 pool 必须全部读完共享历史后才能提交 state，不能随每核进度
    提前覆盖其他核尚未读取的历史。单独的阶段编译不能证明这些组合依赖。
    所有持久 GM 工作区由外层分配/绑定，函数头部注明容量与复用边界；
    核内返回的 UB Tile 只能同核消费。每份信号工作区须固定阶段布局与
    参与核映射，跨次调用使用代际，首次清零及图/流独占由外层保证。

当前源码分工：
    1. request_geometry.py：读取 Native TND 的原始 query 前缀和及 KV 长度，
       在核内确定请求有效区间和 C4 压缩行；不建立 Host 或逐 token 映射。
    2. hc_stages.py：HC 加宽/RMS、投影、四分片归约、门值/Sinkhorn、混合归一化。
    3. qkv_stages.py：Q_A 与 QR，保留实际 T、矩阵 tile 与原量化舍入链。
    4. query_projection_stages.py：Q_B 与 query 的反量化/RMS/RoPE。
    5. kv_stages.py：归一化输入的 KV 投影→RMS→NOPE/RoPE；AIC 写 pad16
       FP32，AIV 读完整行并输出 BF16[T,512]，不在这里写 cache。
    6. cache_stages.py：读取 Native device slots，将原始 KV 按行写入 Native
       页存储的二维别名；负责写回分工，不承担上游数学或请求生命周期管理。
    7. metadata_stages.py：从 request_geometry 的请求边界生成 C4 紧凑偏移/实际行数，
       压缩消费者在 AIV 内直接读取 Native 紧凑 RoPE，避免展开为 T 行。
    8. indexer_query_stages.py：Indexer query 投影/反量化/
       RoPE/Hadamard/量化及 head 权重分支；不是完整 Indexer QK/Top-K。
    9. indexer_topk_stages.py：叶内排序与多根Top512归并，保留原pair编码和tie顺序；
       有效候选数不等于arena容量，最终请求映射和Top-K发布见indexer_publish_stages。
   10. compressor_projection_stages.py：主/Indexer 的KV/score双投影，按真实T分配M/N块。
   11. compressor_pool_stages.py：Q/L驱动八位置在线池化；全部历史读取结束后
       单独提交Native state，保留共享历史的读完依赖。
   12. compressor_output_stages.py：主压缩RMS/RoPE及Native BF16 cache写回。
   13. indexer_compressor_output_stages.py：真实紧凑行RMS/RoPE与Hadamard；
       不按固定S预留每请求两行，实际C来自device计数，pad只表示容量。
   14. indexer_cache_stages.py：紧凑key量化/写入，再单写者提交64B区域内的scale。
   15. phase_sync.py → signals.py：单写者阶段信号、调用代际、生产者发布和
       消费者等待；显式区分信号可见性与业务数据可读。
   16. wait_codegen.py：可选的后端等待循环变换；默认不自动启用。
   17. query_source.py：原多 task 数学的隔离源码准备工具，不是融合入口。
   18. indexer_group_stages.py：device Q/L前缀、真实query分组、叶计划和head系数。
   19. indexer_score_stages.py：AIC/双AIV成对Score流水，保留动态尾query与槽ACK。
   20. indexer_publish_stages.py：按真实query可见范围合并叶根、发布TopK和dummy行。
   21. attention_plan_stages.py：TopK后的有效块计划与inverse RoPE符号；不能跨层预算。
   22. attention_stream_stages.py：AIC QK/PV与双AIV采KV/softmax/归并的三槽流水。
   23. output_projection_a_stages.py、output_projection_b_stages.py：输出投影、量化和HCpost。
   24. runtime_abi.py：动态容量轴、60个阶段槽及24 AIC/48 AIV映射。
   25. cube_schedule.py、vector_schedule.py：完整角色编排；公共metadata由前段空闲
       AIV提前计算，真正读取计划时才等，不增加外部meta任务。
   26. entry.py：唯一pl.jit入口decode_csa_spmd，内部一次spmd与一个MIX cluster。
   27. policy.py、binding.py：区间策略、Native无损别名和捕获前工作区；当前首轮
       整链使用NZ2，压缩历史<2048的Vector Score策略仍待迁移。Host只用已有
       descriptor选数学策略；所有逐请求有效范围继续由device Q/L计算。

数学 helper 接收显式 worker 编号，由外层统一指定 AIC/AIV 工作量和同步。
T/B/容量轴的具体合同在各文件头部；只变更容量不能推断所有元素有效。
阶段 helper 本身不创建 task、没有独立调度完成语义，完整组合由两种schedule
角色建立。CPU编译及组合依赖模型只能证明相应层次的合同；整链设备验证
目前覆盖上述七档和两档动态重放，不能外推为全部动态边界或真实服务完成。
性能采用完整L1首尾；当前实测仍使用测试编译hook的100tick轮询退避，正式
binding尚未默认启用该扩展，不能把测试结果直接当作服务路由性能。
过程与验证边界统一记入上级 DSV4_CSA_SPMD_OPTIMIZATION_LOG.md。

导入本包不初始化 NPU，不加载 PyPTO 编译器或 torch。
"""

__all__ = []
