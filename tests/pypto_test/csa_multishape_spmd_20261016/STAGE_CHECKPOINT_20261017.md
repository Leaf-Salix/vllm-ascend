# CSA 单 SPMD 阶段快照：2026-10-17 晚

本次集中补齐此前未提交的静态探索、动态迁移及完整链路验证过程；保留失败尝试，
不重新制造过去的commit顺序。后续按有结论的优化阶段及时做详细中文commit。

当前正式源码：`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`。
完整过程主记录：`vllm_ascend/ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md`。
当前测试入口与限制：[动态整链README](dynamic_full_pipeline_20261017/README.md)。
历史静态原型与实值化生成链仍在tests中，不能充当当前动态目标实现。

## 本次保留的阶段

1. 静态单SPMD的轮询/信号、分核、行依赖、Indexer/Attention流水等探索；
   采用与未采用项、编译失败、模型分析及同进程复核均保留原记录。
2. 将形状实值化生成链撤回tests，重新实现源级动态的HC、QKV、压缩/state、
   Indexer、Attention、输出投影与HCpost，接通一次24组MIX、130根Tensor。
3. 完整链路的metadata准备、页有效列修正、HC行依赖和输出组通知；
   最新七档固定S6完整精度/图/保护检查与同轮L1全部保留，性能均未达30%。
4. 128K/B16、8K/B32同址metadata六次变化及整请求padding图重放通过；
   不外推为mixed-query、空rank、请求生命周期、连续状态或真实服务接入。
5. Attention几何复用候选：128K/B16慢1.66%，8K/B32快1.33%，默认关闭。
   原始开关矩阵、候选前后源码、CPU与设备证据保留，不把局部收益当作达标。
6. 最新用户约束：历史KV长度与batch为主要策略维度，S6优先并补S5；
   允许S5/S6分别特化。参考历史静态SPMD和独立hc_pre，但保留动态请求边界。

## 产物保留边界

`LOCAL_ARTIFACTS_STAGE_20261017.json`列出本次过程文件及本地目录数量/大小。
权重、二进制、完整编译目录、原始大trace及大快照没有加入Git，原路径保留。
不做额外hash校验。历史JSON中的旧worktree路径保留执行来源，迁移映射见
`WORKSPACE_MIGRATION_20261017.json`。测试仍依赖隔离PyPTO/Simpler worktree，
本次没有提交或修改这些依赖仓，也没有改变共享安装。

性能计时为完整L1 incore首尾，测试编译hook仍有100tick失败轮询退避；
正式binding未默认启用该扩展，也尚未替换真实服务路由。
本次仅归档并核对已有结果，不重复已完成的七档真机测试。
