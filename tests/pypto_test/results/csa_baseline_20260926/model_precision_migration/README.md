# 精度版迁移后的整模型看护

`comparison.json`：16 rank、B16/H8192、每请求 96 token，24576 个 token 和所有 DSpark
计数一致。固定正式权重、mode=2、precision、atomic=0、HCCL_DETERMINISTIC=true、EPLB 关闭。
任务 task_20260926_214325_324503619485，源码 d5ba31dc。

该轮旧 CLI 在外层请求 Native level=1，但未在实际 worker 读取级别；新进程 CPU 复现
确认此设置不会由 spawn 自动继承。该结果证明本轮 token/DSpark 一致，不能当作 worker
level=1 生效证据。测试入口后续已修复，当前 run.sh 重跑会使用实际 worker 配置。

`run.sh` 必须通过 task-submit 分配 16 卡，复用现有 h8192_bank，不重新生成缓存。
本目录结果不作为性能或全部逐元素误差验收。
