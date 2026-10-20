# 2026-10-20 新SDK精度与常驻后缀证据

完整CSA 8K/B16对Native SK0八边界逐位一致，耗时慢41.55%。
独立常驻O-proj/HC-post候选T18/T126/T240对冻结PTO后缀五边界逐位一致，但三档均性能回退。
候选没有接入正式CSA，本轮没有新的整模型token结果。

- `raw-results.tar.gz`：原始JSON、CPU门禁失败/通过、运行历史与失败启动日志；`RAW_SHA256.json`为包内文件SHA。
- `reproduction-tools.tar.gz`：实际harness、冻结基准绑定、独立候选、SDK预检、源身份收据和harness独立复核。
- `device-logs.tar.gz`：四项成功设备任务的原始日志及CPU编译日志。
- `REPRODUCTION.md`：逐项复现命令、环境与资格限制；`CHECKLIST.md`为本轮完成清单。
- `REPORT_RESULT_REVIEW.md`与`result-review.tar.gz`：独立结果复核及实际被审文件SHA。
- `SHA256.json`：本目录发布文件的SHA，不包含它自身。

正式运行源仍为48c58ce7c；新候选仅为实验。原始官方JSON保留原字段，用压缩包存储避免文档拼写工具改写证据。
