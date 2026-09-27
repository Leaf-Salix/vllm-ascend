# 71153bb3 / atomic0 七档 profile

14 份真实 EP16 模型 rank0 PyTorch JSON，加 7 份单卡第二个 CSA 层的 PTO DFX JSON。
双方 TP1、DP=EP16、出5验6、mode2、det0、EPLB关；完整来源见 manifest.json。

正式性能以 RESULTS.md 中预热后十步无 profiler forward 为准，全部 rank 样本见 forward.json。
模型 JSON 是独立三步 profile，其他十五个 rank 的原始数据保留在来源目录。
PTO DFX 使用第4层真实权重和人工历史，metadata reuse、graph replay，一次根调用；它与模型输入和计时轮次不同，不能直接拿 DFX 首尾代替正式整网成绩。
