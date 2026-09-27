# 30f2b228 七档对照

14份真实EP16模型rank0 PyTorch JSON、7份单卡第二CSA层PTO泳道。来源与范围见manifest.json。
TP1、DP=EP16、出5验6、mode2、det0、EPLB关，PTO原始cache物理页/atomic0。
正式结果用RESULTS.md的预热后10步forward；模型profile独立3步，单层DFX使用真实权重和合成历史。
七档均值更快，但128K/B8、B16的P95较高；ARRIVAL.md保留异常，尾部问题未关闭。
