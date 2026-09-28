# 554b3bca七档profile及泳道

01–14为真实EP16 rank0独立三步PyTorch JSON；15–21为对应七档单卡layer4合成历史PTO泳道。
二者采集独立，单卡DFX不等于真实模型某一步某层。DFX固定保留窗口0供下载，全部窗口见manifest与单卡报告，未按耗时筛选。

- [01_128K_B4_native_FullModel_Rank0.json](01_128K_B4_native_FullModel_Rank0.json)
- [02_128K_B4_pto_FullModel_Rank0.json](02_128K_B4_pto_FullModel_Rank0.json)
- [03_128K_B8_native_FullModel_Rank0.json](03_128K_B8_native_FullModel_Rank0.json)
- [04_128K_B8_pto_FullModel_Rank0.json](04_128K_B8_pto_FullModel_Rank0.json)
- [05_128K_B16_native_FullModel_Rank0.json](05_128K_B16_native_FullModel_Rank0.json)
- [06_128K_B16_pto_FullModel_Rank0.json](06_128K_B16_pto_FullModel_Rank0.json)
- [07_8K_B16_native_FullModel_Rank0.json](07_8K_B16_native_FullModel_Rank0.json)
- [08_8K_B16_pto_FullModel_Rank0.json](08_8K_B16_pto_FullModel_Rank0.json)
- [09_8K_B24_native_FullModel_Rank0.json](09_8K_B24_native_FullModel_Rank0.json)
- [10_8K_B24_pto_FullModel_Rank0.json](10_8K_B24_pto_FullModel_Rank0.json)
- [11_8K_B32_native_FullModel_Rank0.json](11_8K_B32_native_FullModel_Rank0.json)
- [12_8K_B32_pto_FullModel_Rank0.json](12_8K_B32_pto_FullModel_Rank0.json)
- [13_8K_B40_native_FullModel_Rank0.json](13_8K_B40_native_FullModel_Rank0.json)
- [14_8K_B40_pto_FullModel_Rank0.json](14_8K_B40_pto_FullModel_Rank0.json)
- [15_128K_B4_PTO_SingleCSA_SyntheticHistory.json](15_128K_B4_PTO_SingleCSA_SyntheticHistory.json)
- [16_128K_B8_PTO_SingleCSA_SyntheticHistory.json](16_128K_B8_PTO_SingleCSA_SyntheticHistory.json)
- [17_128K_B16_PTO_SingleCSA_SyntheticHistory.json](17_128K_B16_PTO_SingleCSA_SyntheticHistory.json)
- [18_8K_B16_PTO_SingleCSA_SyntheticHistory.json](18_8K_B16_PTO_SingleCSA_SyntheticHistory.json)
- [19_8K_B24_PTO_SingleCSA_SyntheticHistory.json](19_8K_B24_PTO_SingleCSA_SyntheticHistory.json)
- [20_8K_B32_PTO_SingleCSA_SyntheticHistory.json](20_8K_B32_PTO_SingleCSA_SyntheticHistory.json)
- [21_8K_B40_PTO_SingleCSA_SyntheticHistory.json](21_8K_B40_PTO_SingleCSA_SyntheticHistory.json)

[来源及全部窗口](manifest.json)、[单卡CSA](../LAYER.md)、[模型forward](../model/RESULTS.md)、[模型CSA及分项](../model/MODEL_GAP.md)、[相邻CSA差异](../model/ADJACENT_CSA.md)。
