# CSA roofline：Cube、Vector与内存

最新报告：[FULL_ROOFLINE.md](FULL_ROOFLINE.md)。
七档数据：[FULL_ROOFLINE.csv](FULL_ROOFLINE.csv)、
[完整资源摘要](FULL_ROOFLINE.json)。

本次补齐了初版遗漏的Vector：26个业务计算分组，按48个AIV实际分工计算；
同时以独立循环核对15个Cube阶段，并补入24个AIC的片上端口及冷热HBM模型。
只复算已有数据，没有新跑设备测试、改变算子或push。

这是一份有明确假设的峰值资源模型。它用于判断算力、带宽及数据流还有多少
乐观余量，不代表已实现或保证可达的完整时延；L2实际聚合带宽、Scalar、
指令头和完整依赖延迟没有伪装成已测数据。

复算命令：

```bash
python tests/pypto_test/csa_five_way_latest_20261020/theory_roofline/full_roofline.py
```

`INITIAL_CUBE_ONLY.md`保留遗漏Vector时的初版及纠正过程。
相邻[静态性能差异审计](../static_history_audit/README.md)已收尾，本轮不重复测试。
