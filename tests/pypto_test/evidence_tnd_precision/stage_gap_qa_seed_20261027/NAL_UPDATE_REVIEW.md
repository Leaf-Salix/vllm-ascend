# nalinaly 1685da69 更新与可借鉴范围

2026-10-27 GitHub `ls-remote` 发现公开分支从 `c346573ab3313fcc71cd58756acd129f34755439` 更新到 `1685da6914ae6196537d7f1a66e84e09a9c48ae5`，compare共4提交。保留compare JSON、23个变更源码/资格记录、叠加旧公开快照得到242文件的相关测试/运行快照；这不是整库历史clone。

## 新增实现

- WKV、Indexer Q_B、Indexer权重投影直接借用Native NZ存储；不修改公开层面的外K循环、量化或舍入表达式。
- WKV/Indexer BF16权重从旧转置ND[K,N]改为Native[N,K]，切片相应调整并b_trans=True；我方原本已是[N,K]+b_trans，因此无需改变方向。
- 新增初始化清零/同步测量：128K/B16全核清零+同步约9.26us，不能凭此解释我方数百us差距。该测量来自其公开报告，不是我方受控复测。
- NZ验收公开声明仅比较各自旧ND实现，不包含Native数值重测；8K/B32 SPMD 940.45→923.04us(-1.85%)，单轮记录不能当我方稳定收益。

## 我方单因素

先准备WKV NZ独立候选，保留原K256顺序表、64x32/same_order与narrow fallback、matmul初始化和BF16舍入；root/prefix/cube/inline的layout全部传递。候选只绑定原Native NZ权重，检查format29、指针和逻辑CPU值一致。

原精度基准仍使用原ND路径。候选构造时仍创建一份未用ND副本，因此当前只测运行期读取，不宣称节省加载转换或显存。源math函数体不变不能保证后端MMA分块不变；需要生成代码审核与获益后的Native A/B8。

第一版缺binding.json，在无设备预检失败；第二版补齐同一已验binding文件，不更改数值或门槛。
