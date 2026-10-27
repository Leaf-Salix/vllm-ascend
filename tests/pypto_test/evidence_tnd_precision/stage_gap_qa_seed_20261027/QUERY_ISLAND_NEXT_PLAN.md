# 下一个较大调度候选：保留Native数学的Query岛

## 已有实测约束

8K/B32新SDK的我方Indexer Q投影约359μs启动，早于主QB约374μs，已有重叠；不能从Python书写顺序声称串行。主Query后处理结束约541μs，Score约553μs开始。前段约293μs的累计参考差距不是全部可由fusion消除。

## 可借鉴的nalinaly调度

参考1685版本cube_schedule.py、vector_schedule.py、projection_overlap_stages.py：QA后AIC先Indexer压缩/Query投影，AIV先Indexer RoPE；再让少量Hadamard核和主QB核并行。

若只融合Indexer投影→RoPE→Hadamard→quant，24核MIX可能挡住现有同时进行的主QB，适合作为隔离诊断，不能先认定有收益。更有意义的闭包是QA→精确QR，再AIC先IndexerQ、后Hadamard/QB分核；AIV先Indexer RoPE/量化，再主Q反量化/Norm/RoPE。第一版只变‘持久岛’的任务边界；分核(24串行/4+20)与owner等待随后各做独立A/B，避免一次改变多因素后无法归因。

## 必须保留

- QA N32、dense64/tail16、95/126/168 packet与完整1024列的真实owner集合。
- QR active=min(ceilT/8,16)，精确sqrt修正、Native BF16和INT8链。
- 我方主Q完整tile/tail与Indexer都先合并actScale*weightScale，再乘accFP32；Native对应边界及BF16 round必须保留。nalinaly主Q源码/文档的(accFP32*actScale)*weightScale不同，不能照抄覆盖我方边界。
- Hadamard现有M/K顺序、BF16→乘归一化→BF16→scaleFP16边界；首轮不改M256 tile。
- 私有QR/投影/Had缓冲留Query子scope；Q与Indexer量化Out在父；与Score arena为兄弟scope。重新检查128K小ring各depth。

## 依赖和性能否决条件

不在24核MIX内等待岛外cache/压缩任务；否则可能占满producer所需资源。禁止manual_dep跳过同task结果。外部Score默认只能在整个Query岛退出后消费结果，即使内部phase已发布也不能假称提前对外可见。若Score入场晚于控制，或完整event/L1未稳定更快，否决该候选，不为了融合继续扩大。

先CPU实生成检查AST、单写owner、MTE3/FIX排空后publish、wait后业务cache获取、scope/dependency；性能及反向重复通过后Native A/B8/ABA、尾部/混长，最后8K/128K整模型token。当前只是经过审查的方案，尚未实现或提交NPU，不计入已获得优化。
