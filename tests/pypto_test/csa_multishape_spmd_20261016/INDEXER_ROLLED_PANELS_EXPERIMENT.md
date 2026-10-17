# 2026-10-17：Indexer N64面板循环化的CPU失败记录

本候选未执行设备，无精度或性能结果，未纳入现选，也未给device driver增加开关。
实现只位于`indexer_rolled_panels.py`，`build_variant`提供CPU实验参数
`indexer_rolled_panels=True`。没有修改PyPTO、PTOAS、ISA或Simpler。

## 尝试的目的

现选每个1024分数块展开16个N64面板。候选保留QK M384/N64/K128、
WS M16/N64/K384、FP16 FIX缩放/ReLU、原Native页读取及AIV Top-K。
先剥离QK0，再循环panel1..15，同时做当前QK和上一面板WS，最后排空WS15。
源码矩阵乘位点从32变4，运行时仍是16次QK与16次WS。
每块FFTS交接次数及叶末尾排空不变，没有减少数学工作或放松同步。

动机是减少重复代码和标量指令流；没有测到指令缓存miss，不能宣称已证明瓶颈在指令缓存。
`indexer_rolled_source_contract.json`只记录源码顺序；其页表达式自比较不是独立正确性测试。

## v1：Key池作为循环携带值

`indexer_rolled_prepare_v1`入口准备通过；`indexer_rolled_cpu_v1`完整代码生成失败：

```text
tile.move requires distinct source and destination addresses in Mat,
but both resolve to byte offset 180224
```

原设计携带Key L1池、Key Right和上一score L1。源码没有主动调用Mat→Mat move，
循环/分支合流lowering插入了移动，而内存规划将相关Key池放到同一地址。
错误在PyPTO后端生成阶段触发，未到设备；没有删除该校验或手工改地址。

## v2：移除不必要的Key池携带

下一Key的L1内容每次都在本迭代写入再提取，不需要跨迭代保留。
v2改成局部L1临时区，只携带Key Right和score L1。
`indexer_rolled_prepare_v2`通过，`indexer_rolled_cpu_v2`仍失败，PTOAS报告：

```text
'pto.tmov' op expects a supported tmov address-space pair for this target
```

实际PTO中合流和循环回边出现Right→Right的INT8 128×64 `pto.tmov`，
另有score的Mat→Mat移动。当前A3编译路径拒绝前者。相关IR保存在
`indexer_rolled_v2_unsupported_moves.pto.txt`，完整产物仍在本地CPU目录。
这不能仅归咎于PTOAS：从源循环携带到插入移动的变换也需要纳入分析。
没有给上游提交未经最小化的新issue，没有将此问题说成现选版本的设备错误。

## 后续可以考虑的方向

可以进一步研究每轮处理两个固定面板的循环，在一轮内完成两份Key和score的消费，
避开Right Tile跨循环携带。这样会改变跨面板预取/排空关系，不能假定保留原性能；
需重新核对实际缓冲与同步后才考虑设备验证。该方向尚未实现或测试。

所有失败和原生成源码保留。当前没有本候选的运行任务，不要把CPU准备成功当作完整编译成功。
