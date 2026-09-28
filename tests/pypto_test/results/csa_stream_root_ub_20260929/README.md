# Top-K 中间 root 留在 UB：保留核内收益

2026-09-29：task_20260929_013942_24190553055 auto/card0完成exit=0。
相对已验证stream2048候选，仅把前2048候选的Top-512中间根留在UB，再与尾段合并。
长档Score AIC/AIV核时分别−4.120%/−4.015%，长短8:2分别−1.163%/−0.481%。
完整CSA长+0.888%/短+0.537%，8:2+0.818%，不宣称本体性能提高；两档P95/max均下降。
按用户“核内受益即保留、随后处理调度”要求，将分段排序和UB根一并纳入性能版。
精度版与Native cache布局不改，四个dummy直接依赖候选没有叠加。

## 代码与Native依据

参考ops-transformer 28f40354的QLI V2：累计root保留在UB，直到需要发布时才写出。
Native内部BASE_TOPK为2048，PTO保留模型要求的512；每query的根4096字节，S6共24KiB。
前2048排序后保留根，尾段保持原排序和“尾段先合并”的并列值规则，最终结果写GM。
Native的Cube WS结果同样先写GM，Vector读入后在UB做key scale、排序与根合并。
本候选仍保留PTO缩放后score_arena的GM中转，只删除临时根的往返，不是完整移植Native流水。
源码依据：quant_lightning_indexer_v2_service_vector_arch22.h的ProcessVec1（388、415、442行）。

私有整包位置见[source.txt](source.txt)，增量见[candidate.patch](candidate.patch)，
正式算子相对旧生产代码的合并修改见[combined_production.patch](combined_production.patch)。
基线是stream2048候选，不是旧生产版，不能跨轮拼接百分比。

## 验证与范围

- 两侧生产/测试根依赖图解析、完整CPU编译及load通过。正式算子纳入同一文件内容后依赖图解析通过。
- 生成代码group6的TLOAD调用点146→140、TSTORE94→76；group3为107→104、47→38。
  这是展开后静态调用点，不是每次执行次数；短路径无prefix_roots，调用点计数不变。
- 128K/B16、8K/B24：5预热/20次正式计时，独立四窗口泳道。
- 八类状态跨版本逐元素零差异，A→B→A、metadata/保护区通过，不代表Native或模型token/DSpark验收。
- 短档算法和生成调用点未变，但Score核时上升、包络和启动分散下降；核内时含等待，不能认定短档算术退化。
- Native列是手工调用控制，可读取私有OPP既有静态包，不能代替真正编译半层的两侧比较。
- 后续补受影响长B8/B24；B4原双query长路径未启用本次分段排序。阶段出口仍要覆盖七档与最终模型验收。

完整采样、核时、P95、检查和全部泳道路径见[RESULTS.md](RESULTS.md)、[evidence.json](evidence.json)。
