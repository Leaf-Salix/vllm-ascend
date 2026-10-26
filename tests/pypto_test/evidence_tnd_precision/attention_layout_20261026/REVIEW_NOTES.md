# 独立复核记录

## 量化融合预检拒绝版本

- bf16-quant-v1：根入口import替换未命中，CPU加载了旧leaf；该CPU收据不能证明新候选接入，未提交设备。
- bf16-quant-v2：确认接入后，4×1列布局只有16B，不满足32B对齐，CPU编译失败；未提交设备。
- bf16-quant-v3：改成8行后真实新leaf接入及生成物已验证，才进入性能初筛。

## 已完成性能候选

- softmax-hoist：实际bias/position/liveK在四head组前；UB147584B，Cube/sync源AST不变；CPU两臂64/load/noNPU。设备结果未过门槛。
- bf16-quant-v3：实际新leaf已接入，旧leaf不存在；16个8×512 BF16驻留无覆盖，UB163904B；global8192最大值/量化舍入保持；CPU两臂64/load/noNPU。设备结果未过门槛。

## Native512跨query流水

- v1：独立审查发现copy64单core容量/三槽寻址错误，及事件模型未消费clear。未提交设备。
- v2：两问题已修；独立重跑597mask/12960交错/3756048步通过。CPU动态event表达式emitc失败；未提交设备。
- v3：8组三常量事件分支在work0..127映射等于kind+4*slot；恢复后整个module AST等于v2。两臂CPU64/load/noNPU通过，实际生成物已完成下述审查。
- 容量不能把不同scope_depth的payload简单相加后与256MiB相比。固定Simpler每depth一个独立heap，8K探针每heap256MiB；alloc4精确58.9473MiB，alloc10 145.8425MiB，alloc11 T192为6MiB，分属不同depth。源层虚拟内存和DMA生命周期仍需最终生成审查。

所有源/生成审查均不构成NPU性能收益或Native逐bit验收。

v3生成主审：UB147968B，持久left[16512,49280)、right[49280,82048)，独立carry统计与scratch不覆盖；L1max196608B，L0A32768/L0B65536/L0C131072。PV每128 panel仍按K0/32/64/96累加，仅首MMA初始化；L1/L0复用存在MTE1/M/FIX依赖。实际12事件分支已生成、3槽容量和copy64地址匹配，qk写heads且后续WOA读取heads，tail等待仍在。单ring峰值确认后才提交8K性能。

v3最终单卡8K/B32性能资格通过：root alloc0+1约32.197MiB；父层alloc3+4约66.013MiB（其它RoPE/metadata后<67）；Indexer alloc10+11约151.843MiB，保守加互斥子分支约160.101MiB。各depth独立256MiB heap有余量。资格不提前覆盖128K whole的小heap或任何Native硬件精度。

Wide-softmax-v3性能资格通过：4masked[65536,81920)/[81920,98304)/[98304,114688)/[114688,131072)至对应exp前无覆盖；max[147584,147712)、alpha[147712,147840)；partial1×64×4存活至concat/gather，UB149888B。4×TROWSUM512×8→64×8、final32×8保持原树；prob CAST_ROUND，GMpitch512，cols0/128/256/384各32head且复用前MTE3→V等待。AIC规范化生成码与Direct相同。只为8K/B32性能初筛，不是Native硬件精度资格。
