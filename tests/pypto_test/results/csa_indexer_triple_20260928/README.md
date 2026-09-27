# Indexer三query/M192候选：仅完成CPU编译

基于2a740c1f，独立工作树`.cache/csa-indexer-triple-2a740c1f`。
尚未取得设备结果、未合入，不修改正在执行的QKV EP16候选。只有[候选补丁](candidate.patch)涉及算子。

## 与既有策略的区别

当前双query将128个head行作为一个INT8 QK矩阵乘，再用一次K128的FP16 Cube合并head权重。
新候选用三query的192行：QK为M192/N128，head规约为K192，系数矩阵前3行各自使用64列对角块。
每个S6请求从3组变为2组，相同Key panel少读一次；各query仍独立应用位置可见性与Top-K。

此前失败的4＋2候选仍用两次M128 QK，且当时未合并head规约，新增循环/流水成本明显。
本候选是一次M192，继承当前已有的合并规约、query/系数L0A驻留、N128分页流式读取和scale预取。
不把“减少读取”直接当作性能收益，也不将旧4＋2失败记录删除。

同一份constexpr实现按实际输入选策略：压缩历史超过8192行且query数至少48时选三query，
其他长档及短档保留双query；更短历史保留现有Vector路径。测试脚本不切历史版本。
双/三query共同将AIV的分数读改为同一次跨行load再切片，所以旧双query成绩也不能替代新代码实测。
FP16输入舍入、FIXPIPE缩放、Key/scale页布局、Top-K树、worker数量和长短档调度标志不变。
新增K192规约是否与K128逐元素一致仍须验证，不能只凭插入零系数声称设备bit一致。

## 片上容量与CPU证据

[编译入口](compile.py)、[完整日志](compile.log)：PyPTO lowering、PTOAS、CCE及链接通过。
进一步以`--full`完整编译CSA并链接也通过：[完整CSA日志](compile_full.log)。
`CompiledProgram.load()`只构建二进制，不创建设备Worker或执行NPU。Ruff与shell语法通过。

生成的三query kernel：

- L0A：Query INT8 192×128，地址0、24 KiB；系数FP16 16×192，地址24576、6 KiB。
- L0B：Key INT8 128×128与前一panel FP16 192×128同时存活，16＋48 KiB恰好占64 KiB。
- L0C：QK INT32 192×128，地址0、96 KiB；head规约FP32 16×128，地址98304、8 KiB。

这些证明编译器能放下当前形状，不证明流水无等待。L0B容量紧，实际收益需要核内计时与泳道。
候选保留原Native cache布局，不依赖页连续或源头分离，不修改PyPTO/Simpler/PTOAS/PTO-ISA。

## 工作量代价，尚非性能结果

按当前24 worker、S6、历史131072、每query真实可见范围和8192行leaf推导：

| B | 原/候选Key页读取总次数 | 原/候选最忙核Key页次 | 原/候选最忙核Score-step×query |
| --- | ---: | ---: | ---: |
| 8 | 25088/16640 | 1056/800 | 66/75 |
| 16 | 50176/33280 | 2112/1568 | 132/147 |

总Key页次减少33.67%，但最忙核的计算量分别增加13.64%/11.36%，并未做到完美配平。
128K/B4与8K档仍选择双query。上表是循环边界推导，既不是DDR实测流量，也不能推算最终加速比。

后续若做设备筛查，先对照长档完整CSA、Score核内时间及状态/保护区，再检查短档共用代码是否回退。
只有核内确有收益才继续处理调度并扩大范围；本目录当前没有任何真机收益或正确性结论。

复现CPU编译：

```bash
source ../env-dsv4-0251rc1.sh
python tests/pypto_test/results/csa_indexer_triple_20260928/compile.py \
  --source ../.cache/csa-indexer-triple-2a740c1f \
  --output tests/pypto_test/results/csa_indexer_triple_20260928/compiled
```

## 已提交的单卡筛查

task_20260928_030634_158754525942：[命令](run_layer.sh)，仅128K/B16。
原版2a740c1f与候选依次测量：正式layer4权重、S6、mode2/atomic1/det0、EPLB关闭，
每侧5次预热＋20次图计时，再独立采4个图重放DFX窗口；不混合profile与主计时。
每次执行仍检查metadata、保护区、有限值及Top-K结构。
当前QKV EP16占用16卡，此任务排队，不与其混测，也不叠加QKV算术候选。
若长档核内没有收益则不扩大矩阵；如有收益再按用户要求保留核内改进并处理整体调度、短档共用代码回归。
该计划不构成设备测试通过，当前尚待结果。
