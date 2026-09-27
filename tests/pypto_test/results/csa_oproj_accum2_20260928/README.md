# WO-B两份INT32中间结果：CPU编译候选

尚未运行设备测试，未合入生产；不属于正在排队的EP16量化候选。
工作树`.cache/csa-oproj-accum2-2a740c1f`，基底2a740c1f先应用
[`csa_oproj_token_20260928/candidate.patch`](../csa_oproj_token_20260928/candidate.patch)，
再应用本目录[incremental.patch](incremental.patch)。

## 改动与上游区别

此前按8个O组分别发布WO-B的INT32中间结果。整token量化共享一个scale后，
可以把完整K8192改成两段K4096，先做整数累加，再按原来的channel→token顺序反量化。
原始pypto-lib按组使用不同scale，不能直接这样合并整数；此候选依赖整token量化合同。
WO-A、BF16边界、量化规则、反量化顺序及cache策略均沿用量化候选。

- N128、K512，2个K分段×32个输出列块，仍为64个AIC工作块。
- B24/32/40的两块M128行数据共用每个权重panel；其他形状仍按实际行块遍历。
- 中间结果分配48→12 MiB；B16有效矩阵写回12→3 MiB，后续读取亦减少。
  这些是形状推导的逻辑字节量，不是实测DDR流量或性能收益。
- INT8长度8192的最坏累加绝对上界134217728，小于INT32范围；改变分段不改变整数数学结果。
  实际设备输出仍须验证，不能以该推导代替算子正确性验收。
- 新BMM等全部组完成量化后启动，可能损失部分调度交叠，必须实测完整CSA成本。

## 已验证与后续条件

[编译入口](compile.py)对NZ mode2与ND mode0通过PyPTO lowering、PTOAS和CCE/链接，
通过`CompiledProgram.load()`完成二进制构建，没有创建设备Worker或执行设备程序。
[NZ日志](compile_nz.log)、[ND日志](compile_nd.log)。Ruff通过。
最初编译入口的SPMD占位任务缺block index、写入常量未显式转INT32，已修正；这两次仅是CPU诊断入口错误。

三个动态M分支均生成，M128的两块INT32累加器各64 KiB，生成代码地址0/65536，恰好占满128 KiB L0C。
编译器同时提示部分Left/Right流水只能保留一个缓冲，不能以`stage=2`宣称所有层级均双缓冲。
当前只有编译可行性证据。

先等待量化候选task_20260928_015726_18185222732的EP16结果，再决定是否值得进行单卡执行/计时。
若进入设备测试，先验证同输入输出和B3尾块，再看B16/B40代表档；不得把CPU通过记成核内性能收益。

复现（已激活`env-dsv4-0251rc1.sh`）：

```bash
python compile.py --source /data/pyptouser/qinchuanyu/pto-eager/.cache/csa-oproj-accum2-2a740c1f --output compiled_nz
python compile.py --source /data/pyptouser/qinchuanyu/pto-eager/.cache/csa-oproj-accum2-2a740c1f --mode 0 --output compiled_nd
```
