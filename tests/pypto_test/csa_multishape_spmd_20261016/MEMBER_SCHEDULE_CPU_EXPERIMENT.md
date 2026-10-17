# 2026-10-17：先按成员依赖分核的 CPU 评估

本轮没有提交 NPU 任务，也没有生成候选二进制。现选七档保持不变。

`member_schedule.py` 从已证明的行依赖重新构造数学 DAG，再做成员级分配；
原方案则是先按整个 task 分核，再收窄成员等待。共 973 个物理成员，
MIX 的 AIC/AIV0/AIV1 绑为一组后得到 877 个调度单元。
保留 Indexer/QK 的同组映射、216 条 FFTS 资源复用边、明确的数学依赖；
重新建立同核顺序、传递约简及发布/等待表，并检查翻译前后祖先集合相同。

以下均是固定原阶段成本的离线估计，不是设备性能，也不是硬件下限：

| 策略 | 原安排估计 μs | 新安排估计 μs | 变化 μs | 远端等待 | 发布 |
| --- | ---: | ---: | ---: | ---: | ---: |
| critical | 771.53 | 802.49 | +30.96 | 15569 | 747 |
| earliest | 771.53 | 768.79 | -2.74 | 12696 | 720 |
| finish | 771.53 | 858.52 | +86.99 | 13643 | 739 |

最好的估计仅减少 2.74 μs，而且比现选的 12183 次远端等待更多，暂不上卡。
这不能证明成员级重排没有价值，只说明当前三种启发式没有形成值得验证的优势。
脚本是 CPU 原型，尚未接入 `shape_specialization.py`。输入使用现选行依赖审计，
将来接入时仍须重新运行源数学合同核对，不能只依赖历史 JSON。

结果位于 `member_schedule_{critical,earliest,finish}_cpu_v1/`。每份目录包含
`audit.json`、`schedule.json`、`logical_and_physical_graph.json`。
生成器和三份归档中已明确标注原分核/等待统计被成员排程替代，避免继承旧审计的
“原核顺序保持”描述。此修正只改描述性元数据，没有改已计算的调度图或结果。

复现示例：

```bash
python member_schedule.py \
  --schedule b16_cached_row_deps_v1/specialization/bindings/h131072_b16_prepared.json \
  --row-audit b16_cached_row_deps_v1/specialization/control_variants/h131072_b16/row_dependencies/audit.json \
  --cost-profile b16_dynamic_l1_cost.json --policy earliest --output 新目录
```
