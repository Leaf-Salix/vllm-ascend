# Indexer WS加载顺序CPU检查（2026-10-17）

**没有提交NPU。** 仅把15处上一块WS的Right extract放到当前QK matmul之后。
完整编译通过，但其中1对当前QK/上一块WS Right区间发生复用，预期的并行存储条件不成立。
当前选择不变，不把源代码顺序变化称为硬件重叠。

现选本来就有QK(current)/WS(previous)和下一Key预取；本候选不重复实现整条流水。
`indexer_ws_load_order.py`逐对检查无值依赖，全部434个原顶层语句AST多重集合保持相同，
原最后5条排空语句位置/内容及其他函数不变。prepare只在该CPU进程内注入catalog，
没有改默认形状派发或新增设备driver选项。

准备v1/v2因遍历被原地交换的列表而重复匹配已交换节点失败；v3修为原列表快照后，
对末块名称的假设不正确，15处交换正确但末块计数0。最终v4核对真实buf_last_pair及
最后5条语句，准备/完整编译通过。CPU_v2在source不存在时退出，未进入编译。
这些是变换器/试验编排错误，不是PyPTO编译器故障。所有日志保留。

| 静态位点 | 现选 | 加载后移 |
|---|---:|---:|
| matmul | 32 | 32 |
| PIPE_ALL | 1 | 1 |
| M→MTE1 wait | 32 | 32 |
| MTE1→M wait | 37 | 37 |
| set/wait各 | 166 | 165 |
| 当前QK/前块WS的Right全部互斥 | 是 | 否 |
| 当前QK/前块WS的Acc全部互斥 | 是 | 是 |

这里只检查实际区域与静态位点，不能用少一个set/wait宣称性能收益。
若继续，应先保证需要重叠的Right数据拥有独立生命周期及存储；不能删掉保护真实复用的同步。

证据：`indexer_ws_order_prepare_v4/`、`indexer_ws_order_cpu_v4/`、
`indexer_ws_order_codegen_v4.json`及`.cpp.txt`、`indexer_ws_order_selected_codegen.json`。
