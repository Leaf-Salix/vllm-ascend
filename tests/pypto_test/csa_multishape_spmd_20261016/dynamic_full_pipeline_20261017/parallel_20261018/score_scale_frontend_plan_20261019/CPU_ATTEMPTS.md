# CPU 编译过程

v1 在新接线处误用了 idx_table，实际参数名为 indexer_table，编译器报缺少 block_table 的推断 metadata。
属于本轮算子接线错误，修正后重编，未修改依赖组件。首版完整源码保留。

v2 动态解析通过，但完整编译报 IR auto-name base cannot contain reserved delimiter: t__inline_arg_v0_inline2819。
新 helper 的 return 表达式内嵌套 inline 调用，现改为显式 base 中间赋值，数学不变。
完整失败源码和 result.json 保留，是否为独立上游问题待最小复现核实。

v3 将嵌套 return 调用拆成 base 赋值后，完整动态编译通过，58 个动态轴保留，单 MIX + sync_start。
CPU 合同检查60组工作区布局、352组页证明及4096个叶读取范围全部通过。
检查脚本初版 AST node.values 误当方法及单页方向的 oracle 已修正，均为测试代码错误。
这些均不代表设备精度/性能通过。两档真机已提交，正式源码恢复参考，候选完整保存在 source_candidate。

最小复现v1（main→layout→base）直接 inline 后 SSA 通过；增加 wrapper
形成三层 inline 后稳定触发 t__inline_arg_v0_inline5 命名错误，完整代码和IR保留。
已提交 PyPTO #3041，附 AddQualifier 保留自动命名 role/version 的建议；
没有修改 PyPTO、PTOAS 或 Simpler，没有 PR 或标签。最新 main 只核对源码，
不宣称已经在该最新版本构建执行。

扩展测试壳初次准备误用相对 Path.parent，未创建启动脚本却提交了4项任务，均在执行算子之前以127退出；这是本次工具编排错误。
修复为解析后的绝对父目录，并验证全部HERE资源和启动脚本存在/语法后才重提v2。未改算子或精度门槛。
