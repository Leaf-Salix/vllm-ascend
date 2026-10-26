# 独立审查结论

- Direct正式迁移两文件，无新增代码阻断。正式QA126完整模块AST除文档外与实验QA95相同。root不重绑Out heads。
- 实际CPU编排QK t50.add_output(heads)，WO-A t51/t52.add_input(heads)，RAW依赖成立，兄弟scope保留；H8和SPARSE_BLOCKS2防护保留，UB147584B。
- 所属AIV保留DIV、BF16 rint、FP32 inverseRoPE两乘加、BF16 rint；NoPE用第一次BF16，mask无效query置零。
- 四单层Native A/B八边界及ABA、guards、readonly通过。8K/128K whole原始16rank审计独立复算通过，不冒充整模型隐藏tensor逐bit。
- 128K实际24提交/21活跃/T126桶144，21CSA/20HCA、capture1/FULL replay、Native/JIT增量零，24000块、2GiB余量、ring56/96/176/1/window2048均核对。Forward85.4286315918、cycle98.2829730385ms。
- 三方12窗口独立复算Native1251.966909/我方1587.461802/nalinaly cafc兼容版1216.794427us。后者A/B输出relativeL2约0.00333，不作为gold；没有对最新工具链或其整模型token作结论。
- Fold64反向重复不稳定、H16/WOA24/panel ACK/独立panel事件均未赢，全部不合入。panel无ACK模型及生成物保护虽通过，不代表有性能收益。
