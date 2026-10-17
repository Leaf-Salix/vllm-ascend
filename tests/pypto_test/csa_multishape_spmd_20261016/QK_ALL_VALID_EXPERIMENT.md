# 2026-10-17：QK全有效块快路径

候选`qk_all_valid.py`在原QK的tick循环外，从device plan扫描本组query×5个块。
仅全部标记>0时使用快循环；任意标记非正则执行原循环。三个物理成员读取同一
不可变plan，原plan前驱和FFTS依赖保持。不读取Host端Tensor内容，不增加GM缓冲。

快循环去掉已证明恒真的外层块判断，保留内部稀疏行负索引、SWA页、尾部tick、
softmax、PV及输出。展开条件的末尾yield改为普通SSA赋值；回退循环AST保持原样。
行依赖审计先检查原访问体，再加入条件专门化，不以新分支绕过访问范围核对。

prepare/cpu_v1完整通过。实际PTOAS C++两个分支的静态统计：

| 物理函数 | 快循环PIPE_ALL | 回退PIPE_ALL | 快循环mask读取点 | 回退mask读取点 |
| --- | ---: | ---: | ---: | ---: |
| AIC | 0 | 0 | 0 | 2 |
| AIV | 3 | 4 | 0 | 4 |

AIV源码六个判断经B16常量分支优化后，实际回退只剩四处mask读取。上述是静态
位点数，不能当作每次执行次数或性能收益。核内前置扫描仍有成本，代码也变大。
FFTS发布统计使用实际`ffts_cross_core_sync`，初版误搜`set_flag_dev`显示零，
已修正审计字段，未改变算子。证据`qk_all_valid_codegen_audit.json`及四份C++片段。

任务`task_20261017_072611_24632809131`已提交，目录
`b16_row_deps_qk_all_valid_v1/`。正式权重128K/B16/S6；先原exact/状态/图/保护区，
通过后比较原动态PTO与单SPMD的完整三窗L1。未运行Native forward。当前尚无终态，
不预记PASS，不改变现选七档。

## 终态：完整通过，未采用

任务exit0，输出/TopK/六状态、eager/图、30代及保护区全部通过。
单SPMD三窗804.30/817.26/820.92μs，中位817.26；同轮原动态905.18μs。
没有现选同进程A/B，未显示明确优势，未采用、不扩七档。
不能把当前完整fixture通过称为所有padding场景或连续decode轨迹已通过。
结果`qk_all_valid_result.json`，全量记录保留于原任务目录。
