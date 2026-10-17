# OA pair 四文件说明整理副本

状态：**未采用**。原 0<T<=96 候选四档结果不支持整体迁移；本目录仅整理中文说明与 import 格式，完全保留其原有谓词、数学、owner、等待和缓存协议。

输入是 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_evidence/source_candidate` 的四文件；输出在 [source_documented](source_documented)。本目录只有四个待迁移文件，并非独立可执行的完整 33 文件包。源级只读结论见 [上一轮审阅](../ultra_oa_pair_readonly_review_20261018/README.md)。

修改范围：

- `output_projection_a_stages.py`：区分原 logical%24 与 pair 五 owner 分工，说明显式 Mat/Left/Right、独立 Acc、动态有效行和尾零；澄清旧整阶段入口不含 pair。
- `cube_schedule.py`：修正“保留原每核任务序”和“全部分支共用 helper”的旧说明；将新增 helper 所在 import 拆成多行，名称及顺序不变。
- `vector_schedule.py`：补齐五 owner 等待及尾零合同；将新增 wait helper 所在 import 拆成多行，名称及顺序不变。
- `phase_sync.py`：区分固定物理 begin 参与核与动态阶段活动 owner，补齐专用 helper 调用合同。

[ast_audit.json](ast_audit.json) 为 PASS：四文件剥除所有 Python module/class/function/async-function docstring 后，完整 AST 相等。比较保留全部导入及其顺序、常量、语句、类型注释与执行节点，只排除源码位置属性；没有只比较函数签名或选定调用。全部 import AST 另行比较相等。**完整原始 AST 不相等**，因为 7 处 docstring 内容发生变化；逐项前后文本在 [docstring_changes.json](docstring_changes.json)。冻结源在准备前后直接读取文本一致。

复现只进行文件复制、文字替换与标准库 AST 解析：

```bash
python3 '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_oa_pair_documented_20261018/prepare.py' \
  --source '/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_evidence/source_candidate' \
  --output '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_oa_pair_documented_20261018' \
  > '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_oa_pair_documented_20261018/prepare.stdout' \
  2> '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_oa_pair_documented_20261018/prepare.stderr'
```

输出清单：`source_documented/` 四文件、[prepare.py](prepare.py)、[documentation.patch](documentation.patch)、[ast_audit.json](ast_audit.json)、[docstring_changes.json](docstring_changes.json)、`prepare.stdout`、`prepare.stderr` 与本 README。patch 从冻结四文件生成，不依赖 Git 或 hash。未导入 PyPTO，未编译、上卡，未修改冻结源、正式源或共享依赖。

