# OB32 小 T OA pair 候选：仅源级准备

**待验证，未采用。** 本候选将原 0<T<=96 版本收窄到既有 OB32 连续小 T 域；不是根据四个设备点建立 batch 白名单。未启动 CPU lowering 或设备任务，没有为新版登记性能收益。

冻结输出：

- [source_candidate](source_candidate)：完整 33 文件候选。
- [source_reference](source_reference)：从主归档 66882634 参考复制的 33 文件。
- 原 96 版冻结源仍在 `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_evidence/source_candidate`，未改动。
- [96 版文档整理包](documentation_review/README.md) 保留原谓词，四文件执行 AST 与原冻结源一致；不是新小 T 源。

## 区间假说与实际改动

`policy.py` 未修改，既有规则是 `OB_ROWS=32 if tokens<=32 else 96 if tokens<=96 else 128`。在 OB32 域，`ceil(T/32)=1`，每个输出组只有一个量化任务；本候选仅检验共享 O_A 输入在该原有组织方式下的完整 L1 效果。这个边界本身不证明收益。

cube/vector 同时采用：

```text
OUTPUT_GROUP_PIPELINE
&& OA_COLS == 128
&& OB_ROWS == 32
&& 0 < tokens <= 32
```

前两项与 OB_ROWS 是现有 constexpr；tokens 继续来自真实 Tensor 动态轴。条件没有冻结 T/B/Q/L/position，也没有增加 host policy 字段。原 pair 数学、single 数学、五 owner 映射、量化/尾零、缓存与阶段通知均不变。两处调度注释明确写出“OB32/单 32 行量化任务域是待实测假说”。

128K/B4 的 T24 与 B5 的 T30 是主会话安排的同轮验证点。其它 OB96/OB128 区间全部关闭，包括原 96 版曾正向的短历史 B16；不按单点结果保留例外。

## AST 证据与边界

[source_ast_audit.json](source_ast_audit.json) 为 PASS：

- 相对“96 版完整 33 源 + 四文件文档覆盖”，剥除所有 module/class/function/async-function docstring 后，只有两处 `If.test` 改变；其余 31 文件完整 AST 一致。
- 将候选这两个谓词各自恢复为旧谓词后，两调度模块的完整 AST 与 96 文档版一致。导入、常量、所有函数体、类型注释和语句顺序均参与比较；没有只比较选定调用或签名。
- 33 文件参考是归档 668 的复制；候选与参考的其它 29 文件直接文本相同。原有 OA 数学与 phase helper 全部函数 AST 保持参考实现，新 pair/wait helper 相对 96 版 AST 不变。
- 关闭路径共 12 个完整调度函数对照通过。核心两组只提供 `OB_ROWS=96` 或 `OB_ROWS=128`，不提供 T/OA_COLS/OUTPUT_GROUP_PIPELINE 的值；新分支已可消去，整个 cube/vector 函数 AST 与 668 一致，pair 和专用 wait 均不可达。
- 另核对 OA256、关闭输出组流水，以及手工 OB32/T0、T33 的防御性回退。后两项是源码分支核对，未生成静态 shape kernel。OB32 源码谓词在 T0..97 的有限检查中仅接受 T1..32。

**不声称整个候选模块与 668 完整 AST 相同。** 候选模块仍有新增 import 和未调用的 pair/wait 定义；关闭证明针对完整调度函数及原有 helper。也不把源级折叠证明写成实际 CPP、片上分配或事件相同，本轮没有编译。所有 docstring 差异不计入执行 AST；文本 patch 保留完整差异。

## 复现与文件

[prepare.py](prepare.py) 仅使用 Python 标准库；输入为原 96 版 33 源、其四文件文档副本、668 参考。实际执行日志为 `prepare.stdout` / `prepare.stderr`。重建应使用新的输出目录，保留本次冻结目录：

```bash
python3 '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_oa_pair_small_t_20261018/prepare.py' \
  --frozen96 '/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_evidence/source_candidate' \
  --reference668 '/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_evidence/source_reference' \
  --documented96 '/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-sync-20261018/tests/pypto_test/ultra_oa_pair_documented_20261018/source_documented' \
  --output /绝对路径/新的输出目录
```

补丁分别为 [small_t_vs_documented96.patch](small_t_vs_documented96.patch) 与 [small_t_vs_reference668.patch](small_t_vs_reference668.patch)。[SOURCE_FILES.json](SOURCE_FILES.json) 仅列两份 33 源的路径与大小。全部文件均为小文本；不含编译产物，不做 hash，不修改正式源、既有冻结源或共享依赖。

本目录源码与准备脚本已交主会话冻结，后续是否提交或采用由主会话决定。

