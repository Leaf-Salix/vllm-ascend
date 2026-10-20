# 当前交接：目标保持 active，尚未达到性能目标

本轮全部工作在此目录；用户新目标为 GOAL.md。正式37源仍未修改。
关键状态见 CHECKPOINT.json，三份上游 issue 已创建，没有 PR/标签/push。

- 排序同分设备probe：46组 score/index 位完全一致；只证明保序压缩后的
  排序树语义，不证明新设备压缩或完整CSA。任务已经completed/exit0。
- PyPTO独立树 `.worktrees/pypto-csa-topk-compact-20261020` 从2dfa7f12
  继承，count物理容量向8个INT32对齐而valid不变；已完成构建，39项UT通过。
  UT v1的xdist子进程曾加载旧editable；补固定sitecustomize后修正了
  full-valid TileView可省略的测试断言，v3为实际结果，未复制.so。
- PTOAS独立树 `.worktrees/ptoas-csa-topk-compact-20261020` 从e5c1c432
  继承；本地A3 helper保持真实Scalar阈值，用Vector生成索引和vreducev2
  压缩，并补GE以保留ties。GT/GE helper的人工拼接最小C++已通过CCEC，
  这不等于新后端完整PASS，也不是生产代码注入方案。
- PTOAS首次配置因GitCode依赖下载失败；第二次沿用旧构建已固定的
  CANN cmake源码路径重新构建。当前exec session为13035，log为
  build_ptoas_v2.log。先核对该handle/日志，不能重复启动。并行限制2。
- source_reference为37文件冻结源，source_topk_candidate为完整38文件候选。
  首段、<=512项及旧根未满时保留原逻辑；其余以真实GE阈值保序压缩，
  只有能减少排序长度才使用较短路径，空候选保留旧根。全部metadata仍device。
- compact_frontend_v1/v2修正valid_shape必须INDEX的编写问题；v3/v4暴露
  TSELS接口写法问题。最终v5删除中间flag，直接调用新GE后端；已通过
  PyPTO阶段，旧PTOAS只在GE验证处拒绝。仍须新后端完整编译和设备验证。
- 新工具链入口run_isolated.sh；run_frontend.sh故意绑定旧PTOAS，只做
  前端诊断，不得用于设备或性能结论。实际完整算子测试继续复用旧
  csa_critical_path_20261020/run_full.py与动态metadata/padding/图/代际门禁。
- 用户已授权Attention数值候选；source_attention_candidate已实现。具体
  数值变化、资源账本和v1～v5尝试见ATTENTION_GLOBAL_SCALE_DESIGN.md。
  v2/v3/v4实际Vec均197376B超188416B；v5使用40KiB score及16头直接
  读取最终PV，Vec172032B、Mat458752B、Acc131072B，完整CPU编译PASS。
  v5编译没有样例shape，58符号轴、唯一MIX与sync_start审计PASS；
  CPU编译不代表真机正确、性能或整模型token一致。
- Attention B16任务task_20261020_190632_268921518547已提交，结果目录
  attention_b16_v1。沿用原逐bit测试器，数值差异如实保留，不放宽门槛。
  使用旧已验证工具链，与等待新PTOAS的TopK候选互相独立。
- 公共helper及多task已冻结97个Python源文件到frozen_pypto；仅路径、
  文件大小和Git提交记录在SHARED_SOURCE_FREEZE.json，不做hash。
  run_frozen_full.py只改测试进程包查找路径，原精度与性能检查不改。
- 用户给的cannbot知识已经定向阅读，采用边界见KNOWLEDGE_REVIEW.md；
  query索引缺失，没有冒充有效检索审计，也没有改知识仓或自动重建索引。

注意共享vllm工作树还有其它会话提交；当前候选冻结时HEAD为24357d11。
最终验收必须冻结当轮multi-task及共享helper，不能用早晨旧表格算提升。
完整目标五档128K相对同轮多task L1降低30%仍未完成，没有新增CSA性能值。
所有仓库未经再次授权不得push；本地阶段commit不等于算子采用。
