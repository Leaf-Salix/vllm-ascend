# 可选GM子视图能力原型（PyPTO #3042）

从pypto-csa-memory-reuse-20261018的d01de8523及36项tracked修改、3项新
Python源码继承，逐文件核对相同；runtime同57e499238。新独立worktree
pypto-csa-gm-subview-20261019，所有构建在本树，不复制.so或编译产物，
不改共享安装、不改正式CSA。WORKTREE_PROVENANCE.json记录源流。

选择显式tensor.slice(..., gm_view=True)而非按下游用途悄悄改普通slice：
普通slice保留原Tile语义，显式模式保留GM描述符后由pl.load实际搬运。
首版只允许连续rank-2 ND完整行，列offset=0、width等于源width，无
clamp/drop_dims/sub-byte；动态起始行和span完整边界由原类型检查与调用方
device guard共同约束。type deducer、Python DSL/IR wrapper、TensorToTile
和共用view codegen一致改动。不存在新增C++公开builder或binding签名；
继续使用既有create_op_call的已登记bool属性。

生成代码拟用元素单位addptr一次推进真实GM基址，再make_tensor_view；
后续view不能丢偏移或重复推进。真实计算Tile仍拒绝作为GM。类型限制是
初期能力边界，不把非法列stride伪装dense，也不搬整个跨度到UB。

已有同名编译错误作为能力需求#3042提交。本轮先独立构建和相应UT，
之后再做CPU完整链与设备地址probe；当前未完成，不宣称修复/收益。
按仓库loader要求PYPTO_BUILD_JOBS=2构建，命令/日志见BUILD_COMMAND.json。
较轻的MGATHER复试在相邻mgather_physical_plan_retest_20261019进行，
使用原工具链；不会把本实验未验证依赖混入其对比。

## 独立原型 v3 验证完成

首次完整构建成功；v1 CPU 暴露输出 TensorView 缺少显式 stride，补齐后
根 view 与 slice-view 两种完整 CPU 编译均通过。首次 UT 启动受旧 editable
loader 干扰，未真正运行测试；通过已有 filesystem bootstrap 固定本树
Python/扩展后，v2 得到 50 通过、2 失败，均为 pass verifier 未接显式
GM slice。补齐 verifier 后本树增量构建成功，v3 **52 项通过**。
上述失败与日志全部保留，不放宽原 Tile 输入拒绝或改测试预期。

A3 最小地址 probe 任务 task_20261019_093031_303984116239 终态 exit0，
26 项同址 graph / 位型 / 只读 / guard 检查通过；不采性能。覆盖动态
R17/97、正序/逆序/重复/尾界/非法与大 span fallback。完整 CSA 的新能力
关闭路径正在做生成物对照；尚未证明新能力用于完整 CSA 后有收益。

v3 快照位于 source_candidate_v3，建议补丁 gm_view_only_v3.patch；
旧 v1 快照/补丁不覆盖。验证汇总 VALIDATION_V3.json。已把局部实现、
验证范围与建议补丁补充到 PyPTO #3042，链接见 PYPTO_ISSUE_COMMENT_URL.txt。
没有提交 PR、标签或修改共享安装。

## 完整链后续范围

v3默认不开gm_view的完整AIC/AIV C++对照通过，见FULL_DEFAULT_CODEGEN_EQUAL.json。
接入完整CSA后另发现父工具链现有tensor.view在FlattenTileNdTo2D中的类型
引用失配，独立issue #3046及建议补丁放在相邻pair_dma_full_pipeline目录；
该工作区在gm_view_v3上叠加其修正后完整128K/B16设备门槛通过。带UB重排
候选性能回退16.120%，不采用；不能拿最小地址probe或编译通过宣称加速。
复现需依次应用gm_view_only_v3.patch与相邻flatten_view_only.patch，
保留继承基线的已有本地修复。两份建议均已通过issue提供给上游，无代码push/PR。
