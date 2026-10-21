# 当前交接：继续优先128K/B1、B16、B28

目标active：相对PTO多task完整incore L1提升10%，公式(多task−SPMD)/SPMD。
重点三档仍未全部达成，不标记complete。正式权重W8A8/NZ2；保留精确输出、
TopK/score同分顺序、cache/state/保护、metadata/padding、图/代际合同。
B1可独立分档，其余只按资源区间；20μs门槛已取消。Attention数学方向已由
用户放弃，不继续。未push，不得未经新授权push。

## 正式状态与本轮成果

正式37文件仍为a241fa61采用的B1原半叶+分数/根驻UB，快照
source_after_b1_half_v2。没有把本轮新候选写入正式目录。B1此前直接确认
401.30→371.94μs，净省29.36，5/5快；S1/S5同分/完整S6已通过，不重复。

B16新组合：长G6、64<=T<=96的WS八列窗写回+连续scale读取。首轮正式
794.50/候选782.66，净省11.84，4/5快；正常反向792.24/784.34，净省7.90，
4/5快；真实B16/S5、capacity6(T80)通过。均完整精确/状态/保护/图检查。
初始统一别名版在池宽1仍多约8KiB生成C++，不预称其执行无代价；修为两处
WS编译期分支，QK大循环共用。source_score_stream_guarded启用完整PTO IR
与已上卡候选alpha相同，关闭完整IR与原正式相同，见FOCUS_INTERVAL_IR_AUDIT。
只删除loc/SSA名/空白，没有删操作/数值/地址。B16候选可准备集成，未采用。

B28核归属重排：保留整个逻辑worker的叶/运算/顺序，按前段KV负载换物理核。
正常正向1161.42→1151.32（净10.10，4/5），反向1162.06→1149.28（净12.78，
5/5）。24核实际置换与模型完全一致。身份映射同地址开关1168.62→1142.76
只作诊断，不能当正式净收益。初版guard==1没有激活，13.84us不得归因重排。

原128<T<=256区间在B24身份回退慢16.16us。均衡闭式证明的改版微例和完整
精度通过，但B24仍慢9.34/1窗快、B28慢6.14/0窗快，已停止。改用原正反向
已测版本，只把Host资源区间收窄为144<T<=256；144=2个KV M64+1个M16尾。
无B点白名单。source_score_owner_interval的B28非policy37文件和constexpr
完全相同；B24关闭仅Scalar cast位置有差异，不能预称完整IR相同。当前IfExp尝试在两CPU均报Unsupported expression type，未上卡；这是使用不支持
的DSL写法，不作为编译器正确性bug。组合source_focus_combined采用普通赋值版，
不为一处Scalar cast位置差异复制整段数学。B16/B24/B28完整集成均通过精确、
状态、保护、metadata/padding、图及代际；完整L1分别为：
- B16：多task838.40、正式781.96、组合780.02，净1.94us，4/5快。
- B28：多task1187.76、正式1166.02、组合1149.78，净16.24us，5/5快。
- B24：正式1026.98、组合1031.66，中位慢4.68us，2/5快，尚不能定为稳定回退。
B24正常角色互换已提交task_20261021_093728_23804715449，尚未收齐前不采用。
正式目录仍a241fa61原B1版本，source_focus_combined是八文件候选，不要误认已采用。

## B1本轮负例与诊断

连续scale读取完整PASS但慢7.48us、0/5快，停止。O_B的M32改M16精度通过，
但371.14→376.02us，慢4.88、0/5快，停止，不扫描M8。正式B1未因此改变。
当前B1全72角色阶段诊断通过，1565通知记录，见B1_CURRENT_PHASE_ANALYSIS。
带探针端点不可当正式完整L1；活跃6个Top-K消费者扣已计GM等待后约5～6.5us，
仍含搬运/Scalar/探针。后续更值得检查O_A权重供给、K流水；不再优先抠根归并。
没有改Attention数学，没有把阶段时间差直接承诺为可省收益。

## 组件与工具链

PyPTO独立pair树8c360367，PTOAS本地c04ecdf9（实际build/bin/ptoas），
Simpler独立core-args-256树。本轮没有修改依赖源码或共享安装。
BOOL转INT32=-1是已报PyPTO#3063，新增A3 literal/constexpr/device对照及
INDEX差异，评论6301892254。纯PTO确认signed i1的index_cast/sitofp常量与
运行时不一致；FP32还在A3 CCEC失败，已提交PTOAS#1652，附建议、无PR/标签。
浮点只做Host原生成算术和CCEC，没有冒称NPU执行。初版Host探针漏设备宏
导致未执行，v1无效；修正后的ptoas_i1_cast_repro_v2才是证据。

## 复现、记录和下一步

公共基线frozen_pypto_goal10_b1update=37c47753，含多task B1优化4d5b2ed9。
共享仓其它会话最新1142b247只新增过程测试，不覆盖当前冻结基线。S5专用
旧harness用frozen_pypto_goal10，明确为精度参考，不混入性能数字。
CPU任务全部终态；只有B24最终反向设备任务待收。当前所有候选/失败/脚本/
轻量结果在此目录，统一过程同步ops/pypto/DSV4_CSA_SPMD_OPTIMIZATION_LOG.md。
不提交权重/.bin/.so/.pt或大编译产物；只存路径/字节数，不做hash。

先收B24最终反向结果。若无稳定退步，采用source_focus_combined的八个差异文件，
补整体中文说明和详细采用commit；若稳定变慢，继续精简关闭分支/核对调度。
B16已三轮正向，B28已三轮正向，不重复未变的独立验证。不push，目标active。

用户中途询问DeepInfer布局，已只读核对main 774ce244，源码临时目录
/tmp/deepinfer-layout-20261021-v2955zvl。主体是TH/TND；decode非融合适配临时
[T,1,N,D]，第一轴仍是token数。关键证据attention_backend.py:37/94/124，
prepare_tensor_input_base.cpp:77/98，zos.cpp:213/264，model_runner.py:180/185。
原CSA优化目标未被取消；答复该问题后从B24待收结果继续。
