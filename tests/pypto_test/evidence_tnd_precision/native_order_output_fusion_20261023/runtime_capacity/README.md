# 初始化容量与128K资格

原128K/B24、24000KV块在生成前触发2GiB余量保护，缺口约975MiB；不认定为精度失败。

两个独立单卡进程，仅调用现有`pypto.torch.init`，无模型或算子：默认3167.320MiB；`ring_heap=[56,96,176,1]MiB, ring_task_window=2048`为628.426MiB，节省2538.895MiB。原始收据记录同设备、总显存、SDK和脚本SHA；无隐藏ring环境的确认来自提交后同env只读shell检查，非进程内收据。

静态payload汇总不包含碎片、元数据和在途任务，不能用它证明容量足够。进入整模型前必须单卡完整T144图、Native A/B八边界和ABA通过；所有rank仍须满足原24000块、2GiB余量保护。小容量不等于算子延迟优化，也不自动成为默认。

单卡资格已通过：128K/B24/T144，event2049.609→2007.813μs（降低2.039%，12/12），先计时后无条件三臂Native A/B八边界与ABA通过。未出现小ring deadlock；原始结果见qualified128k_b24.json。

## 复现顺序

先参考外层README准备父目录的共同SDK、3399文件固定源码及输出融合根，再解包本目录executed_helpers.tar.gz到同一新实验根，保留ring-qualified/及whole-ring/相对层次。

1. 在ring-qualified/ source env.sh、设置唯一PREFLIGHT_RECEIPT，先用无设备任务运行compile.py fused，确认root64、NPU未初始化和源码收据。
2. 用单卡task-submit提交`bash ./run.sh <新标签> 24 131071`，max-time780（内部750），要求accuracy_executed、comparisons和B_comparisons两臂各八项bit_equal、三臂live_input_ABA全部为真；不能只看exit0或performance winners。
3. whole-ring/先执行无设备preflight.py 128k（固定bank、Native gold和新输出路径）；重新核对manifest。实际init参数必须被runtime_sizing观察并保存，签名兼容CPU检查必须保留。
4. 只有前两级通过，才可提交16卡`bash ./run.sh 128k`，max-time780；原脚本使用固定输出名，复现须新隔离根，禁止覆盖旧证据。run.sh内部750秒，发现首差即失败。
5. 运行audit.py核对16rank、192token和事件、实际FULL图轨迹、PTO捕获资格与真实ring kwargs，再用父目录summarize_whole.py统计同一步16rank最大值。保留原始终态，不能用CPU重审覆盖原设备退出码。

原始bank、Native黄金、SDK和编译产物需自有可读资产；归档不含权重、cache payload或SDK二进制。迁移到其他路径需显式重建manifest与CPU收据，不能去掉身份和精度门禁。

## 整模型终态

16卡task_20261023_133342_949308790已exit0，模型入口13:34:20→13:42:58（8分38秒）。全部rank原24000KV块、21GiB upper及2GiB预留保护通过；实测分配前空闲25.784～26.733GB。16个worker实际init记录小ring配置，常驻约623.94～628.66MiB；每rank隔离visible device，因此进程内device均为逻辑0，并非全部使用一张物理卡。

每卡提交24，计时窗口实际21请求/T126、图桶144。warm192＋测192共147456个token位置、接受事件与Native黄金一致，零抢占，实际FULL replay与21层PTO捕获资格通过。完整原始数据见上级whole128k_raw.tar.gz及SHA清单。target forward均值89.824597ms，周期102.120351ms；无本轮新跑的Native/冻结版整模型配对计时，不给整模型加速百分比。
