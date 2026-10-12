# 过程090：定位无标记普通路径的初始化开销

2026-10-12。当前状态：CPU检查通过，单卡任务排队；尚无本候选的性能结论，未接入正式SDK。
上一轮三项验收见[过程089](../superkernel_feature_acceptance_20261012/README.md)。

## 范围与假设

只处理用户第1项：PTO脚本完全没有Wait/Set时，新增运行时流程是否造成回退。
过程089的短任务链相对旧SDK慢9.92%，不能用长任务链启用标记的1.65%收益抵消这个问题。
本候选检查`prepare_subtask_to_core()`每次初始化独立交接缓存行的成本，不提前将全部回退归因于它。

候选修改私有Simpler两处：`scheduler_types.h`复用原padding字节记录两个缓冲槽是否干净；
`scheduler_dispatch.cpp`仅对被明确证明不会读写交接状态的普通入口复用已清理的状态。
首次使用、带Wait/Set入口、旧入口及不明来源入口均完整初始化；切换到这些入口立即取消复用资格。
真实Wait/tail合同优先于证明位，MIX按AIC/AIV0/AIV1分别判断。
派发、ACK、选核、原有allow_early_resolve与真正流水等待保持原样。

该实验暂用现有metadata字节的bit3～5传入已知无标记测试入口的证明，不增加用户参数或环境开关。
证明由测试编译钩子注入；**尚未实现通用编译器识别**。
若有收益，后续须保守排除legacy async、外部C++/模板入口及其他不可证明的函数，不能直接把“没有两个局部标记”当成完整证明。
运行时只构建在本实验独立目录，原Simpler源码已经恢复。普通安装、公共算子和另一个会话均未切换。

## 已完成检查

独立CPU构建复用当前测试目标的其他对象，只重编候选scheduler及增加了状态切换测试的测试文件。
不修改正在排队任务所用的SDK源码和库。19项通过，包括：

- 两个缓冲槽依次经过普通→带标记/旧入口→普通，后续不读取残留前驱、回调和trace。
- AIC、AIV0、AIV1各自的证明不会冒用其他成员的证明；矛盾的Wait/tail声明保守处理。
- 原有派发、直接参数、寄存器代次和谓词相关检查。

CPU脚本初次因机器没有`patch`程序而退出；改用Git应用隔离副本补丁后通过，没有进入设备执行。
[CPU结果](cpu_results.xml)、[CPU日志](cpu.log)、[候选补丁](candidate.patch)、[复现](repro/)。
复现脚本沿用私有目录结构：根目录为
`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009`；
`cpu_check.py`、`transition_tests.inc`和`candidate.patch`放在其`artifacts/boundary_normal_state_20261012/`下。
在根目录执行`python3 artifacts/boundary_normal_state_20261012/cpu_check.py`；
队列脚本位于`artifacts/run_boundary_normal_state_20261012.sh`。归档副本不能直接从本报告的repro子目录执行。

## 排队中的设备对照

任务`task_20261012_093458_69270213448`，自动分配一张卡；09:34提交，当前仍pending。
原命令：`task-submit --timeout 3600 --max-time 900 --device auto --device-num 1 --run 'bash <私有目录>/artifacts/run_boundary_normal_state_20261012.sh'`。

同一占卡任务顺序为：旧SDK前测→当前SDK前测→候选→当前SDK后测→旧SDK后测。
三个当前SDK进程使用同一份带普通入口证明的编排及同一组kernel；原当前运行时忽略证明位。
普通kernel指令仍与过程089的当前普通入口相同，编译时仅增加证明metadata。
因此候选与当前运行时的对照只改变运行时；对旧SDK的对照包含最终采用该证明所需的编排成本。

沿用单task约5/50μs、相同Scalar256轮、48个AIV block、1级和17级任务链，
每路5次预热、30个样本、每次图内连续8次调用后均摊。无profiler，保留所有min/mean/max。
各路逐级输出、变更输入重放和保护区检查随性能测试执行。
随后只补AIC/AIV/MIX局部交接的3项功能回归；它们检查完整等待，不替代过程089的真实M/V事件重叠证据。

私有基础版本：Simpler `3d0b9c820`、PyPTO `94c37915f`、PTOAS `fdebb38b`，CANN9.2；
旧SDK仍为Simpler `ef6812891`/PyPTO `cb2484471`，各自完整配对，不混用绑定。
设备任务结束前，不宣称候选消除了普通路径回退，也不提交通用运行时实现。
