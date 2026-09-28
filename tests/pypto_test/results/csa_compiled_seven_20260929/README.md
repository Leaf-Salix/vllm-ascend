# CANN9.2：当前性能版与Native真实编译七档对照

阶段出口矩阵：128K B4/B8/B16/B24，8K B24/B32/B40；不再新增8K B16。
Query驻留候选已判定不保留；最终整包冻结为c93ec723，版本及路径见[source.json](source.json)。
两根依赖图解析通过，task_20260929_032251_2642121879已通过auto单卡启动。
当前尚未完成本轮矩阵，不拼接历史局部数字填表。

同一auto单卡依次测七档，每档交替Native/PTO先后次序；各侧独立OPP静态包与AOT缓存。
复用已验证的实际编译半层入口，要求Native实际安装静态包、PTO实际调用且不回退。
同一CANN9.2、mode2、det0，PTO atomic0；EPLB关闭，ring=[256,128,256,32]MiB/task_window4096。
Native部署模板编译配置与PTO一致；Worker绑核和MoE共享专家多流不在单卡半层范围。

每侧5预热/20次无profiler图事件，独立PyTorch profile；PTO再采四个DFX窗口。
报告均值、P95/max、完整原始样本、Native热点PMU与PTO任务核时/包络/启动分散。
长档内部等权、短档内部等权，再按8:2计算变化率；不删除拖尾，不将独立profile核时与正式事件相减。
该矩阵覆盖HC_pre+norm+CSA+HC_post，不等同整模型forward、EP16稳定性或token/DSpark验收。
本阶段复用已有必要状态检查，不为数值中性未改路径重复全套精度测试。
