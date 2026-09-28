# forward计时事件提前初始化

旧观察器在每个正式forward中构造新的begin/end Event。当前安装的torch_npu 2.10.0.post2
`torch_npu/npu/streams.py:120–123`明确说明底层事件直到首次record才初始化。
这可能使某rank在begin.record之前被观察器自身拖迟，并让其他EP rank在forward内等待；
**尚未证明这是既有3ms/14ms尾部的根因**，尤其准备阶段的延迟发生在forward包装器之前。

现在于offline_begin_forward中预建10对事件、全部record，并等待最后一个完成。
该一次等待发生在生成和warmup开始之前；正式10步只重记事件，不逐步同步。
仍只包围原_model_forward，保留全部等待样本，不扣除EP等待、不改变CSA、GC或生产Runner。

结果新增`timing_event_setup=prewarm_before_generation`。旧记录视为`lazy_per_forward`，
同为旧记录仍可比较；Native/PTO观测策略或host诊断开关不同时收集器拒绝配对。
新旧测量结果分别记录，不能把这项工具修正算作PTO算法收益。

验证：

- CPU定向7项通过：forward边界、缺失/重复调用等失败窗口、事件准备次数、正式步无新建/同步、混合策略拒绝。
  [日志](cpu_test.log)。Ruff和git diff --check通过。
- 单卡task_20260928_090211_187325630621，device8，退出0；独立进程分别测试CANN event mode0/1。
  预创建后重记10次图重放，时间戳更新且严格递增，耗时为正，变更输入后的输出精确通过。
  [mode0](mode0.json)、[mode1](mode1.json)、[脚本](run_probe.sh)、[探针](probe.py)。
- 该小图只验证事件复用和图重放功能，不比较微小kernel性能，也不是EP16尾部修复证明。
  后续必要模型对照两侧均使用这项修正及已有可选主机分项诊断，不为工具单独重跑七档。

仓库全量format检查此前因环境缺pre-commit无法执行，见验证日志§265；本项不重复安装或扩大检查。
