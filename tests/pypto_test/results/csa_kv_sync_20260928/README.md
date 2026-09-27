# 固定K KV整组启动调度候选

基底88d0744f，仅给KV投影SPMD设置sync_start=True；暂时只支持本实验atomic0，不能作为atomic1的生产实现。
固定K最多12个AIC block，小于24个可用AIC。若保留，必须在生产按固定K分支隔离，atomic1不加此开关。

依据：自适应KV的B40四窗口中一个启动分散53.82μs，8块约123μs启动、4块等QR完成约176μs启动。
Simpler现有normal/pending分配可将剩余块预放到忙核；sync_start走整组准入和全局资源计数，
会优先于普通就绪任务，也可能推迟weights/Compressor。故只作为待测假设，不宣称一定修复整层或EP尾部。

真实EP16自适应KV任务已完成；本候选单卡随后开始，没有干扰其正式10步。
只验证128K/B16与8K/B40，各20次本体计时，状态/图重放；新增两侧各2个DFX窗口检查关键链转移。

CPU完整编译通过；编译结束时EP16仍在模型权重加载，未与正式forward计时并发。
task_20260928_063940_83121115114已在16卡任务完成后取得device0，当前运行中。此次只使用atomic0。

[Scheduler/Worker配对](pending_analysis.json)补充了忙核预派发的直接证据：window_1最后4块在约127.24μs已派发，
对应QR约175～176μs结束，KV随后才被Worker接收；dispatch→receive约48～49μs。
这与普通任务可进入忙核pending slot的源码机制吻合。整组准入的代价仍须按长短本体和关键链实测。

其他三个窗口的KV块多排在indexer Compressor的kv_score_proj_0之后，dispatch→receive约15～21μs。
因此不能只修饰window_1的图：还需比较完整Worker末尾、Sparse启动与无profilerP95，防止代价转移。

16卡任务已退出0，本候选随后取得device0开始执行；当前尚无采纳结论，生产调度未改。
