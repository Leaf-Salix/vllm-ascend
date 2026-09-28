# 长档双query的Key预取候选

基底554b3bca；独立工作树`.cache/csa-score-key-l1-pair-554b3bca`。
只让长档双query/M128/N128分支使用已经过S6验证的独立Key L1池与下一面板预取。
预期适用128K/B4；长档S6、三query与两个短档分组保持原策略。
不改算术、分片任务数、原生cache视图、页表/尾块保护或调度标志。

依据本地最新ops-transformer b5b33e14的QLI V2 `InitBuffers/ProcessQk`独立Key/Score缓冲寿命。
当前长S6分组已采用，双query尚未采用；预期两份Key L0B各16KiB、WS占32KiB，合计64KiB。
三query/M192/N128则需要32KiB Key加48KiB WS，超过64KiB，不能直接套用同一策略。
这里只是容量推导，必须核对生成地址和等待，再决定是否做有针对性的单卡对照。

pypto-lib连续cache接口与当前Native分页物理视图不同，本项保持页表读取；
只复用已验证的面板预取，不重新引入入口复制或改变Native分配。

[CPU入口](compile.py)完整生成/编译/链接并load，未执行设备；
沿用`csa_score_key_prefetch_20260928/compile_guarded.py`，
观察七档模型task_20260928_182811_290034987：模型启动即挂起此CPU子进程组，避免与正式计时竞争。
完整CPU lowering/PTOAS/CCE/AICPU链接及load已通过，编译全程模型仍在排队；未发生CPU任务与本轮正式计时重叠。
[生成代码核查](codegen.py)、[地址/等待/九份未改执行核](codegen_evidence.json)：
Key L1基址0，占32KiB；Score稳态基址32768/65536、各32KiB，Key用完后尾段可复用基址0。
Key L0B首块32768、稳态0/16384，WS位于32768起32KiB，合计64KiB。
每N1024仍8次Key读取、8次QK/WS，Key TEXTRACT列偏移0/128交替。
尾段L1复用增加一处MTE1→FIX保护等待（原版0处），保留此成本，不宣称生成码一定更快。
其他四组Score AIC/AIV及该组AIV的9份执行bin一致；不以含路径元数据的.o比较来判断执行码是否改变。

[单卡入口](run_layer.sh)已准备，等待七档EP16完成后才提交，不占用16卡队列所需设备。
只测受影响的128K/B4和8K/B16控制档，每侧20次无profiler计时、四个独立DFX窗口；
八类状态零容差、metadata/保护区及A→B→A，长短分别计核内/完整CSA七三并检查P95。
[汇总入口](summarize.sh)使用实际任务句柄，不重复测试未改变的B8/B16长档。
尚无设备性能结论，不合入生产、不影响已经冻结的七档EP16。
