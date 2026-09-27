# v9：8K采用Native片上Score和Cube head规约

压缩历史≥2048时复用v8双query路径，小于2048仍保留原Vector路径。
适配器按同一阈值预留768行尾块空间，防止扩大tile后读取超过连续cache分配。
8K不再将64行INT32分数送Vector；增加Native式FP16 QK舍入、FP16 head系数与Cube WS。
单leaf改为两个半leaf排序后归并，因此量化及Top-K/tie规则变化单独记录。

8K/B24本体1081.029→1039.979 μs（−3.80%），p50/p95=1038.260/1061.940 μs；
同轮Native1150.171 μs，完整PTO路径1369.895 μs，仍慢于Native。
Score核内从v7的76.59～91.92 μs降到37.23～40.34 μs；
Score→publish为75.66～117.36 μs，仍有独立merge和物理分配等待。
四窗口使用23/24/24/19个AIC承载24块，不能把整个span当成核内耗时。

metadata/slot保护区、Top-K结构通过，输出非有限值0，max_abs=0.03125、RMSE=0.0032981。
Top-K集合替换545，v7为534；零容差FAIL仍保留，未完成新策略的16卡token/DSpark验收。
CPU完整编译通过，任务task_20260927_125220_30470310985；[运行命令](run.sh)。
[计时](h8192_b24/timing/report.json)、[聚合](summary.json)、[泳道](h8192_b24/swimlane/dfx/merged_swimlane.json)。
