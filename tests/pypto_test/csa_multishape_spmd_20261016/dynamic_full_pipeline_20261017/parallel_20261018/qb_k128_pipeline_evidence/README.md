# 动态Q_B K128双L0供数候选

用户要求有必要时启用double buffer/cache预取。本候选针对原NZ Q_B的真实约束：
N256/K256的单个INT8 Right就占64KiB，`pipeline(stage=2)`没有形成两Right。
保留原N256列owner和M128动态尾行，把K1024分成8个K128数学块，显式提前一片
到Left/Right。不是只把stage数字改为2，也不是重跑旧N128列分核负例。

## 实现与动态合同

修改仅query_projection_stages.py::_q_b_nz_stage，ND及其余32个模块保持。
T、B、metadata及完整SPMD入口继续动态，列块数/owner与原版相同；仅模型
固定K=1024的八个数学块展开。INT8点积最大绝对和1024*128*128<INT32上限，
沿K顺序累加，无FP32重排。源级18个T边界的输出行/列唯一覆盖通过。

完整动态CPU v1遇到两个独立if间Tile定义的SSA作用域错误，未运行设备。
v2将最后一片改为显式排空，移除该定义跨作用域，完整符号编译通过：
130根Tensor、58动态轴、一个MIX且sync_start，未提供样例shape。

生成物审阅CODEGEN_REVIEW.json通过：Left实际地址0/16384，Right 0/32768，
8次矩阵按槽0/1交替消费；当前MMA等待自己的MTE1事件，非下一槽完成事件。
Acc保留CompactMode::Normal和动态有效行，写回只覆盖真实T。L1 Mat仍复用，
不声称GM读也拥有相同双槽；K128增加MMA次数，是否划算由完整CSA判断。

## 验证

任务task_20261018_131417_2840949760，正式权重128K/B16/S6。沿用原完整
A→B→A、精确输出/TopK/score、cache/state、保护区和代际检查；功能通过后，
同进程原分task、正式SPMD、候选各取五窗完整L1并轮换次序。尚无设备结论。

source_reference/source_candidate均保留完整动态源码，用正式相同entry和
runtime接口编译，没有冻结B/S生成器。CPU命令见run_cpu.sh，真机命令见
SUBMISSION_V1.json；设备只经task-submit --device auto。当前未采用候选。

## 设备结果：功能通过，性能回退，不采用

任务exit0，完整A→B→A/精确输出、Top-K、score、cache/state、保护区、
图和代际检查通过。同轮五窗完整L1中位：原分task885.24、
正式SPMD854.66、K128双L0候选894.94μs，候选相对正式回退
4.713%。五窗均更慢，没有剔除样本。

真实双槽不等于整体收益：此版同时增加K块和MMA次数，L1 Mat仍复用，
仅有完整L1结果不能把40.28μs全部归给其中某一项。不采用、不扩七档，
不将不同轮正式SPMD的绝对时间差当成优化收益。源码和生成物保留。
