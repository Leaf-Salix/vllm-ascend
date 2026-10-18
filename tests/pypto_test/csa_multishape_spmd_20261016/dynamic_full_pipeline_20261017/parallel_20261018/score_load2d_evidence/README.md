# Score WS 的 load2d 搬运消融

根据相同真实中间输入的Score PMU，MTE1三步活跃时间中位154.814μs、矩阵
103.951μs。两者可重叠；不以相减宣称可省耗时。现有Key双槽、提前一面板
到L0B及QK/前面板WS交错均已存在。本轮只改变WS的FP16 NZ→L0B搬运指令。

## 候选及原理

原PTO-ISA TEXTRACT在该布局调用img2colv2_cbuf_to_cb；候选把每个16×16分形
通过load_cbuf_to_cb_transpose搬到相同L0B地址。源NZ分形编号n1*K1+k1，
目标ZN编号k1*N1+n1，每个n1发一条、重复K1次，目标间隔N1-1。物理容量、
数据类型、数学、舍入、valid范围、所有事件及原双缓冲地址全部保持。

修改位于测试侧load2d_hook.py，只接受完整零偏移FP16 WS，限定G6/N64/1024
的本次代表诊断；没有把此限制写入正式动态算子支持范围。回退调用并移除helper
之后，实际C++与原文逐字符一致。没有改共享安装或依赖仓库。

CPU初次准备因两个核函数使用相同v编号误计17个站点，准备即失败，无设备运行；
首次CCEC包装随后因未生成源文件退出。修正为在AIC函数范围匹配，且兼容原
AICORE与wrapper的__aicore__拼写。第二次实际CCEC编译cpu_v2通过，16站点
全部替换；没有用CPU编译通过代替布局数值验证。

## 设备验证

任务task_20261018_130455_238129612172经自动队列提交。先完成正式权重完整CSA
A→B→A，再以相同中间Tensor运行原load3d和候选load2d Score。两侧各自对
完整pair/score allocation做eager/graph/profiling后三次逐bit比较；各采3个
PyTorch PMU step。图外Event包括调度，只作筛选；两组阶段采样未交错，
也不是完整CSA L1性能验收。设备已完成，结果见下。

复现：`bash run_device.sh <全新输出目录> --batch 16 --history 131072 --launches 3`，
必须通过task-submit --device auto运行。当前正式动态算法没有变化。

## 结果：数值通过，收益不足，不迁入正式实现

任务exit0，原load3d与候选load2d在eager、graph、profiling后三次完整pair/score
均逐bit一致。原/新独立Score Event中位329.790/328.700μs，候选单次392.540μs
慢样本保留；这不是交错配对完整L1。三个PMU kernel Duration中位287.120/
286.780μs，MTE1活跃154.691/152.870μs，矩阵103.910/103.793μs。

搬运指令等价验证通过，但没有足以支持完整CSA明显收益的结果。不修改正式
算子、不改依赖、不追加完整CSA或七档重测；保留该路径作为已验证的小收益/
无整体收益先例。原始所有样本、CSV及PMU_COMPARISON.json完整保留。
