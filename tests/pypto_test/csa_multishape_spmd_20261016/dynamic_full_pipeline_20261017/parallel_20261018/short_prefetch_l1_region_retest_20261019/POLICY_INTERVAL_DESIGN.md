# 受资源边界约束的短历史预取候选（尚未采用）

当前G2三个8K场景受益，G6相邻分组和G2历史上沿不支持全区间开启。
现有非balanced Score每步交接768列，两AIV每条lane每步384列；每个
半叶随后调用indexer_topk_half_leaf。该函数对<=2048候选直接归并，
更大范围转入2560/3072/4096分支，UB工作量和归并安排不同。

Host已有max_seq_len提供每请求长度上界，visible又受length//4钳制，
因此不用新增device取值即可保守估计最大半叶跨度：

```python
cache_rows = max_seq_len // 4
score_steps_upper = (cache_rows + 767) // 768
half_span_upper = score_steps_upper * 384
short_prefetch = not long_history and group == 2 and half_span_upper <= 2048
```

当前短区间cache_rows<=8192，仅一叶；不需要加入4096钳制来判断<=2048。
这等价于最多5步、最多3840压缩行。S1/S5的query数仍在device影响实际
组和尾组，Host只取已有上界，不能用这个估算代替实际循环/边界检查。

该条件的意义是限定既有排序资源区间，不是已经证明TopK导致上沿回退。
两侧H15352/H15360真机仍在执行；不将公式、CPU结果或两个单点的趋势
当作整个区间必然提速。若边界内也不支持收益，保持候选，避免继续造
孤立测试点规则。长历史原prefetch=1不变，短G6维持0。

## 终态

两侧完整通过；1920半叶上界降低3.160%，2304上界回退0.758%。
已采用上述区间，正式改动仅policy.py，详见README和ADOPTION_AUDIT。
此结果支持当前保守选择，不将分界等同于已证明的瓶颈成因。
