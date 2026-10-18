### 问题与影响

MemoryReuse 对互斥分支的活跃区间建模过于保守。在融合的单 SPMD 中，两类合法代码会触发 Vec 假超限，而逐个缩小数学 tile 只是绕过问题：

1. 分支内 `gather_row` 原地循环的最后一个别名进入 phi family，但同一分支、同一 allocation 的其它别名没有全部进入，导致 sibling 分支间伪干涉。
2. then→else 的线性遍历把“只在 else 继续读取的旧值”与“仅 then 创建的替代值”当作同时存活。两者实际互斥，但 allocator 为它们保留独立存储。

本次主线为 `14736f65c04a7db06547edb656f515054950555c`。复现环境已经带 #3027/#3028 的 view 与 While 修正，因此这里不是缺少这两个修正的重复报告；下述分支容量例子仍失败。局部修正不放宽平台容量，不移除同分支后续读、循环回边或外部独立观察的约束。

### 无模型、无设备的复现

保存以下文件，分别执行 `python repro.py gather` 与 `python repro.py reset`。A3 backend 的 Vec 可用容量为 188416 B：

```python
import pypto.language as pl
from pypto import backend, passes
from pypto.backend import BackendType
backend.reset_for_testing()
backend.set_backend_type(BackendType.Ascend910B)

@pl.program
class BranchGather:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self,
        pool: pl.Tensor[[64, 512], pl.BF16],
        flags: pl.Tensor[[1], pl.INT32],
        out: pl.Out[pl.Tensor[[64, 512], pl.BF16]],
        state: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
    ) -> tuple[pl.Tensor[[64, 512], pl.BF16], pl.Tensor[[64, 256], pl.FP32]]:
        first = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        second = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        flag = pl.tensor.read(flags, [0])
        for i, (s0, s1) in pl.range(2, init_values=(first, second)):
            seed = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
            if flag != 0:
                filled = pl.tile.gather_row(seed, pool, [0, 0], [0, 0], [64, 512])
                selected = filled
            else:
                fresh = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                for j, (carry,) in pl.range(64, init_values=(fresh,)):
                    if j < flag:
                        next_value = pl.tile.gather_row(
                            carry, pool, [j, 0], [j, 0], [1, 512]
                        )
                    else:
                        next_value = carry
                    gathered = pl.yield_(next_value)
                selected = gathered
            saved = pl.tile.store(selected, [0, 0], out)  # noqa: F841 - observe the selected tile
            next0 = pl.tile.add(s0, 1.0)
            next1 = pl.tile.add(s1, 2.0)
            result0, result1 = pl.yield_(next0, next1)
        state0 = pl.tile.store(result0, [0, 0], state)
        state1 = pl.tile.store(result1, [32, 0], state0)
        return out, state1

@pl.program
class ConditionalReset:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self,
        pool: pl.Tensor[[128, 256], pl.FP32],
        flags: pl.Tensor[[1], pl.INT32],
        out: pl.Out[pl.Tensor[[128, 256], pl.FP32]],
    ) -> pl.Tensor[[128, 256], pl.FP32]:
        held = pl.tile.load(pool, [64, 0], [64, 256], target_memory=pl.Mem.Vec)
        current = pl.tile.load(pool, [0, 0], [64, 256], target_memory=pl.Mem.Vec)
        flag = pl.tensor.read(flags, [0])
        if flag != 0:
            reset = pl.tile.full([64, 256], dtype=pl.FP32, value=0.0)
            selected = reset
        else:
            selected = current
        saved = pl.tile.store(selected, [0, 0], out)
        result = pl.tile.store(held, [64, 0], saved)
        return result

if __name__ == "__main__":
    import sys
    p = BranchGather if sys.argv[1] == "gather" else ConditionalReset
    p = passes.convert_to_ssa()(p)
    p = passes.init_mem_ref()(p)
    p = passes.materialize_semantic_aliases()(p)
    p = passes.memory_reuse()(p)
    passes.allocate_memory_addr()(p)
```

原实现在上述独立脚本中两例均报告 `Vec buffer usage (196608 bytes) exceeds platform limit (188416 bytes)`；pytest fixture 下同类两例为 262144 B，两个上下文的具体分配差异本次未单独归因；两者均超过 188416 B 上限。重建正确的互斥关系后两例均可编译；本地仍保留“同分支写后还要读旧值”和“下一轮还需旧值”的负例，二者不得误复用。

### 修改建议

建议拆成两项可独立审查的改进：

- 收集每个 allocation 的定义点，并按定义顺序建立索引；只给实际定义在该分支区间内的同 base 别名打上该 phi family。不能把分支外 seed 或仍独立活跃的值一起标成互斥。
- 仅在原布局超限时做一次有界的反向 sibling 遍历，用同一套生命周期规则检查原顺序产生的假干涉；每个分支内部、if 后续使用及循环回边顺序保持。不要全局把所有分支值当成互斥，不因优化丢失 runtime pipeline/hazard 限制。

局部实现的关键逻辑如下（审查思路，不是可直接整文件应用的补丁）：

```cpp
// 每个分支各自记录 [start, end]；按 allocation 索引范围内定义。
auto begin = std::lower_bound(group.begin(), group.end(), start,
    [&](const VarPtr& var, int order) { return var_def_order_.at(var) < order; });
for (auto member = begin; member != group.end() && var_def_order_.at(*member) <= end; ++member) {
    var_family_ids_[member->get()].insert(family_id);
}

// 只在容量压力下细化 sibling 顺序；最多建立一次反向生命周期索引。
if (cap != 0 && footprint(buffers) > cap && !phi_family_ids.empty()) {
    refine_branch_order = true;
    auto candidate = pack(/*allow_subranges=*/false);
    if (footprint(candidate) <= footprint(buffers)) buffers = std::move(candidate);
    else refine_branch_order = false;
}
```

实际建议还需保留 GetMemRef 的 base/offset 语义、循环 recurrence、同分支负例。新增扫描应是固定次数，避免对每对值重复扫描整个 IR。

### 验证边界

本次主线派生环境的定向 11 例为 5 失败 / 6 通过，其中 3 例属于上述分支容量范围；另两个失败分别属于独立 arena 需求 #1908 和 JIT reshape 问题。局部修正的分支类及相关 JIT 共 9 例全部通过；arena 回归在另外 65 例检查内通过。旧别名始终需读的反例保持失败/隔离要求。

这里报告编译和资源规划问题，不宣称整层性能收益。仅提 issue 和修改建议，不创建 PR。
