"""仅 CPU 的源码及本地环槽协议审计，不声称覆盖 A3 执行或性能。

基线由本 worktree 的 Git 对象读取；候选仅允许 Attention AIV 的 mi/li 同核
缓存变化。核对全部原算术 RHS、GM 写出和同步调用不变，再穷举三 query 的
所有有效块组合。槽标签模拟的是生命周期，不模拟 FP32 舍入或 DMA 完成。
"""
import ast
import collections
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/attention_stream_stages.py"
NAME = "attention_qk_pv_vector_lane_stage"

def function(source):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == NAME)

def assignments(fn):
    result = collections.defaultdict(list)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                result[ast.dump(target)].append(ast.dump(node.value))
    return result

def calls(fn, names):
    return collections.Counter(ast.dump(n) for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) in names)

def main():
    before = subprocess.check_output(["git", "show", "593c5d25:" + SOURCE], cwd=ROOT, text=True)
    after = (ROOT / SOURCE).read_text()
    old, new = function(before), function(after)
    old_assign, new_assign = assignments(old), assignments(new)
    excluded = {ast.dump(ast.Name(id=n, ctx=ast.Store())) for n in ("pv_m", "pv_l")}
    preserved = []
    for target, values in old_assign.items():
        # 原循环四值 yield 扩为五值，额外一个值只有局部缓存，不是数学。
        if target in excluded or "id='running_m'" in target:
            continue
        assert values == new_assign[target], target
        preserved.append(target)
    assert ast.dump(old.args) == ast.dump(new.args)
    assert calls(old, {"pl.store"}) == calls(new, {"pl.store"})
    assert calls(old, {"pl.system.sync_set", "pl.system.sync_wait", "pl.system.set_ffts"}) == calls(new, {"pl.system.sync_set", "pl.system.sync_wait", "pl.system.set_ffts"})
    events = 0
    masks = 0
    longest = 0
    # 工作量只取 0/1/2/3 query，穷举可覆盖跨 query、跨槽及最后 drain；
    # 每个 query 五块，任意失效块组合均纳入，不假设“只可能尾块无效”。
    for queries in range(4):
        blocks = queries * 5
        for bits in range(1 << blocks):
            ring = [None] * 3
            consumed = set()
            published = set()
            for tick in range(blocks + 3):
                soft = tick - 1
                if 0 <= soft < blocks and (bits >> soft) & 1:
                    previous = ring[soft % 3]
                    assert previous is None or previous in consumed
                    ring[soft % 3] = soft
                    published.add(soft)
                merge = tick - 3
                if 0 <= merge < blocks and (bits >> merge) & 1:
                    assert ring[merge % 3] == merge, (queries, bits, tick)
                    consumed.add(merge)
                    events += 1
            assert consumed == published
            masks += 1
            longest = max(longest, blocks)
    result = dict(
        passed=True,
        baseline="593c5d25",
        candidate=SOURCE,
        original_assignment_targets_preserved=len(preserved),
        replaced_rhs_only=["pv_m: GM load -> local FP32 slice/reshape", "pv_l: GM load -> local FP32 slice/reshape"],
        signature_identical=True,
        all_gm_stores_identical=True,
        all_sync_calls_identical=True,
        mask_cases=masks,
        valid_consume_events=events,
        max_blocks_per_mask=longest,
        ring_logical_bytes_per_aiv=3 * 32 * 4 * 2,
        stage_relation="softmax(work) at tick work+1; consume at work+3; next same-slot write at work+4",
        limitations=["不模拟 A3 FP32/UB 实际物理布局", "不验证编译器寄存器/UB 别名", "不替代完整精度、graph、保护区与 L1 实测"],
    )
    (Path(__file__).parent / "source_protocol_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
