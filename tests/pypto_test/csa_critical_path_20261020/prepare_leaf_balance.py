# SPDX-License-Identifier: Apache-2.0
"""对长历史的较多query组对齐叶数并轮转叶owner；不改数学tile或Host metadata。"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new)


def main():
    base = HERE / "source_score_fix_acc"
    target = HERE / "source_leaf_balanced"
    target.mkdir(exist_ok=False)
    for path in base.glob("*.py"):
        (target / path.name).write_bytes(path.read_bytes())
    path = target / "indexer_group_stages.py"
    source = path.read_text()
    source = once(source, "整体流程与阅读顺序\n", """本候选的整体变化
    较多query组时，把原最小叶数向上对齐到24核组数的最小可用约数，再将
    实际1024候选块均匀分配。每个核轮转领取同组的长短叶，避免固定余数
    总拿到长叶。小group范围保持原计划；叶数容量及每叶至多8块保持。
    group/历史来自本轮device计划，未按B28或任何验收点写分支。
    logical leaf决定连续候选范围及Top-K根地址；physical leaf只决定分核。
    正向分配和Top-K反向找writer使用同一rotation，不能只改生产者。
    不新增task、Tensor根、Host回读、FFTS或GM信号；精度/收益待完整验证。

整体流程与阅读顺序
""")
    source = once(source, "    return leaf_count, leaf_tiles, extra_leaves\n", """    # 较多query组原先保留8/8/.../极小尾叶，导致同一核的Score步数偏多。
    # 选择24的最小可用约数，让group-major的整轮能覆盖完整叶集合。
    # 这里只做device Scalar计划，保留1024候选粒度和原arena容量。
    if query_groups >= SCORE_WORKERS and leaf_count <= SCORE_WORKERS:
        aligned_leaves = pl.cast(SCORE_WORKERS, pl.INDEX)
        for candidate_leaves in pl.unroll(1, SCORE_WORKERS + 1):
            if SCORE_WORKERS % candidate_leaves == 0:
                if candidate_leaves >= leaf_count and candidate_leaves < aligned_leaves:
                    aligned_leaves = pl.cast(candidate_leaves, pl.INDEX)
        if aligned_leaves <= tile_count and aligned_leaves <= MAX_LEAVES:
            leaf_count = aligned_leaves
            leaf_tiles = tile_count // aligned_leaves
            extra_leaves = tile_count % aligned_leaves
    return leaf_count, leaf_tiles, extra_leaves
""")
    marker = "@pl.jit.inline(auto_scope=False)\ndef indexer_work_item("
    source = once(source, marker, """@pl.jit.inline(auto_scope=False)
def indexer_group_leaf_rotation(
    group: pl.Scalar[pl.INDEX],
    groups: pl.Scalar[pl.INDEX],
    leaves: pl.Scalar[pl.INDEX],
    BALANCED: pl.constexpr,
):
    \"\"\"同一组所有叶使用同一rotation，保证仍为双射；反向等待复用此值。\"\"\"
    rotation = pl.cast(0, pl.INDEX)
    if BALANCED:
        if groups >= SCORE_WORKERS and leaves > 0 and leaves <= SCORE_WORKERS:
            if SCORE_WORKERS % leaves == 0:
                groups_per_round = SCORE_WORKERS // leaves
                rotation = (group // groups_per_round) % leaves
    return rotation


@pl.jit.inline(auto_scope=False)
def indexer_work_item(""")
    source = once(source, """    leaves: pl.Scalar[pl.INDEX],
):
    \"\"\"少组时先铺满不同组，多组时同组连续叶；空输入返回-1而不除零。\"\"\"
""", """    leaves: pl.Scalar[pl.INDEX],
    BALANCED: pl.constexpr = False,
):
    \"\"\"返回逻辑叶；多组balanced轮转长短叶，空输入不除零。\"\"\"
""")
    source = once(source, """            group = item // leaves
            leaf = item % leaves
    return group, leaf
""", """            group = item // leaves
            leaf = item % leaves
            if BALANCED:
                rotation = indexer_group_leaf_rotation(group, groups, leaves, BALANCED)
                if rotation != 0:
                    leaf = (leaf + rotation) % leaves
    return group, leaf
""")
    ast.parse(source)
    path.write_text(source)

    path = target / "indexer_score_stages.py"
    source = path.read_text()
    old = "indexer_work_item(buf_item, groups, buf_max_leaves)"
    assert source.count(old) == 2
    source = source.replace(old, "indexer_work_item(buf_item, groups, buf_max_leaves, balance_leaves)")
    ast.parse(source)
    path.write_text(source)

    path = target / "attention_dependencies.py"
    source = path.read_text()
    source = once(source, "from .indexer_group_stages import PLAN_WORDS",
                  "from .indexer_group_stages import PLAN_WORDS, indexer_group_leaf_rotation")
    source = once(source, "    每个 item 归 item%24 的 AIC 组。balanced 每个query仅由一个 AIV lane",
                  """    多组balanced还对组内叶作轮转，须反解physical leaf后再求item。
    每个 item 归 item%24 的 AIC 组。balanced 每个query仅由一个 AIV lane""")
    source = once(source, "    for root in pl.range(root_count):", """    # rotation只由本query的group与本代计划决定，放在根循环外计算一次。
    # root对应逻辑叶；Score按物理叶领取工作，必须反向选择真实writer。
    rotation = indexer_group_leaf_rotation(group, groups, leaves, BALANCED)
    for root in pl.range(root_count):""")
    source = once(source, """        else:
            item = group * leaves + leaf
        producer = GROUPS + 2 * (item % GROUPS) + lane
""", """        else:
            physical_leaf = leaf
            if BALANCED:
                if rotation != 0:
                    physical_leaf = (leaf + leaves - rotation) % leaves
            item = group * leaves + physical_leaf
        producer = GROUPS + 2 * (item % GROUPS) + lane
""")
    ast.parse(source)
    path.write_text(source)

    changed = sorted(p.name for p in base.glob("*.py") if p.read_bytes() != (target / p.name).read_bytes())
    assert changed == ["attention_dependencies.py", "indexer_group_stages.py", "indexer_score_stages.py"]
    report = dict(changed_vs_fix_acc=changed, includes_previous_fix_acc_candidate=True,
                  formal_source_changed=False, policy_uses_device_group_count=True,
                  added_host_inputs=False, device_validated=False, performance_eligible=False)
    (HERE / "LEAF_BALANCE_PREPARE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
