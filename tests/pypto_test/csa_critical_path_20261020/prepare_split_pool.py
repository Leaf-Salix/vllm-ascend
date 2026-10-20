# SPDX-License-Identifier: Apache-2.0
"""在当前双QK早FIX数学中合并WS写回；不叠加旧单QK或宽N候选。"""

import ast
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "score_split_pool"
    source = output / "source_candidate"
    source.mkdir(parents=True, exist_ok=False)
    for path in (HERE / "source_after_score_split").glob("*.py"):
        shutil.copyfile(path, source / path.name)
    path = source / "indexer_score_stages.py"
    text = path.read_text()
    tree = ast.parse(text)
    fn, = [n for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name == "_indexer_score_cube_split_store"]
    lines = text.splitlines(keepends=True)
    start = fn.lineno - 1
    body = "".join(lines[start:fn.end_lineno])
    old = '''            split_ws_acc = pl.tile.create(
                [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc,
            )'''
    assert body.count(old) == 1
    body = body.replace(old, '''            # 两份48KiB QK仍独立；WS由4KiB扩成八个列窗的32KiB父池，
            # 合计恰128KiB。每列窗先init再acc，凑满八窗才写GM，不读旧数据。
            split_ws_pool = pl.tile.create(
                [NATIVE_QLI_WEIGHT_ROWS, SPLIT_WS_POOL_PANELS * qk_panel_cols],
                dtype=pl.FP32, target_memory=pl.MemorySpace.Acc,
            )''')
    old = '''                        split_ws_first = pl.tile.matmul_acc(
                            split_ws_acc, split_coefficient0, split_right0, init_cond=True,
                        )'''
    assert body.count(old) == 1
    body = body.replace(old, '''                        # 上一面板WS写入父池的一个列窗；列窗是地址别名，不搬Acc。
                        # 两段K192保持原累加次序和一次初始化，未交换系数或FP16边界。
                        split_ws_col = pl.max(buf_qk_panel - 1, 0) % SPLIT_WS_POOL_PANELS * qk_panel_cols
                        split_ws_slot = pl.tile.slice(
                            split_ws_pool, [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], [0, split_ws_col],
                        )
                        split_ws_first = pl.tile.matmul_acc(
                            split_ws_slot, split_coefficient0, split_right0, init_cond=True,
                        )''')
    old = '''                        # 每panel仅一次原六行写回，保留1024列FIFO布局和顺序。
                        pl.store(
                            pl.set_validshape(split_ws_done, query_group_size, qk_panel_cols),
                            [buf_transfer_row, buf_panel_col - qk_panel_cols], buf_score_transfer,
                        )'''
    assert body.count(old) == 1
    body = body.replace(old, '''                        split_ws_pool = pl.tile.assemble(split_ws_pool, split_ws_done, [0, split_ws_col])
                        if buf_qk_panel % SPLIT_WS_POOL_PANELS == 0:
                            # 当前QK8已提交，WS0..7完整后写原FIFO的前512列。
                            # 下一轮覆盖pool列0仍须等本次FIX读完；不删除这条真依赖。
                            # max仅让未执行的展开分支偏移合法，真实写回时偏移本已非负。
                            pl.store(
                                pl.set_validshape(split_ws_pool, query_group_size, SPLIT_WS_POOL_PANELS * qk_panel_cols),
                                [buf_transfer_row, pl.max(buf_panel_col - SPLIT_WS_POOL_PANELS * qk_panel_cols, 0)],
                                buf_score_transfer,
                            )''')
    old = '''                split_ws_first_last = pl.tile.matmul_acc(
                    split_ws_acc, split_coefficient0, split_right0_last, init_cond=True,
                )'''
    assert body.count(old) == 1
    body = body.replace(old, '''                split_ws_last_col = (SPLIT_WS_POOL_PANELS - 1) * qk_panel_cols
                split_ws_last_slot = pl.tile.slice(
                    split_ws_pool, [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], [0, split_ws_last_col],
                )
                split_ws_first_last = pl.tile.matmul_acc(
                    split_ws_last_slot, split_coefficient0, split_right0_last, init_cond=True,
                )''')
    old = '''                pl.store(
                    pl.set_validshape(split_ws_done_last, query_group_size, qk_panel_cols),
                    [buf_transfer_row, score_tile - qk_panel_cols], buf_score_transfer,
                )'''
    assert body.count(old) == 1
    body = body.replace(old, '''                split_ws_pool = pl.tile.assemble(split_ws_pool, split_ws_done_last, [0, split_ws_last_col])
                # WS8..15全部覆盖后写后512列，随后才沿原协议发布整step READY。
                pl.store(
                    pl.set_validshape(split_ws_pool, query_group_size, SPLIT_WS_POOL_PANELS * qk_panel_cols),
                    [buf_transfer_row, score_tile - SPLIT_WS_POOL_PANELS * qk_panel_cols], buf_score_transfer,
                )''')
    body = body.replace('长G6/N64：两半QK早FIX；原完整16行系数分K累加后一次写回。',
                        '长G6/N64：保留两半QK早FIX，八个WS列窗合并一次原布局写回。')
    body = body.replace('QK两份48KiB Acc及WS一份4KiB跨panel保留，各次init_cond=True重新初始化。',
                        'QK两份48KiB Acc及WS32KiB列池跨panel保留，各列窗init_cond=True初始化。')
    body = body.replace('两次WS按原K顺序累计到同一Acc，不能把两个三query结果分别写回。',
                        '两次WS按原K顺序累计到同一列窗，不能拆成两个三query结果或重排数学。')
    text = "".join(lines[:start]) + body + "".join(lines[fn.end_lineno:])
    # 私有helper由原wrapper用固定G6/N64/1024调用，常量不按B/S验收点选择。
    marker = "@pl.jit.inline(auto_scope=False)\ndef _indexer_score_cube_split_store"
    assert text.count(marker) == 1
    text = text.replace(marker, "# 八个N64结果占32KiB Acc，与双48KiB QK并存；非新的shape白名单。\n"
                        "SPLIT_WS_POOL_PANELS = 8\n\n\n" + marker)
    ast.parse(text)
    path.write_text(text)
    for batch in (16, 28):
        shutil.copyfile(HERE / "wave_current_math" / f"controls_b{batch}.json",
                        output / f"controls_b{batch}.json")
    report = {"reference": "source_after_score_split/13259f92",
              "source": str(source), "modified_files": [path.name],
              "qk_acc_bytes_each": 49152, "qk_acc_count": 2, "ws_pool_bytes": 32768,
              "expected_acc_total": 131072, "ws_stores_per_1024_before": 16,
              "ws_stores_per_1024_expected": 2, "device_or_performance_proven": False}
    (output / "PREPARATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
