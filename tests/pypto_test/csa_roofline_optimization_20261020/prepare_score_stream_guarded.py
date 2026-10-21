# SPDX-License-Identifier: Apache-2.0
"""让WS池宽1编译回原数学体，避免未启用区间保留额外Acc别名操作。

只在两处WS乘加/写回选择单窗或列池；大段QK循环仍共用。两支使用各自
具名临时量，避免constexpr分支在前端类型检查时合并不同宽度Tile。
声明只用create，不用full初始化；是否删除未使用Tile由完整生成物核对。
"""

import ast
import json
from pathlib import Path
import shutil


def main():
    here = Path(__file__).resolve().parent
    original = here / "source_score_stream_interval"
    target = here / "source_score_stream_guarded"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = target / "indexer_score_stages.py"
    text = path.read_text()
    marker = "            split_ws_pool = pl.tile.create("
    assert text.count(marker) == 1
    text = text.replace(marker, """            split_ws_acc = pl.tile.create(
                [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc,
            )
""" + marker)
    start = text.index("                        # 上一面板WS写入父池")
    end = text.index("                    else:\n                        split_qk0 =", start)
    pooled = text[start:end]
    single = '''                        else:
                            single_ws_first = pl.tile.matmul_acc(
                                split_ws_acc, split_coefficient0, split_right0, init_cond=True,
                            )
                            single_ws_done = pl.tile.matmul_acc(
                                single_ws_first, split_coefficient1, split_right1,
                            )
                            pl.store(
                                pl.set_validshape(single_ws_done, query_group_size, qk_panel_cols),
                                [buf_transfer_row, buf_panel_col - qk_panel_cols], buf_score_transfer,
                            )
'''
    text = text[:start] + "                        if WS_POOL_PANELS > 1:\n" + "".join(
        "    " + line if line.strip() else line for line in pooled.splitlines(True)) + single + text[end:]
    start = text.index("                split_ws_last_col =")
    end = text.index("                pl.system.sync_set(SCORE_READY_EVENT", start)
    pooled = text[start:end]
    single = '''                else:
                    single_ws_first_last = pl.tile.matmul_acc(
                        split_ws_acc, split_coefficient0, split_right0_last, init_cond=True,
                    )
                    single_ws_done_last = pl.tile.matmul_acc(
                        single_ws_first_last, split_coefficient1, split_right1_last,
                    )
                    pl.store(
                        pl.set_validshape(single_ws_done_last, query_group_size, qk_panel_cols),
                        [buf_transfer_row, score_tile - qk_panel_cols], buf_score_transfer,
                    )
'''
    text = text[:start] + "                if WS_POOL_PANELS > 1:\n" + "".join(
        "    " + line if line.strip() else line for line in pooled.splitlines(True)) + single + text[end:]
    path.write_text(text)
    ast.parse(text)
    report = dict(reference=str(original), candidate=str(target), changed_files=[path.name],
                  reason="池宽1统一别名版本生成约8KiB额外C++，不能预称关闭无代价；编译期拆两处WS分支",
                  full_qk_loop_copied=False, uninitialized_create_only=True,
                  active_ir_equivalence_pending=True, disabled_ir_equivalence_pending=True)
    (here / "SCORE_STREAM_GUARDED_SOURCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
