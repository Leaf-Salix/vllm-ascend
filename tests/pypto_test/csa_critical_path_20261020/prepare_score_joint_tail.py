# SPDX-License-Identifier: Apache-2.0
"""生成当前双QK/列池下生产与消费一致的动态短尾候选，不改正式算子。"""

import ast
import json
from pathlib import Path
import re
import shutil
import textwrap

HERE = Path(__file__).resolve().parent


def once(text, old, new):
    assert text.count(old) == 1, (old[:160], text.count(old))
    return text.replace(old, new)


def main():
    output = HERE / "score_joint_tail"
    source = output / "source_candidate"
    source.mkdir(parents=True, exist_ok=False)
    for path in (HERE / "score_split_pool/source_candidate").glob("*.py"):
        shutil.copyfile(path, source / path.name)
    path = source / "indexer_score_stages.py"
    text = path.read_text()
    begin = text.index("def _indexer_score_cube_split_store(")
    first = text.index("                # 固定panel顺序：", begin)
    last = text.index("                pl.system.sync_set(SCORE_READY_EVENT", first)
    original = text[first:last]
    short = '''                # 仅本轮有效候选<=N64时缩短生产；条件来自device叶计划，
                # 不依赖验收batch或固定历史。第一面板的key已按原页保护预取。
                # QK两半及WS两段K192保持，六个query物理行不能随live缩小。
                if key_prefetch_panels > 0 and key_prefetch_to_l0 and buf_valid_count - buf_score_begin <= qk_panel_cols:
                    tail_qk0 = pl.tile.matmul_acc(
                        split_qk_acc0, split_query0, buf_current_key, init_cond=True,
                    )
                    split_previous0 = pl.tile.assemble(
                        split_previous0, tail_qk0, [0, 0],
                        pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True,
                    )
                    tail_qk1 = pl.tile.matmul_acc(
                        split_qk_acc1, split_query1, buf_current_key, init_cond=True,
                    )
                    split_previous1 = pl.tile.assemble(
                        split_previous1, tail_qk1, [0, 0],
                        pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True,
                    )
                    tail_right0 = pl.tile.extract(
                        split_previous0, 0, 0, [SPLIT_QK_ROWS, qk_panel_cols],
                        target_memory=pl.MemorySpace.Right,
                    )
                    tail_right1 = pl.tile.extract(
                        split_previous1, 0, 0, [SPLIT_QK_ROWS, qk_panel_cols],
                        target_memory=pl.MemorySpace.Right,
                    )
                    tail_ws_slot = pl.tile.slice(
                        split_ws_pool, [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], [0, 0],
                    )
                    tail_ws_first = pl.tile.matmul_acc(
                        tail_ws_slot, split_coefficient0, tail_right0, init_cond=True,
                    )
                    tail_ws_done = pl.tile.matmul_acc(
                        tail_ws_first, split_coefficient1, tail_right1,
                    )
                    # 只写FIFO已计算的前64列；AIV同步采用相同条件，绝不读取
                    # 剩余960列。下一步重用前仍走原ACK，leaf末仍排空事件。
                    tail_ws_store = pl.tile.slice(
                        tail_ws_done, [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], [0, 0],
                        valid_shape=[query_group_size, qk_panel_cols],
                    )
                    pl.store(tail_ws_store, [buf_transfer_row, 0], buf_score_transfer)
                    split_ws_pool = pl.tile.assemble(split_ws_pool, tail_ws_done, [0, 0])
                else:
'''
    text = text[:first] + short + textwrap.indent(original, "    ") + text[last:]
    vbegin = text.index("def indexer_score_vector_lane_stage(")
    text = text[:vbegin] + once(text[vbegin:],
        "    SCORE_CONSUMED_EVENT: pl.constexpr,\n):",
        "    SCORE_CONSUMED_EVENT: pl.constexpr,\n    SPLIT_SCORE_ROWS: pl.constexpr = False,\n):")
    first = text.index("                    # A3没有INT8常量填充。", vbegin)
    last = text.index("                    if single_step % 2 == 1", first)
    original = text[first:last]
    # 片上形状缩至64，叶和step的逻辑跨度仍为1024；合并scale读取仍保留。
    small = original.replace("single_step * (score_tile // BLOCK_SIZE)", "__LOGICAL_PAGE_STEP__")
    small = small.replace("score_tile", "qk_panel_cols")
    small = small.replace("__LOGICAL_PAGE_STEP__", "single_step * (score_tile // BLOCK_SIZE)")
    tree = ast.parse(textwrap.dedent(small))
    locals_ = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
               and isinstance(n.ctx, ast.Store)} - {"single_segments"}
    # 两种Tile宽度不合并成同一个SSA局部，只有已初始化父segments跨分支。
    small = re.sub(r"\b\w+\b", lambda m: "tail_" + m[0] if m[0] in locals_ else m[0], small)
    # TGATHERB的UINT32索引行要求8列对齐。短尾只有至多两页，直接交换
    # 两个32元素FP16片段即可；一页时保持原fillpad结果，不引入假页读取。
    reverse_begin = small.index("                        if single_contiguous < 0:\n                            # MTE2")
    reverse_end = small.index("                        tail_single_scale_bytes =", reverse_begin)
    small = small[:reverse_begin] + '''                        if single_contiguous < 0 and tail_run_pages == 2:
                            tail_reverse_first = pl.tile.extract(
                                tail_run_flat, 0, BLOCK_SIZE, [1, BLOCK_SIZE], target_memory=pl.MemorySpace.Vec,
                            )
                            tail_reverse_second = pl.tile.extract(
                                tail_run_flat, 0, 0, [1, BLOCK_SIZE], target_memory=pl.MemorySpace.Vec,
                            )
                            # A3 Vec assemble按完整物理列宽搬运；用两行页布局，
                            # 不向一行64宽Tile窄列插入32元素。
                            tail_reverse = pl.tile.create([2, BLOCK_SIZE], dtype=pl.FP16)
                            tail_reverse = pl.tile.assemble(tail_reverse, tail_reverse_first, [0, 0])
                            tail_reverse = pl.tile.assemble(tail_reverse, tail_reverse_second, [1, 0])
                            tail_run_flat = pl.fillpad(pl.reshape(tail_reverse, [1, qk_panel_cols]), pad_value=pl.PadValue.zero)
''' + small[reverse_end:]
    small = once(small, '''                        single_segments = pl.tile.assemble(
                            single_segments, tail_single_scaled, [tail_single_local * 2 + single_step % 2, 0]
                        )''', '''                        # 窄尾前缀在同一UB父缓冲上解释成N64整行视图；避免A3
                        # 不支持的窄列assemble。reshape只变描述，不新增GM或拷贝。
                        tail_segments_rows = pl.reshape(
                            single_segments, [query_group_size * (BUFFERED_LONG_SCORE_TILE // qk_panel_cols), qk_panel_cols],
                        )
                        tail_segments_rows = pl.tile.assemble(
                            tail_segments_rows, tail_single_scaled,
                            [(tail_single_local * 2 + single_step % 2) * (BUFFERED_LONG_SCORE_TILE // qk_panel_cols), 0],
                        )
                        single_segments = pl.reshape(tail_segments_rows, [query_group_size, BUFFERED_LONG_SCORE_TILE])''')
    guard = '''                    # 两AIV和Cube使用同一资源条件与device有效计数。
                    # 仅加载已写前缀，scale/乘法也缩小；排序与逐query裁剪完全沿旧路径。
                    # 连续页仍用二维搬运；step物理偏移保持1024列而非64列。
                    if SPLIT_SCORE_ROWS and query_group_size == 6 and qk_panel_cols == 64 and score_tile == 1024 and key_prefetch_panels > 0 and key_prefetch_to_l0 and single_valid - single_step * score_tile <= qk_panel_cols:
'''
    text = text[:first] + guard + textwrap.indent(small, "    ") + "                    else:\n" + textwrap.indent(original, "    ") + text[last:]
    text = text.replace('"""动态CSA Indexer的成对Cube/Vector score流水：请求内分组、两槽READY/ACK。',
        '"""动态CSA Indexer短尾候选：保持双QK/列池，Cube和AIV共同裁剪无效尾列。\n'
        '\n    实验总览：沿现有长G6区间，在device当前step有效候选为1..64时只生产\n'
        '    和消费前64列。其余step保持列池完整1024列路径；所有query数学顺序、\n'
        '    页保护、排序同分规则、FIFO与READY/ACK计数不变。尚须完整设备验证。\n')
    ast.parse(text)
    path.write_text(text)
    path = source / "vector_schedule.py"
    text = once(path.read_text(),
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n):",
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n    SPLIT_SCORE_ROWS: pl.constexpr = False,\n):")
    text = once(text, "        SCALE_PAGE_COALESCING,\n        0,\n        1,\n    )",
        "        SCALE_PAGE_COALESCING,\n        0,\n        1,\n        SPLIT_SCORE_ROWS,\n    )")
    path.write_text(text)
    path = source / "entry.py"
    text = once(path.read_text(), "            SCALE_PAGE_COALESCING,\n            PLAN_PHYSICAL_ROWS,\n        )",
        "            SCALE_PAGE_COALESCING,\n            PLAN_PHYSICAL_ROWS,\n            SPLIT_SCORE_ROWS,\n        )")
    path.write_text(text)
    for batch in (16, 28):
        shutil.copyfile(HERE / "score_split_pool" / f"controls_b{batch}.json", output / f"controls_b{batch}.json")
    for path in source.glob("*.py"):
        ast.parse(path.read_text())
    report = {"reference": "source_after_score_split/13259f92", "base_candidate": "score_split_pool/source_candidate",
              "files": ["entry.py", "vector_schedule.py", "indexer_score_stages.py"],
              "condition": "existing split G6/N64/1024 + prefetch_to_l0 + device remaining candidates in 1..64",
              "host_inputs_added": False, "device_or_performance_proven": False,
              "expected_tail_qk_panels": 1, "expected_full_qk_panels": 16,
              "fifo_width": 1024, "tail_written_and_loaded_width": 64,
              "unchanged": ["query/head arithmetic", "sort/tie order", "page guards", "READY/ACK/drain"]}
    (output / "PREPARATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
