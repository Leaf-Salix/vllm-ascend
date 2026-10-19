# SPDX-License-Identifier: Apache-2.0
"""构造N128单QK Acc、跨面板WS覆盖FIX的动态Score候选。

源自当前已采用的N64双Acc版，保留动态页/叶/有效行及六行一次写回。
此处私有helper固定数学N128，原公共QK_PANEL=64仍是选择该G6区间的条件，
不是本候选实际内层面板宽度；报告必须显式记录actual_cube_panel_cols=128。
未通过真实完整门槛前不改变正式源或声称已采用这个资源策略。
"""

import ast
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

STEADY = '''                    if buf_qk_panel > 0:
                        # 当前Key占16KiB Right；上一panel的每半WS占48KiB。
                        # 两半WS依次取入同一Right区，不同时保留两个48KiB槽。
                        split_right0 = pl.tile.extract(
                            split_previous0, 0, 0, [SPLIT_QK_ROWS, qk_panel_cols],
                            target_memory=pl.MemorySpace.Right,
                        )
                        split_qk0 = pl.tile.matmul_acc(
                            split_qk_acc, split_query0, buf_key_panel, init_cond=True,
                        )
                        # 立即FIX第一半，随后执行独立的上一panel WS0；
                        # QK1复用同一96KiB Acc所需的FIX完成等待保持不删。
                        split_current0 = pl.tile.assemble(
                            split_current0, split_qk0, [0, 0],
                            pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True,
                        )
                        split_ws_first = pl.tile.matmul_acc(
                            split_ws_acc, split_coefficient0, split_right0, init_cond=True,
                        )
                        split_right1 = pl.tile.extract(
                            split_previous1, 0, 0, [SPLIT_QK_ROWS, qk_panel_cols],
                            target_memory=pl.MemorySpace.Right,
                        )
                        split_qk1 = pl.tile.matmul_acc(
                            split_qk_acc, split_query1, buf_key_panel, init_cond=True,
                        )
                        split_current1 = pl.tile.assemble(
                            split_current1, split_qk1, [0, 0],
                            pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True,
                        )
                        # 原六行系数K0..191再192..383累加；一次六行写回，
                        # 不恢复历史N128候选的两个三query分别STORE。
                        split_ws_done = pl.tile.matmul_acc(
                            split_ws_first, split_coefficient1, split_right1,
                        )
                        pl.store(
                            pl.set_validshape(split_ws_done, query_group_size, qk_panel_cols),
                            [buf_transfer_row, buf_panel_col - qk_panel_cols], buf_score_transfer,
                        )
'''


def named_function(source, name):
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, fn)


def main():
    base = HERE / "source_after_score_split"
    target = HERE / "source_score_wide_pipeline"
    target.mkdir(exist_ok=False)
    for p in base.glob("*.py"):
        (target / p.name).write_bytes(p.read_bytes())
    path = target / "indexer_score_stages.py"
    original = path.read_text()
    source = original.replace("_indexer_score_cube_split_store", "_indexer_score_cube_wide_split_store")
    old = "SPLIT_QK_ROWS = 3 * IDX_N_HEADS  # 既有G6一半的数学M，非batch专用shape。\n"
    assert source.count(old) == 1
    source = source.replace(old, old + "WIDE_QK_PANEL_COLS = 128  # 本候选私有Cube数学面板；不冻结T/B/历史。\n")
    start = source.index("def _indexer_score_cube_wide_split_store(")
    end = source.index("\n@pl.jit.inline", start)
    body = source[start:end]
    old_acc1 = '''            split_qk_acc1 = pl.tile.create(
                [SPLIT_QK_ROWS, qk_panel_cols], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc,
            )
'''
    assert body.count(old_acc1) == 1
    body = body.replace(old_acc1, "")
    body = body.replace("split_qk_acc0", "split_qk_acc").replace("split_qk_acc1", "split_qk_acc")
    first = body.index("                    if buf_qk_panel > 0:\n")
    last = body.index("                    else:\n", first)
    body = body[:first] + STEADY + body[last:]
    # 尾部两半WS同样不能同时占用96KiB Right。第一半计算完成后才载第二半。
    tail_start = body.index("                split_right0_last = pl.tile.extract(")
    tail_end = body.index("                split_ws_done_last =", tail_start)
    body = body[:tail_start] + '''                split_right0_last = pl.tile.extract(
                    split_previous0, 0, 0, [SPLIT_QK_ROWS, qk_panel_cols],
                    target_memory=pl.MemorySpace.Right,
                )
                split_ws_first_last = pl.tile.matmul_acc(
                    split_ws_acc, split_coefficient0, split_right0_last, init_cond=True,
                )
                split_right1_last = pl.tile.extract(
                    split_previous1, 0, 0, [SPLIT_QK_ROWS, qk_panel_cols],
                    target_memory=pl.MemorySpace.Right,
                )
''' + body[tail_end:]
    body = body.replace('''    """长G6/N64：两半QK早FIX；原完整16行系数分K累加后一次写回。

    本helper保留原group/leaf/device页检查、尾query安全填充和双槽协议。
    QK两份48KiB Acc及WS一份4KiB跨panel保留，各次init_cond=True重新初始化。
    两次WS按原K顺序累计到同一Acc，不能把两个三query结果分别写回。
    是否减少等待、保持舍入和产生完整收益，分别由代码审计与真机检查决定。
    """''', '''    """长G6/N128：一份96KiB QK Acc与独立8KiB WS Acc交错。

    Query/系数Left仍共60KiB，Key Right16KiB与一份WS Right48KiB共64KiB。
    每面板执行QK0→FIX0→前面板WS0→QK1→FIX1→前面板WS1，原六行一次
    写回。QK1等待同Acc的FIX0完成，WS则使用另一份Acc形成覆盖机会。
    Key只提前到L1，不同时预取两个16KiB Right。面板数16→8，1024列
    FIFO、原动态页/叶/尾query及FFTS交接保持；首面板和末面板按原顺序排空。
    这是待验候选，容量估算和提交顺序均不代替完整精度/性能证据。
    """''')
    body = body.replace("# 第一份FIX位于第二份QK之前；两份Acc明确独立才有重叠机会。",
                        "# 首面板无前面板WS；仍先FIX，再按同Acc依赖执行下一半。")
    source = source[:start] + body + source[end:]
    # 当前SPLIT_SCORE_ROWS区间选择仍沿正式条件；只改该私有helper的资源常量。
    begin = source.index("def indexer_score_cube_stage(")
    wrapper = source[begin:]
    old_args = '''            6,
            64,
            1024,
            key_prefetch_panels,
            key_prefetch_to_l0,
            True,
'''
    assert wrapper.count(old_args) == 1
    wrapper = wrapper.replace(old_args, '''            6,
            WIDE_QK_PANEL_COLS,
            1024,
            1,
            False,
            True,
''')
    source = source[:begin] + wrapper
    doc_end = source.index('"""', source.index('"""') + 3)
    source = source[:doc_end] + '''
本轮宽面板候选总览
    仅替换已采用SPLIT_SCORE_ROWS区间的私有Cube helper：实际N128、单96KiB
    QK Acc、单8KiB WS Acc，跨面板WS覆盖当前FIX；其余数学与AIV不动。
    公共QK_PANEL=64在本候选仅保留为原区间选择条件，实际内层N128见命名
    常量WIDE_QK_PANEL_COLS及生成审计。若采用须整理正式策略描述，不能把
    本候选报告成内层N64。T/B/页表/长度/尾行仍动态，不是冻结静态entry。
''' + source[doc_end:]
    ast.parse(source)
    for name in ("_indexer_score_cube_original", "indexer_score_vector_lane_stage"):
        assert named_function(source, name) == named_function(original, name)
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("wide_score_policy", target / "policy.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    controls = {}
    for batch in (16, 28):
        controls[str(batch)] = module.select_policy(tokens=batch*6, requests=batch, max_query_tokens=6,
                                                   max_seq_len=131078).constexprs()
        assert controls[str(batch)]["SPLIT_SCORE_ROWS"]
        (HERE/f"score_wide_controls_b{batch}.json").write_text(json.dumps(controls[str(batch)],indent=2)+"\n")
    changed = [p.name for p in target.glob("*.py") if p.read_bytes() != (base/p.name).read_bytes()]
    assert changed == ["indexer_score_stages.py"]
    report = dict(base="source_after_score_split/13259f92", changed=changed,
                  actual_cube_panel_cols=128, private_key_prefetch_panels=1,
                  private_key_prefetch_to_l0=False, expected_acc_bytes=[98304,8192],
                  expected_right_bytes=[49152,16384], expected_left_bytes=61440,
                  expected_step_counts=dict(qk=16, qk_fix=16, ws_init=8, ws_acc=8, store=8),
                  original_helper_and_aiv_source_unchanged=True, controls=controls,
                  formal_changed=False, cpu_validated=False, device_validated=False)
    (HERE/'SCORE_WIDE_PREPARE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print({k:v for k,v in report.items() if k!='controls'})


if __name__ == "__main__":
    main()
