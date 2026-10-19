# SPDX-License-Identifier: Apache-2.0
"""为提前 FIX 的 WS 保留原生 Acc 句柄，避免与尚在写出的 QK 结果同址。"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    source_dir = HERE / "source_score_fix_early"
    target = HERE / "source_score_fix_acc"
    target.mkdir(exist_ok=False)
    for path in source_dir.glob("*.py"):
        (target / path.name).write_bytes(path.read_bytes())
    path = target / "indexer_score_stages.py"
    source = path.read_text()
    marker = "            # 双槽最多允许两份未消费分数；第2轮起等两个AIV的MTE2 ACK后才重用。\n"
    assert source.count(marker) == 1
    source = source.replace(marker, """            # WS仅需16×N的FP32 Acc，N64时4KiB。句柄在本叶所有面板保持
            # 活跃，要求与QK的96KiB Acc分离；init_cond=True逐次清初始化，
            # 不把上一面板结果累加进来。实际能否分离由最终地址审计决定。
            buf_ws_acc = pl.tile.create(
                [NATIVE_QLI_WEIGHT_ROWS, qk_panel_cols], dtype=pl.FP32,
                target_memory=pl.MemorySpace.Acc,
            )
""" + marker)
    old = "buf_reduced_pair = pl.tile.matmul(buf_coefficient_pair, buf_previous_pair)"
    assert source.count(old) == 1
    source = source.replace(old, "buf_reduced_pair = pl.tile.matmul_acc(buf_ws_acc, buf_coefficient_pair, buf_previous_pair, init_cond=True)")
    old = "buf_last_scores = pl.tile.matmul(buf_coefficient_pair, buf_last_pair)"
    assert source.count(old) == 1
    source = source.replace(old, "buf_last_scores = pl.tile.matmul_acc(buf_ws_acc, buf_coefficient_pair, buf_last_pair, init_cond=True)")
    marker = "    此版本只在独立测试包中验证；未取得完整设备结果前不进入正式入口。"
    assert source.count(marker) == 1
    source = source.replace(marker, """    v1实际将QK和WS复用Acc0，在FIX后插入等待，未形成目标重叠，未上卡。
    v2用已有matmul_acc(init_cond=True)绑定本叶独立WS句柄；每次重新初始化，
    保持原矩阵乘语义。这是算子资源表达，未改编译器或硬件容量。
""" + marker)
    ast.parse(source)
    # 把指定的初始化 matmul_acc 规范回 matmul 后，数学调用须与v1相同。
    class Normalize(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "matmul_acc"
                    and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "buf_ws_acc"):
                assert len(node.keywords) == 1 and node.keywords[0].arg == "init_cond"
                assert isinstance(node.keywords[0].value, ast.Constant) and node.keywords[0].value.value is True
                node.func.attr = "matmul"
                node.args = node.args[1:]
                node.keywords = []
            return node
    before = ast.parse(path.read_text())
    after = Normalize().visit(ast.parse(source))
    mathematical = lambda t: sorted(ast.dump(n, include_attributes=False) for n in ast.walk(t)
                                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                  and n.func.attr in ("matmul", "matmul_acc", "assemble", "store"))
    assert mathematical(before) == mathematical(after)
    path.write_text(source)
    report = dict(formal_source_changed=False, device_validated=False,
                  init_true_matmul_semantics_equal=True, requested_ws_acc_bytes=16 * 64 * 4,
                  note="独立地址和有效重叠仍须CPU及真机验证")
    (HERE / "SCORE_FIX_ACC_PREPARE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
