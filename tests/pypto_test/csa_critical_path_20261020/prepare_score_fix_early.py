# SPDX-License-Identifier: Apache-2.0
"""保持 N64 数学/分工，提前发出当前 QK 的 FIX，再执行上一面板 WS。"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    reference = HERE / "source_reference"
    target = HERE / "source_score_fix_early"
    target.mkdir(exist_ok=False)
    for path in reference.glob("*.py"):
        (target / path.name).write_bytes(path.read_bytes())
    path = target / "indexer_score_stages.py"
    source = path.read_text()
    old = """                        buf_scores = pl.tile.matmul(buf_query_left, buf_key_panel)
                        buf_reduced_pair = pl.tile.matmul(buf_coefficient_pair, buf_previous_pair)
                        buf_scores_l1 = pl.tile.assemble(
                            buf_scores_l1, buf_scores, [0, 0], pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True
                        )
"""
    new = """                        buf_scores = pl.tile.matmul(buf_query_left, buf_key_panel)
                        # 当前QK已具备FIX输入；先提交它，再执行只读取上一面板的WS。
                        # 目标是让M→FIX完成边界不被无关WS扩大。必须核对实际
                        # Acc地址：若编译器复用同址，FIX→M会重新串行，不能据
                        # 源码顺序宣称重叠。页/Key预取和WS K384数学全部保持。
                        buf_scores_l1 = pl.tile.assemble(
                            buf_scores_l1, buf_scores, [0, 0], pre_quant=1.0 / BUFFERED_SCORE_SCALE, pre_relu=True
                        )
                        buf_reduced_pair = pl.tile.matmul(buf_coefficient_pair, buf_previous_pair)
"""
    assert source.count(old) == 1
    source = source.replace(old, new)
    marker = "阅读顺序与范围\n"
    assert source.count(marker) == 1
    source = source.replace(marker, """本候选的整体变化
    仅在现有面板流水内部把当前QK的FIX提交移到上一面板WS之前，保持N64、
    INT8点积、FP16缩放/ReLU、WS K384、Key两槽预取、FFTS和AIV原数学。
    目标是让当前QK写L1可与不依赖它的上一WS计算重叠。是否确实形成重叠，
    必须检查最终Acc地址及M/FIX事件；顺序调整也可能导致更激进的地址复用。
    这与旧“把WS Right加载移到QK后”的方案不同，本版不移动任何Right加载。
    此版本只在独立测试包中验证；未取得完整设备结果前不进入正式入口。

阅读顺序与范围
""")
    # 删除文档/注释后，数学调用及参数的多重集合完全相同；只允许语句顺序变化。
    before = ast.parse(path.read_text())
    after = ast.parse(source)
    calls = lambda tree: sorted(ast.dump(n, include_attributes=False) for n in ast.walk(tree) if isinstance(n, ast.Call))
    assert calls(before) == calls(after)
    path.write_text(source)
    changed = [p.name for p in reference.glob("*.py") if p.read_bytes() != (target / p.name).read_bytes()]
    assert changed == ["indexer_score_stages.py"]
    report = dict(changed=changed, mathematical_calls_equal=True,
                  control_shape_changed=False, formal_source_changed=False,
                  device_validated=False, performance_eligible=False)
    (HERE / "SCORE_FIX_EARLY_PREPARE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
