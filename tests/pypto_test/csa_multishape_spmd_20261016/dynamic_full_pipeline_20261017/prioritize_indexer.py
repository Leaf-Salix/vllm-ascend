# SPDX-License-Identifier: Apache-2.0
"""将Indexer输入链提前，利用主Query投影时间执行Indexer向量阶段。

保留所有源级数学和动态分工；只移动独立阶段以及相应获取依赖。没有历史
shape、固定分核表或根据测得时刻发任务。函数也供迁移记录生成器复现顺序。
"""

import ast
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def phase_end(text, phase, role, indent):
    marker = indent + f"signal = publish_phase(signal, PHASE_{phase}, {role}, CORES, generation)\n"
    assert text.count(marker) == 1
    return text.index(marker) + len(marker)


def prioritize_cube(text, indent):
    begin = phase_end(text, "QA", "worker", indent)
    kv_end = phase_end(text, "KV_PROJECT", "worker", indent)
    qb_begin = text.index(indent + "# Q_B列块n归n%24；", kv_end)
    qb_end = phase_end(text, "QB", "worker", indent)
    weights_end = phase_end(text, "WEIGHTS_PROJECT", "worker", indent)
    idx_end = phase_end(text, "IDX_PROJECT", "worker", indent)
    had_end = phase_end(text, "IDX_QUERY_HAD", "worker", indent)
    assert begin < kv_end < qb_begin < qb_end < weights_end < idx_end < had_end
    kv, qr_wait, qb = text[begin:kv_end], text[kv_end:qb_begin], text[qb_begin:qb_end]
    idx_query, idx_project, had = text[qb_end:weights_end], text[weights_end:idx_end], text[idx_end:had_end]
    comment = (
        indent
        + "# 先准备Indexer压缩，再做Indexer query；这两条链决定长历史Score何时可启动。\n"
        + indent
        + "# 独立KV投影填在等待query RoPE之前，随后发Hadamard，再做主Q_B。\n"
        + indent
        + "# AIV在Q_B期间可完成Indexer量化/系数；所有K链、owner和权重保持不变。\n"
    )
    return text[:begin] + comment + idx_project + qr_wait + idx_query + kv + had + qb + text[had_end:]


def split_shared_query_waits(text, indent):
    """把主Query段的QR/RoPE获取前移到第一个真正消费者，QB获取留在原段。"""
    lines = text.splitlines(keepends=True)
    nodes = ast.parse(textwrap.dedent(text)).body
    selected = [
        n
        for n in nodes
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "wait_row_owners"
        and ast.unparse(n.value.args[1]) in ("PHASE_QR", "PHASE_ROPE")
    ]
    assert len(selected) == 2
    moved = "".join("".join(lines[n.lineno - 1 : n.end_lineno]) for n in selected)
    for node in reversed(selected):
        del lines[node.lineno - 1 : node.end_lineno]
    return moved, "".join(lines)


def prioritize_vector(text, indent):
    begin = phase_end(text, "ROPE", "core", indent)
    cache_end = phase_end(text, "ORIGINAL_CACHE", "core", indent)
    query_end = phase_end(text, "QUERY", "core", indent)
    idx_end = phase_end(text, "IDX_NORM", "core", indent)
    coeff_end = phase_end(text, "COEFF", "core", indent)
    assert begin < cache_end < query_end < idx_end < coeff_end
    kv = text[begin:cache_end]
    shared_waits, query = split_shared_query_waits(text[cache_end:query_end], indent)
    idx, coeff = text[query_end:idx_end], text[idx_end:coeff_end]
    comment = (
        indent
        + "# 主Query与Indexer query数学独立；先完成后者的RoPE/pool/norm，\n"
        + indent
        + "# 让AIC继续query Hadamard并在其主Q_B期间做量化/系数。QR/ROPE获取\n"
        + indent
        + "# 随第一个消费者前移，不能只移动计算却遗漏跨核数据可见性。\n"
    )
    return text[:begin] + comment + shared_waits + idx + kv + coeff + query + text[coeff_end:]


def main():
    before = HERE / "indexer_priority_v1/before"
    before.mkdir(parents=True, exist_ok=False)
    for role, reorder in (("cube", prioritize_cube), ("vector", prioritize_vector)):
        path = PKG / f"{role}_schedule.py"
        source = path.read_text()
        updated = reorder(source, "    ")
        (before / path.name).write_text(source)
        path.write_text(updated)


if __name__ == "__main__":
    main()
