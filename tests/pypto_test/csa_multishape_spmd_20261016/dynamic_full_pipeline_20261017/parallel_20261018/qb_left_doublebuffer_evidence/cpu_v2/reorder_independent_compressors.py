# SPDX-License-Identifier: Apache-2.0
"""保留并重排动态源编排的独立分支；本脚本只记录本次优化，不参与运行时。

不接收shape、不读取历史DAG，也不生成数学。将Indexer压缩提前，允许其AIV
pool/norm与主压缩AIC投影重叠；各分支数学、owner、信号、state读完集合不变。
正式源码与最初手工组装文本同步更新，旧文本存于schedule_overlap_v1/before。
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def reorder(text, kind, indent):
    if kind == "cube":
        begin = text.index(indent + "main_value, main_score = compressor_project_stage(")
        end = text.index(indent + "idx_value, idx_score = indexer_compressor_project_stage(", begin)
        main = text[begin:end]
        text = text[:begin] + text[end:]
        anchor = indent + "signal = publish_phase(signal, PHASE_IDX_QUERY_HAD, worker, CORES, generation)\n"
        assert text.count(anchor) == 1
        comment = (
            indent
            + "# 先发布Indexer两种投影和query Hadamard，再执行独立的主压缩投影。\n"
            + indent
            + "# 此时AIV可以做Indexer pool/norm及query量化；主投影不等待这些结果，\n"
            + indent
            + "# 等IDX_NORM放在主投影之后，避免空等，也不改变任一矩阵累加顺序。\n"
        )
        return text.replace(anchor, anchor + "\n" + comment + main)
    begin = text.index(indent + "# Indexer压缩与主压缩独立state/metadata；")
    final = indent + "signal = publish_phase(signal, PHASE_IDX_NORM, core, CORES, generation)\n"
    end = text.index(final, begin) + len(final)
    indexer = text[begin:end]
    text = text[:begin] + text[end:]
    meta = indent + "signal = wait_producer(signal, PHASE_META, GROUPS + planner, CORES, generation)\n"
    assert text.count(meta) == 1
    text = text.replace(meta, "")
    anchor = indent + "signal = wait_row_owners(signal, PHASE_IDX_QUERY_HAD,"
    insertion = text.index(anchor)
    comment = (
        indent
        + "# Indexer压缩只依赖自身投影/Native state；不必排在主压缩输出之后。\n"
        + indent
        + "# 先取得compact offsets，再做pool→读完集合→commit→norm，和AIC上的\n"
        + indent
        + "# 主压缩投影重叠。仍需所有pool读完后才能写共享prefix state。\n"
    )
    text = text[:insertion] + comment + meta + indexer + "\n" + text[insertion:]
    text = text.replace(
        "# 系数要读取其他核的head权重/scale及请求分组。公共计划到这里才需要；",
        "# 系数要读取其他核的head权重/scale及请求分组。公共计划此前已等待；",
    )
    text = text.replace(
        "# 系数要读取其他核的head权重/scale及请求分组。公共计划到这里才需要，",
        "# 系数要读取其他核的head权重/scale及请求分组。公共计划此前已等待，",
    )
    return text


def main():
    before = HERE / "schedule_overlap_v1/before"
    before.mkdir(parents=True, exist_ok=False)
    for kind in ("cube", "vector"):
        for path, indent in ((PKG / f"{kind}_schedule.py", "    "), (HERE / f"{kind}_schedule.txt", "")):
            source = path.read_text()
            (before / path.name).write_text(source)
            path.write_text(reorder(source, kind, indent))


if __name__ == "__main__":
    main()
