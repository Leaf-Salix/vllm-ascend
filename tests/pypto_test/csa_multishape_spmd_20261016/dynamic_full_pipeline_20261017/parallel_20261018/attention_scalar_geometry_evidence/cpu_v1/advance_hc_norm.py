# SPDX-License-Identifier: Apache-2.0
"""将不依赖comb的HC混合/归一化提前；只记录源级编排调整，不参与生产调用。"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def advance(text, indent):
    start = text.index(indent + "comb = hc_comb_range(")
    comb_last = indent + "signal = publish_phase(signal, PHASE_COMB, core, CORES, generation)\n"
    comb_end = text.index(comb_last, start) + len(comb_last)
    mix_start = text.index(indent + "mixed, normed = hc_mix_norm_range(", comb_end)
    norm_last = indent + "signal = publish_phase(signal, PHASE_NORM, core, CORES, generation)\n"
    norm_end = text.index(norm_last, mix_start) + len(norm_last)
    removed_wait = text[comb_end:mix_start]
    assert removed_wait.count("wait_row_owners(") == 1 and removed_wait.count("PHASE_GATES") == 1
    comment = (
        indent + "# gates与mix_norm按相同M8条带归同一AIV；pre由本核写，且GATES发布\n"
        + indent + "# 已排空写回，无需等待所有其他gate owner。先混合/归一化并发布NORM，\n"
        + indent + "# AIC便可开始QA；后面的Sinkhorn只供最终HCpost，不是QA的依赖。\n"
    )
    return text[:start] + comment + text[mix_start:norm_end] + text[start:comb_end] + text[norm_end:]


def main():
    path = PKG / "vector_schedule.py"
    before = HERE / "hc_norm_early_v1/before"
    before.mkdir(parents=True, exist_ok=False)
    source = path.read_text()
    (before / path.name).write_text(source)
    path.write_text(advance(source, "    "))


if __name__ == "__main__":
    main()
