# SPDX-License-Identifier: Apache-2.0
"""仅将主Compressor源码K分块对齐实际K256，保留两条独立累加与尾行合同。"""

import ast
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "main_shared_left"
    source = output / "source_candidate"
    source.mkdir(parents=True, exist_ok=False)
    for path in (HERE / "source_after_score_split").glob("*.py"):
        shutil.copyfile(path, source / path.name)
    path = source / "compressor_projection_stages.py"
    text = path.read_text()
    fn, = [node for node in ast.parse(text).body
           if isinstance(node, ast.FunctionDef) and node.name == "compressor_project_stage"]
    lines = text.splitlines(keepends=True)
    body = "".join(lines[fn.lineno - 1:fn.end_lineno])
    count = len(re.findall(r"\bK_TILE\b", body))
    assert count == 6, count
    body = re.sub(r"\bK_TILE\b", "MAIN_SHARED_K_TILE", body)
    body = body.replace(
        "# 仅对K搬运做两级流水；八块按0..7顺序更新同一accumulator，不移动加法。",
        "# 原K512在后端拆成K256，但value/gate分别拆分会重复恢复同一Left输入。\n"
        "        # 源码按实际K256组织，两条独立链按原顺序累加，尝试共享同一Left；\n"
        "        # 仍用stage=2预取，是否真实共享及保持双缓冲须检查完整生成物。",
    )
    text = "".join(lines[:fn.lineno - 1]) + body + "".join(lines[fn.end_lineno:])
    text = text.replace(
        "CMP_N_TILE = 64\n", "CMP_N_TILE = 64\n"
        "# 主Compressor的资源表达，与实际硬件K256一致；Indexer仍用原K512。\n"
        "MAIN_SHARED_K_TILE = 256\n", 1,
    )
    text = text.replace(
        '"""动态 CSA 的两路压缩投影：保留原数学、显式 AIC 分工的核内阶段。',
        '"""动态 CSA 的两路压缩投影：主投影按实际K256共享value/gate输入搬运。\n\n'
        '本候选总览\n'
        '    原主投影源码K512由后端为value/gate分别拆成两个K256，造成同一\n'
        '    Mat输入重复搬入Left。这里仅把主投影源码K块改为256，让两个独立\n'
        '    输出复用同一个输入块；每个输出的实际K256累加顺序、M64/N64、\n'
        '    原动态尾行compact初始化、GM写回与外部owner/同步合同均保持。\n'
        '    不假定减少源码分块就一定提升性能；完整生成代码、精度和同轮L1\n'
        '    验证后再决定采用。Indexer M16/N32/K512不变，正式ops未修改。', 1,
    )
    text = text.replace("每个点独占 value/score 的同一输出块。每核完成全部 K4096 的八个 K512\n    分片，",
                        "每个点独占 value/score 的同一输出块。每核完成全部 K4096，主投影为\n    十六个K256，Indexer为八个K512，")
    ast.parse(text)
    path.write_text(text)
    for batch in (16, 28):
        shutil.copyfile(HERE / "wave_current_math" / f"controls_b{batch}.json",
                        output / f"controls_b{batch}.json")
    report = {"source_reference": "source_after_score_split/13259f92",
              "source_candidate": str(source), "modified_files": [path.name],
              "replaced_main_k_references": count, "reference_source_k": 512,
              "reference_actual_l0_k": 256, "candidate_source_k": 256,
              "indexer_k_unchanged": 512, "hardware_or_performance_proven": False}
    (output / "PREPARATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
