# SPDX-License-Identifier: Apache-2.0
"""为单M128行块的长历史区间准备O_A N192动态尾列候选；冻结后恢复正式源码。"""
import ast
import json
import shutil
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=here, text=True,
    ).strip())
    formal = repo / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    base = here.parent / "attention_explicit_row_dma_20261019"
    saved = {p.name: p.read_bytes() for p in formal.glob('*.py')}
    assert all(data == (base / 'source_reference' / name).read_bytes()
               for name, data in saved.items()), '正式源已变化，不能覆盖'
    texts = {name: data.decode() for name, data in saved.items()}
    name = 'output_projection_a_stages.py'
    t = texts[name]
    t = t.replace('输入输出与阅读顺序\n',
        '输入输出与阅读顺序\n'
        '    新候选支持N192：1024列切为5个192列块与末64列块，避免N128时\n'
        '    64个输出任务在24核形成三轮。这里48项在24核恰好两轮，实际是否\n'
        '    受益仍由完整CSA验证；更大tile可能增加局部资源与搬运代价。\n'
        '    N尾列由GM视图的valid_shape裁剪，不能补读下一group的权重或写越界。\n'
        '    AIC组发布和AIV量化等待同步使用ceil(1024/N)，避免漏掉末64列。\n', 1)
    t = t.replace('128/256列tile', '128/192/256列tile')
    t = t.replace('A_COL_TILE为128/256', 'A_COL_TILE为128/192/256')
    t = t.replace('O_LORA // A_COL_TILE', '(O_LORA + A_COL_TILE - 1) // A_COL_TILE')
    t = t.replace(
        '        # row_blocks*column_blocks整除列块数，因此与pa_unit%column_blocks等价。',
        '        # row_blocks*column_blocks整除列块数，因此与pa_unit%column_blocks等价。\n'
        '        # N192末块只有64列，Tensor slice保留有效列；列块数须向上取整。', 1)
    texts[name] = t
    for name in ['cube_schedule.py', 'vector_schedule.py']:
        texts[name] = texts[name].replace('1024 // OA_COLS', '(1024 + OA_COLS - 1) // OA_COLS')
    old = '        oa_cols=128 if tokens <= 96 else 256,'
    assert old in texts['policy.py']
    texts['policy.py'] = texts['policy.py'].replace(old,
        '        # 64≤T≤128的长历史域：一个M128行块，N192给出48个输出任务，\n'
        '        # 每个AIC两项，末块valid N64。动态行数/列尾仍由实际shape决定。\n'
        '        # 这是待实测的资源策略，不按B16/S6点名，也不扩到双M行块。\n'
        '        oa_cols=192 if long_history and 64 <= tokens <= 128 else 128 if tokens <= 96 else 256,', 1)
    for text in texts.values():
        ast.parse(text)
    for name, data in saved.items():
        path = here / 'source_reference' / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
    try:
        for name, text in texts.items():
            (formal / name).write_text(text)
        shutil.copytree(formal, here / 'source_candidate',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    finally:
        for name, data in saved.items():
            (formal / name).write_bytes(data)
    for name in ['run_cpu.sh', 'run_device.sh', 'policy_candidate.json']:
        shutil.copyfile(base / name, here / name)
    controls = json.loads((base / 'controls_on.json').read_text())
    controls['OA_COLS'] = 192
    (here / 'controls_on.json').write_text(json.dumps(controls,ensure_ascii=False,indent=2)+'\n')
    (here / 'SOURCE_SCOPE.json').write_text(json.dumps(dict(
        reference_commit='289cc072', authored_in_ops=True, formal_source_restored=True,
        changed_files=[n for n in texts if texts[n].encode()!=saved[n]],
        interval='long_history && 64 <= T <= 128',
        math='原16次K256 FP32累加顺序；仅N分块及对应owner数量改变',
        attention_candidate_combined=False, dependencies_changed=False,
        cpu_validated=False, device_validated=False,
    ),ensure_ascii=False,indent=2)+'\n')
    print('prepared', here)


if __name__ == '__main__':
    main()
