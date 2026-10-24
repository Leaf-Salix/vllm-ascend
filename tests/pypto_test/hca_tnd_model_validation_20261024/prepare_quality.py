# SPDX-License-Identifier: Apache-2.0
"""冻结正常EOS/2048token答案验证；只改采样与记录，不改模型/数学/调度。"""
import argparse
import difflib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args = parser.parse_args()
    origin = HERE / 'experiments/v04_graph_service'
    target = HERE / 'experiments' / args.name
    source = target / 'source'
    shutil.copytree(origin / 'source', source, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    path = source / 'tests/pypto_test/low_acceptance_20261001/mixed.py'
    original = path.read_text()
    text = original
    changes = [
        ('max_tokens=args.decode_tokens, ignore_eos=True,', 'max_tokens=args.decode_tokens, ignore_eos=False,'),
        ("'output_token_ids': list(output.outputs[0].token_ids)})",
         "'output_token_ids': list(output.outputs[0].token_ids),\n"
         "                            'output_text': output.outputs[0].text,\n"
         "                            'finish_reason': output.outputs[0].finish_reason,\n"
         "                            'stop_reason': output.outputs[0].stop_reason, 'finished': output.finished})"),
        ("'observation': observation, 'spec_decode': spec_decode_metrics(llm)})",
         "'observation': observation, 'spec_decode': spec_decode_metrics(llm),\n"
         "            'sampling': {'temperature': 0.0, 'ignore_eos': False, 'max_tokens': args.decode_tokens},\n"
         "            'tokenizer_eos_token_id': llm.get_tokenizer().eos_token_id})"),
    ]
    for before, after in changes:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    path.write_text(text)
    (target / 'QUALITY_HARNESS.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),
        fromfile='a/tests/pypto_test/low_acceptance_20261001/mixed.py',tofile='b/tests/pypto_test/low_acceptance_20261001/mixed.py')))
    manifest = json.loads((origin / 'manifest.json').read_text())
    manifest.update(source=str(source), parent_snapshot=str(origin), quality=True,
                    sampling=dict(temperature=0.0, ignore_eos=False, max_tokens=2048),
                    scope='B16同一16问题：完整答案质量；正式FP32算子，不使用BF16候选，不做性能对比')
    (target / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    # 沿用已验证的环境/16卡/图容量启动方法，固定长度的验收换成正常EOS检查。
    launcher = (HERE / 'run_model.sh').read_text()
    launcher = launcher.replace('# HCA整模型功能验证：沿用CSA已验证的PD bank与DP/EP16配置，不测性能。',
                                '# HCA/Native答案质量：正常EOS、最多2048token；不测性能。')
    launcher = launcher.replace('decode_tokens=192,', 'decode_tokens=2048,ignore_eos=False,')
    launcher = launcher.replace("assert data['toolchain']==current and current['status']=='validated'",
        "assert data['toolchain']==current and current['status']=='validated'\n"
        "assert json.load(open(pathlib.Path(data['repository']) / "
        "'tests/pypto_test/hca_tnd_model_validation_20261024/runs/model_b24_v04/functional_check.json'))['status']=='PASS'")
    launcher = launcher.replace('--decode-tokens 192 ', '--decode-tokens 2048 ')
    launcher = launcher.replace('--port 30991', '--port 31291')
    marker = 'python "$hca_source/tests/pypto_test/low_acceptance_20261001/mixed.py" \\\n'
    assert marker in launcher
    launcher = launcher[:launcher.index(marker)] + (
        'python "$hca_repo/tests/pypto_test/hca_tnd_model_validation_20261024/check_quality_run.py" "$hca_output"\n')
    (target / 'run_quality.sh').write_text(launcher)
    print(target)


if __name__ == '__main__':
    main()
