# SPDX-License-Identifier: Apache-2.0
"""导出HCA与Native已有完整输出；不重新推理，不以token一致替代答案正确性。"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    output = HERE.parents[3] / "downloads/HCA_TND_MODEL_INTEGRATION_20261024"
    output.mkdir(parents=True, exist_ok=True)
    questions = {r['key']:r for r in json.loads((HERE.parents[3] /
        'csa-repro-20261021-clean/kv-repro/mixed128k_questions.json').read_text())['cases']}
    summary = json.loads((HERE / 'MODEL_INTEGRATION_SUMMARY.json').read_text())
    assert all(row['cross_dp_tokens_equal'] for row in summary['variants'])
    note = ('每条输出完整保留实际192token，可能尚未完成回答；没有摘录或补写。'
            '16个DP副本逐token相同后才去重。Native仅为参考，按题意判断答案正确性，不要求token一致。')
    for number,batch in enumerate((16,24),1):
        comparison = json.loads((HERE / f'B{batch}_COMPARISON.json').read_text())
        rows = [r for r in comparison['rows'] if r['rank'] == 0]
        for rank in range(1,16):
            other = [r for r in comparison['rows'] if r['rank'] == rank]
            assert [(r['key'],r['native_tokens'],r['hca_tokens']) for r in other] == [
                (r['key'],r['native_tokens'],r['hca_tokens']) for r in rows], '两侧副本均相同才去重'
        title = f'128K / B{batch}：Native与HCA TND MPMD原始问答'
        md = [f'# {title}',note]
        sections = []
        for row in rows:
            prompt = questions[row['key']]['question']
            md += [f'## {row["key"]}',prompt,'### Native','``````text',row['native_text'],'``````',
                   '### HCA MPMD','``````text',row['hca_text'],'``````']
            sections.append(f'<section><h2>{row["key"]}</h2><pre>{html.escape(prompt)}</pre>'
                f'<div class="answers"><article><h3>Native</h3><pre>{html.escape(row["native_text"])}</pre></article>'
                f'<article><h3>HCA MPMD</h3><pre>{html.escape(row["hca_text"])}</pre></article></div></section>')
        stem = f'{number:02d}_128K_B{batch}_Native_vs_HCA_TND_MPMD'
        (output/(stem+'.md')).write_text('\n\n'.join(md)+'\n')
        (output/(stem+'.json')).write_text(json.dumps({'source_report':str(HERE/f'B{batch}_COMPARISON.json'),
            'note':note,'rows':[{**r,'question':questions[r['key']]['question']} for r in rows]},ensure_ascii=False,indent=2)+'\n')
        style = ('body{font-family:system-ui;margin:24px;background:#f8f9fb}section{background:white;padding:20px;margin:20px 0}'
                 '.answers{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
                 '@media(max-width:800px){.answers{grid-template-columns:1fr}}')
        (output/(stem+'.html')).write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title>'
            f'<style>{style}</style><h1>{title}</h1><p>{note}</p>'+''.join(sections)+'</html>')
    (output/'00_阅读说明.md').write_text('# HCA真实模型接入问答\n\n'+note+'\n\n'
        '本目录展示完整问题后缀与已生成回答。含128K背景的实际输入仍在原offline KV bank中；'
        '该导出不复制权重、KV或生成新的回答。功能/图重放与答案质量分开报告。\n')
    print(output)


if __name__ == '__main__':
    main()
