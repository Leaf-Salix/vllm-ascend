# SPDX-License-Identifier: Apache-2.0
"""参考CSA的问答展示方式，导出Native/HCA全文、结束原因、原始token及完整输入。"""
import html
import json
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def fenced(text):
    fence = '````````'
    while fence in text:
        fence += '`'
    return fence+'text\n'+text+'\n'+fence


def main():
    output = HERE.parents[3] / 'downloads/HCA_TND_MODEL_INTEGRATION_20261024'
    output.mkdir(parents=True,exist_ok=True)
    inputs = output/'完整模型输入'
    inputs.mkdir(exist_ok=True)
    report = read(HERE/'QUALITY_EOS2048.json')
    roots = {mode:Path(record['path']) for mode,record in report['sources'].items()}
    configurations = {mode:read(root/'configuration.json') for mode,root in roots.items()}
    for key in ('source_commit','source','bank','batch','decode_tokens','ignore_eos','selected_cases','TP','DP','EP',
                'NZ','EPLB','static_kernel','superkernel','graph_mode','deterministic_level','hccl_buffer_mb','toolchain'):
        assert configurations['native'][key] == configurations['pto'][key],key
    assert all(record['execution']['all_dp_outputs_equal'] for record in report['sources'].values())
    data = {mode:read(root/'rank0.mixed.json') for mode,root in roots.items()}
    by_key = {mode:{r['key']:r for r in record['requests']} for mode,record in data.items()}
    bank = Path(configurations['native']['bank'])
    plan = read(bank/'plan.json')
    cases = {row['key']:row for row in plan['cases']}
    tokenizer = Tokenizer.from_file(str(Path(plan['model'])/'tokenizer.json'))
    labels = {'native':'Native','pto':'HCA TND MPMD'}
    verdicts = {
        'CORRECT':'已解析的完整题意内容符合', 'WRONG_NUMBERS':'明确数值不同',
        'WRONG_PARTIAL_NUMBERS':'已输出部分存在明确数值差异',
        'WRONG_NUMBERS_OR_ORDER':'账本数值或排序不同', 'WRONG_TEXT_OR_COUNT':'重复内容或次数不符',
        'INCOMPLETE_OR_UNPARSED':'答案未完成或尚不能完整解析', 'FINAL_QUEUE_NOT_FOUND':'未找到要求的最终队列',
    }
    note = ('正常EOS停止，最多2048token。实际输出全文完整保留，不截取、不补写。'
            '16个DP副本逐token相同后才去重；Native不是标准答案，token不同本身不判精度失败。'
            '题意核对单独展示可复算的数值证据，未完成与明确错误分开。')
    title = '128K / B16：Native与HCA正常EOS、2048token完整问答对照'
    manifest = {'comparison_report':str(HERE/'QUALITY_EOS2048.json'),'configurations':configurations,
                'summary':report['summary'],'note':note,'requests':[]}
    md = ['# '+title,note,'## 结果概况',
          '| 实现 | 正常结束 | 到达2048上限 | 已核验完整内容符合题意 |',
          '|---|---:|---:|---:|']
    summary_rows = []
    for mode,label in labels.items():
        s = report['summary'][mode]
        md.append(f'| {label} | {s["stopped"]}/16 | {s["token_limit"]}/16 | {s["content_correct"]}/16 |')
        summary_rows.append(f'<tr><td>{label}</td><td>{s["stopped"]}/16</td><td>{s["token_limit"]}/16</td><td>{s["content_correct"]}/16</td></tr>')
    md += ['', '此处“完整内容符合题意”只针对这16个题目，不是通用模型准确率，也不能单独归因为HCA算子。',
           '## 原始问答']
    sections, nav = [], []
    for row in report['cases']:
        key = row['key']
        tokens = read(bank/cases[key]['tokens'])
        input_name = f'Q{key[-2:]}_{row["kind"]}_{len(tokens)}tokens_完整输入.txt'
        (inputs/input_name).write_text(tokenizer.decode(tokens,skip_special_tokens=False))
        input_link = str(Path(inputs.name)/input_name)
        n,p = (by_key[mode][key]['output_token_ids'] for mode in ('native','pto'))
        common = next((i for i,(a,b) in enumerate(zip(n,p)) if a != b),min(len(n),len(p)))
        differences = dict(common_prefix_tokens=common,common_positions=min(len(n),len(p)),
                           different_common_positions=sum(a != b for a,b in zip(n,p)),
                           length_difference=abs(len(n)-len(p)),all_tokens_equal=n==p)
        record = {**row,'input_file':input_link,'input_tokens':len(tokens),'token_comparison':differences,
                  'raw_outputs':{mode:by_key[mode][key] for mode in labels}}
        manifest['requests'].append(record)
        nav.append(f'<a href="#{key}">Q{key[-2:]}</a>')
        question = html.escape(row['question'])
        comparison = (f'共同前缀{common}token；共同位置中不同{differences["different_common_positions"]}'
                      f'/{differences["common_positions"]}；输出长度差{differences["length_difference"]}。位置差异不是语义错误率。')
        md += [f'### {key} · {row["kind"]}',f'[含128K背景的完整输入]({input_link})',row['question'],comparison]
        cards = []
        for mode,label in labels.items():
            item = row[mode]
            finish = '正常EOS/停止' if item['finish_reason']=='stop' else '达到2048token上限'
            status = verdicts[item['verdict']['content']]
            header = f'{label}：{finish}，{item["tokens"]}token；{status}'
            cards.append(f'<article><h3>{header}</h3><pre>{html.escape(item["text"])}</pre></article>')
            md += [f'#### {header}',fenced(item['text'])]
        evidence = json.dumps({'expected':row['expected'],'native':row['native']['verdict'],
                               'hca':row['pto']['verdict']},ensure_ascii=False,indent=2)
        md += ['#### 题意标准结果及核对证据',fenced(evidence)]
        sections.append(f'<section id="{key}"><h2>Q{key[-2:]} · {row["kind"]}</h2>'
                        f'<a href="{html.escape(input_link,quote=True)}">查看含128K背景的完整输入</a>'
                        f'<h3>完整问题后缀</h3><pre>{question}</pre><p>{comparison}</p>'
                        f'<div class="answers">{"".join(cards)}</div><details><summary>展开题意标准结果及核对证据</summary>'
                        f'<pre>{html.escape(evidence)}</pre></details></section>')
    stem='03_128K_B16_Native_vs_HCA_正常EOS_2048token_完整问答'
    (output/(stem+'.md')).write_text('\n\n'.join(md)+'\n')
    style=('body{font-family:system-ui,sans-serif;background:#f5f7fa;color:#172636;margin:24px}'
           'main{max-width:1700px;margin:auto}section{background:white;margin:24px 0;padding:24px;border:1px solid #d9e1e8}'
           '.answers{display:grid;grid-template-columns:1fr 1fr;gap:24px}.answers article{min-width:0}'
           'pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;font-family:inherit;background:#f8f9fc;padding:16px}'
           'nav a{margin-right:18px;line-height:2.5}.note{padding:16px;background:#fff2cc}'
           'table{border-collapse:collapse;background:white}th,td{border:1px solid #d9e1e8;padding:10px;text-align:left}'
           '@media(max-width:850px){.answers{grid-template-columns:1fr}}')
    page=(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title><style>{style}</style>'
          f'<main><h1>{title}</h1><p class="note">{note}</p><table><thead><tr><th>实现</th><th>正常结束</th>'
          '<th>达到2048上限</th><th>已核验完整内容符合题意</th></tr></thead><tbody>'
          +''.join(summary_rows)+'</tbody></table><p>只对本批16个问题作内容核对，不是通用模型准确率，也不能单独归因为HCA。</p>'
          +f'<nav>{"".join(nav)}</nav>'+''.join(sections)+'</main></html>')
    (output/(stem+'.html')).write_text(page)
    (output/'91_正常EOS2048_问题与原始输出token.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    readme = ('# HCA真实模型接入对照\n\n优先打开`03_128K_B16_Native_vs_HCA_正常EOS_2048token_完整问答.html`。\n\n'
        '- 03：本次正常EOS、上限2048token，两列展示Native/HCA实际完整回答、结束原因和题意核对。\n'
        '- 01/02：先前B16/B24固定192token功能验证；可能未答完，不与03混算。\n'
        '- 完整模型输入：16个实际128K输入，保留特殊token文本；页面展示完整问题后缀。\n'
        '- 91 JSON：输出token、正文、结束原因、题意标准结果及全部来源。16个DP副本相同后才去重。\n'
        '- token差异与答案质量分开。正常结束不等于答对，到上限不冒充完整答案。\n'
        '- 无新推理、无摘录补写、无权重/KV/二进制，不作性能结论。\n')
    (output/'00_阅读说明.md').write_text(readme)
    (HERE/'QUALITY_EXPORT.json').write_text(json.dumps(dict(directory=str(output),entry=str(output/(stem+'.html')),
        files=[{'path':str(p.relative_to(output)),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()],
        source=str(HERE/'QUALITY_EOS2048.json')),ensure_ascii=False,indent=2)+'\n')
    print(output/(stem+'.html'))


if __name__ == '__main__':
    main()
