# SPDX-License-Identifier: Apache-2.0
"""按题意核对完整decode答案，终止原因、内容正确性和格式分别报告。"""
import json
from pathlib import Path
import re

from audit_answers import expected

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def arrays(text):
    """仅解析JSON兼容列表，不执行模型代码。保留位置以区分最终队列和初始队列。"""
    decoder = json.JSONDecoder()
    values = []
    for match in re.finditer(r'\[',text):
        try:
            value,length = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value,list):
            values.append((match.start(),match.start()+length,value))
    return values


def final_integer(cell):
    match = re.search(r'(-?\d+)\s*[;；。]?\s*$',cell.strip())
    return int(match[1]) if match else None


def score(text, kind, gold):
    notes = []
    if '```' in text:
        notes.append('含Markdown代码围栏，内容与格式分开核对')
    if kind == 'easy_repeat':
        wanted = gold['text'].rstrip('\r\n')
        actual = text.rstrip('\r\n')
        return dict(content='CORRECT' if actual == wanted else 'WRONG_TEXT_OR_COUNT',
                    observed_lines=len(actual.splitlines()),required_lines=gold['count'],format_notes=notes)
    found = arrays(text)
    if kind == 'python_trace':
        complete = [(start,end,value) for start,end,value in found if len(value) == 24
                    and all(isinstance(r,list) and len(r) == 4 and all(type(v) is int for v in r) for r in value)]
        if complete:
            start,end,value = complete[-1]
            differences = [dict(row=i+1, observed=a, expected=b) for i,(a,b) in enumerate(zip(value,gold['rows'])) if a != b]
            queues = [v for s,e,v in found if s >= end and len(v) == 7 and all(type(x) is int for x in v)]
            queue = queues[-1] if queues else None
            return dict(content='WRONG_NUMBERS' if differences or (queue is not None and queue != gold['final_queue'])
                        else 'CORRECT' if queue is not None else 'FINAL_QUEUE_NOT_FOUND',
                        parsed_rows=24, first_wrong=differences[0] if differences else None,
                        expected_queue=gold['final_queue'],observed_queue=queue,format_notes=notes)
        flat = [v for _,_,v in found if len(v) == 4 and all(type(x) is int for x in v) and 1 <= v[0] <= 24]
        wrong = [dict(observed=v,expected=gold['rows'][v[0]-1]) for v in flat if v != gold['rows'][v[0]-1]]
        return dict(content='WRONG_PARTIAL_NUMBERS' if wrong else 'INCOMPLETE_OR_UNPARSED',
                    parsed_rows=len(flat),first_wrong=wrong[0] if wrong else None,format_notes=notes)
    if kind == 'ledger_transform':
        lists = [v for _,_,v in found if v and all(isinstance(r,dict) for r in v)]
        if lists:
            value = lists[-1]
        else:
            value = []
            for match in re.finditer(r'\{[^{}]*\}',text):
                try:
                    item = json.loads(match[0])
                except ValueError:
                    continue
                if set(item) == {'id','net','score','tag'}:
                    value.append(item)
        wrong = [dict(position=i+1,observed=row,expected=gold['rows'][i] if i < 20 else None)
                 for i,row in enumerate(value) if i >= 20 or row != gold['rows'][i]]
        if any(list(row) != ['id','net','score','tag'] for row in value):
            notes.append('部分记录字段顺序不符合题目要求')
        return dict(content='WRONG_NUMBERS_OR_ORDER' if wrong else 'CORRECT' if value == gold['rows']
                    else 'INCOMPLETE_OR_UNPARSED',parsed_rows=len(value),
                    first_wrong=wrong[0] if wrong else None,format_notes=notes)
    rows = []
    for line in text.splitlines():
        cells = line.strip().strip('|').split('|')
        if len(cells) < 3 or not cells[0].strip().isdigit() or not cells[1].strip().isdigit():
            continue
        i,a = int(cells[0]),int(cells[1])
        if len(cells) == 5:
            values = [final_integer(cell) for cell in cells[2:]]
        else:
            body = '|'.join(cells[2:])
            values = [final_integer(cells[2].split(';')[0].split('；')[0])]
            for variable in ('新S','新R'):
                match = re.search(variable+r'\s*=(.*?)(?:[;；|]|$)',body)
                values.append(final_integer(match[1]) if match else None)
        if 1 <= i <= 24 and all(v is not None for v in values):
            rows.append([i,a,*values])
    # 同一i在推演及最终表中重复时，以最后一份明确结果核对。
    by_index = {row[0]:row for row in rows}
    values = [by_index[i] for i in sorted(by_index)]
    wrong = [dict(observed=row,expected=gold['rows'][row[0]-1]) for row in values if row != gold['rows'][row[0]-1]]
    sum_lines = [line for line in text.splitlines() if re.search(r'总和|新R.{0,8}和|R.{0,8}总|总计|Σ|sum',line)]
    sums = [final_integer(line.replace('**','').replace('`','')) for line in sum_lines]
    sums = [value for value in sums if value is not None]
    observed_sum = sums[-1] if sums else None
    return dict(content='WRONG_NUMBERS' if wrong or (observed_sum is not None and observed_sum != gold['new_r_sum'])
                else 'CORRECT' if values == gold['rows'] and observed_sum is not None else 'INCOMPLETE_OR_UNPARSED',
                parsed_rows=len(values),first_wrong=wrong[0] if wrong else None,
                observed_sum=observed_sum,expected_sum=gold['new_r_sum'],format_notes=notes)


def main():
    questions = {r['key']:r for r in read(HERE.parents[3] /
        'csa-repro-20261021-clean/kv-repro/mixed128k_questions.json')['cases']}
    sources, cases = {}, {}
    for mode in ('native','pto'):
        root = HERE / f'runs/quality_{mode}_b16_v07'
        execution = read(root/'quality_execution.json')
        assert execution['status'] == 'EXECUTION_PASS_QUALITY_NOT_SCORED'
        assert execution['all_dp_outputs_equal'], '副本不同则应逐rank核对，不能去重评分'
        sources[mode] = dict(path=str(root),execution=execution)
        for row in read(root/'rank0.mixed.json')['requests']:
            question = questions[row['key']]
            gold = expected(question['question'],question['kind'])
            entry = cases.setdefault(row['key'],dict(key=row['key'],kind=question['kind'],question=question['question'],expected=gold))
            entry[mode] = dict(finish_reason=row['finish_reason'], stop_reason=row['stop_reason'],
                               tokens=len(row['output_token_ids']),text=row['output_text'],
                               verdict=score(row['output_text'],question['kind'],gold))
    summary = {mode: dict(content_correct=sum(c[mode]['verdict']['content']=='CORRECT' for c in cases.values()),
                          token_limit=sum(c[mode]['finish_reason']=='length' for c in cases.values()),
                          stopped=sum(c[mode]['finish_reason']=='stop' for c in cases.values()),
                          unique_questions=len(cases)) for mode in sources}
    report = dict(status='ANSWER_CONTENT_REVIEW',criteria='正常EOS、2048token上限；题意计算真值，Native不作真值',
                  sources=sources,summary=summary,cases=list(cases.values()),
                  limitation='解析不足标为INCOMPLETE_OR_UNPARSED并须人工核对，不能直接计错或计对；格式备注不与数值正确性混算')
    (HERE/'QUALITY_EOS2048.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    md = ['# HCA/Native正常EOS、2048token上限答案验证',report['criteria'],
          '16个问题各在16rank重复；副本相同才去重。以下完整保留每个实际回答，不截断、不补写。',
          '| 问题 | Native结束/长度 | Native题意内容 | HCA结束/长度 | HCA题意内容 |',
          '|---|---|---|---|---|']
    for c in cases.values():
        n,p=c['native'],c['pto']
        md.append(f'| {c["key"]} | {n["finish_reason"]}/{n["tokens"]} | {n["verdict"]["content"]} | {p["finish_reason"]}/{p["tokens"]} | {p["verdict"]["content"]} |')
    for c in cases.values():
        md += [f'## {c["key"]}',c['question'],'### 题意标准结果','``````json',json.dumps(c['expected'],ensure_ascii=False),'``````']
        for mode in ('native','pto'):
            md += [f'### {mode}实际回答',json.dumps(c[mode]['verdict'],ensure_ascii=False),
                   '``````text',c[mode]['text'],'``````']
    (HERE/'QUALITY_EOS2048.md').write_text('\n\n'.join(md)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
