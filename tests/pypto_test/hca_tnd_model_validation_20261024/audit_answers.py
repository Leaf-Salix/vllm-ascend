# SPDX-License-Identifier: Apache-2.0
"""在CPU上核对已有192-token答案；标准答案来自题意，不把Native输出当真值。

只检查能够可靠解析的完整数值行/账本记录和重复句子的前缀。
未生成完、解释文字和未实现的解析分开标记，不据此计算整套题的准确率。
不执行题目或模型生成的Python代码，不调用模型，不增加NPU测试。
"""
import ast
from collections import deque
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BANK = Path("/data/pyptouser/qinchuanyu/pto-eager/csa-repro-20261021-clean/kv-repro")


def expected(question, kind):
    if kind == "python_trace":
        q = deque(ast.literal_eval(re.search(r"q = deque\((\[[^\n]+\])\)", question)[1]))
        s = int(re.search(r"\ns = (\d+)", question)[1])
        multiplier = int(re.search(r"b = \(a \* (\d+) \+", question)[1])
        rows = []
        for i in range(1, 25):
            q.rotate((i % 5) - 2)
            a = q.popleft()
            b = (a * multiplier + s * 3 + i * i) % 257
            if b % 3 == 0:
                q.appendleft((b + i * 11) % 263)
            else:
                q.append(b)
            s = (s ^ (b + i)) % 251
            rows.append([i, a, b, s])
        return {"rows": rows, "final_queue": list(q)}
    if kind == "ledger_transform":
        fee_multiple = int(re.search(r"net=amount-(\d+)\*fee", question)[1])
        net_multiple, score_fee = map(int, re.search(r"score=\(net\*(\d+)\+fee\*(\d+)\)", question).groups())
        rows = []
        for key, amount, fee, tag in re.findall(r"^([a-z]\d),(\d+),(\d+),([ABC])$", question, re.M):
            net = int(amount) - fee_multiple * int(fee)
            rows.append(dict(id=key, net=net, score=(net*net_multiple + int(fee)*score_fee) % 1009, tag=tag))
        return {"rows": sorted(rows, key=lambda row: (-row["score"], row["id"]))}
    if kind == "transition_trace":
        s, r = map(int, re.search(r"初始状态S=(\d+)、寄存器R=(\d+)", question).groups())
        values = list(map(int, re.search(r"输入依次为：\n([\d,]+)", question)[1].split(',')))
        coefficient_a, coefficient_s = map(int, re.search(r"t=\(R\+(\d+)\*a\+(\d+)\*S\)", question).groups())
        rows = []
        for i, a in enumerate(values, 1):
            t = (r + coefficient_a*a + coefficient_s*s) % 509
            if t % 4 == 0:
                s = (s+a+3) % 11
            elif t % 4 == 1:
                s = (2*s+5) % 11
            else:
                s = (s*3+a) % 11
            r = (t*7+s*13+i) % 521
            rows.append([i,a,t,s,r])
        return {"rows": rows, "new_r_sum": sum(row[-1] for row in rows)}
    count = int(re.search(r"这句话(\d+)次", question)[1])
    return {"text": (question.splitlines()[-1]+'\n')*count, "count": count}


def inspect_answer(text, gold, kind):
    if kind == "easy_repeat":
        return {"status": "COMPLETE_PASS" if text == gold["text"] else
                "PREFIX_ONLY" if bool(text) and gold["text"].startswith(text) else "WRONG_TEXT",
                "complete_lines": text.count('\n'), "required_lines": gold["count"]}
    if kind == "python_trace":
        # 只解析四个整数的完整输出行；解释段的初始7项队列不会误入。
        parsed = [list(map(int, row)) for row in re.findall(
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)]
        bad = [dict(observed=row, expected=gold["rows"][row[0]-1] if 1 <= row[0] <= 24 else None)
               for row in parsed if not 1 <= row[0] <= 24 or row != gold["rows"][row[0]-1]]
    elif kind == "ledger_transform":
        parsed = [json.loads(record) for record in re.findall(r'\{[^{}]+\}', text)]
        bad = [dict(observed=row, expected=gold["rows"][i] if i < len(gold["rows"]) else None)
               for i,row in enumerate(parsed) if i >= len(gold["rows"]) or row != gold["rows"][i]]
    else:
        # 当前两侧含长算式，不尝试从任意解释文本猜出最终答案。
        parsed = [list(map(int, row)) for row in re.findall(
            r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*$", text, re.M)]
        bad = [dict(observed=row, expected=gold["rows"][row[0]-1] if 1 <= row[0] <= 24 else None)
               for row in parsed if not 1 <= row[0] <= 24 or row != gold["rows"][row[0]-1]]
    return {"status": "WRONG_NUMBERS" if bad else "PARTIAL_NUMBERS_ONLY" if parsed else "NOT_SCORED",
            "parsed_records": len(parsed), "required_records": len(gold["rows"]),
            "wrong_records": len(bad), "first_wrong": bad[0] if bad else None,
            "complete_answer_pass": False,
            "note": "只统计已出现的明确数值；完整答案/所有格式要求尚未验收"}


def main():
    questions = {row["key"]: row for row in json.loads((BANK / "mixed128k_questions.json").read_text())["cases"]}
    source = json.loads((HERE / "B16_COMPARISON.json").read_text())
    rows = []
    for row in source["rows"]:
        if row["rank"] != 0:
            continue
        question = questions[row["key"]]["question"]
        gold = expected(question, row["kind"])
        rows.append({"key": row["key"], "kind": row["kind"], "question": question, "expected": gold,
                     "native": inspect_answer(row["native_text"], gold, row["kind"]),
                     "hca": inspect_answer(row["hca_text"], gold, row["kind"])})
    report = {"source": "B16_COMPARISON.json已有实测；无新模型测试", "unique_questions": len(rows),
              "replicas": 16, "output_token_limit": 192,
              "acceptance": "答案满足题目要求；Native逐bit/逐token一致不是硬门槛",
              "scope": "这组固定192token用于功能轨迹；不是已完成的答案质量基准",
              "summary": {side: {state: sum(row[side]["status"] == state for row in rows)
                                  for state in sorted({row[side]["status"] for row in rows})}
                          for side in ("native", "hca")}, "rows": rows}
    (HERE / "ANSWER_AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    text = ['# 已有B16答案的题意核对', '',
            '2026-10-24用户明确：以decode答案是否符合要求为准，不要求对齐Native token。', '',
            '仅分析已有192token输出，不新跑设备，不将Native当标准答案。16个问题各重复16个rank，不能当成256个独立质量样本。', '',
            '| 问题 | 类型 | Native | HCA MPMD |', '|---|---|---|---|']
    for row in rows:
        text.append(f'| {row["key"]} | {row["kind"]} | {row["native"]["status"]} | {row["hca"]["status"]} |')
    text += ['', 'WRONG_NUMBERS表示已生成的明确数值错误；PREFIX_ONLY只表示生成前缀符合要求；',
             'NOT_SCORED表示当前没有可可靠解析的完整数值行，不等于正确或错误。',
             '长解释/未完成输出不计算整题准确率。此轮记录不能证明HCA导致了新增答案错误，也不能宣布答案质量验收通过。', '']
    for row in rows:
        if any(row[side].get('first_wrong') for side in ('native','hca')):
            text += [f'## {row["key"]}首条明确数值错误', '']
            for side in ('native','hca'):
                text.append(f'- {side}: `{json.dumps(row[side].get("first_wrong"),ensure_ascii=False)}`')
            text.append('')
    (HERE / "ANSWER_AUDIT.md").write_text('\n'.join(text))
    print(json.dumps(report['summary'],ensure_ascii=False))


if __name__ == "__main__":
    main()
