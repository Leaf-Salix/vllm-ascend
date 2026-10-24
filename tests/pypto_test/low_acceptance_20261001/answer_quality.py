# SPDX-License-Identifier: Apache-2.0
"""固定混合问题的答案核对，沿用HCA已使用的题意计算与解析规则。

只计算题目定义的确定性结果，不执行模型生成代码；解析不完整单独记录。
"""

import ast
import json
import re
from collections import deque


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
            rows.append(dict(id=key, net=net, score=(net * net_multiple + int(fee) * score_fee) % 1009, tag=tag))
        return {"rows": sorted(rows, key=lambda row: (-row["score"], row["id"]))}
    if kind == "transition_trace":
        s, r = map(int, re.search(r"初始状态S=(\d+)、寄存器R=(\d+)", question).groups())
        values = list(map(int, re.search(r"输入依次为：\n([\d,]+)", question)[1].split(",")))
        coefficient_a, coefficient_s = map(int, re.search(r"t=\(R\+(\d+)\*a\+(\d+)\*S\)", question).groups())
        rows = []
        for i, a in enumerate(values, 1):
            t = (r + coefficient_a * a + coefficient_s * s) % 509
            if t % 4 == 0:
                s = (s + a + 3) % 11
            elif t % 4 == 1:
                s = (2 * s + 5) % 11
            else:
                s = (s * 3 + a) % 11
            r = (t * 7 + s * 13 + i) % 521
            rows.append([i, a, t, s, r])
        return {"rows": rows, "new_r_sum": sum(row[-1] for row in rows)}
    count = int(re.search(r"这句话(\d+)次", question)[1])
    return {"text": (question.splitlines()[-1] + "\n") * count, "count": count}


def arrays(text):
    """仅解析JSON兼容列表，不执行模型代码。保留位置以区分最终队列和初始队列。"""
    decoder = json.JSONDecoder()
    values = []
    for match in re.finditer(r"\[", text):
        try:
            value, length = decoder.raw_decode(text[match.start() :])
        except ValueError:
            continue
        if isinstance(value, list):
            values.append((match.start(), match.start() + length, value))
    return values


def final_integer(cell):
    match = re.search(r"(-?\d+)\s*[;；。]?\s*$", cell.strip())
    return int(match[1]) if match else None


def score(text, kind, gold):
    notes = []
    if "```" in text:
        notes.append("含Markdown代码围栏，内容与格式分开核对")
    if kind == "easy_repeat":
        wanted = gold["text"].rstrip("\r\n")
        actual = text.rstrip("\r\n")
        return dict(
            content="CORRECT" if actual == wanted else "WRONG_TEXT_OR_COUNT",
            observed_lines=len(actual.splitlines()),
            required_lines=gold["count"],
            format_notes=notes,
        )
    found = arrays(text)
    if kind == "python_trace":
        complete = [
            (start, end, value)
            for start, end, value in found
            if len(value) == 24
            and all(isinstance(r, list) and len(r) == 4 and all(type(v) is int for v in r) for r in value)
        ]
        if complete:
            start, end, value = complete[-1]
            differences = [
                dict(row=i + 1, observed=a, expected=b) for i, (a, b) in enumerate(zip(value, gold["rows"])) if a != b
            ]
            queues = [v for s, e, v in found if s >= end and len(v) == 7 and all(type(x) is int for x in v)]
            queue = queues[-1] if queues else None
            return dict(
                content="WRONG_NUMBERS"
                if differences or (queue is not None and queue != gold["final_queue"])
                else "CORRECT"
                if queue is not None
                else "FINAL_QUEUE_NOT_FOUND",
                parsed_rows=24,
                first_wrong=differences[0] if differences else None,
                expected_queue=gold["final_queue"],
                observed_queue=queue,
                format_notes=notes,
            )
        flat = [v for _, _, v in found if len(v) == 4 and all(type(x) is int for x in v) and 1 <= v[0] <= 24]
        wrong = [dict(observed=v, expected=gold["rows"][v[0] - 1]) for v in flat if v != gold["rows"][v[0] - 1]]
        return dict(
            content="WRONG_PARTIAL_NUMBERS" if wrong else "INCOMPLETE_OR_UNPARSED",
            parsed_rows=len(flat),
            first_wrong=wrong[0] if wrong else None,
            format_notes=notes,
        )
    if kind == "ledger_transform":
        lists = [v for _, _, v in found if v and all(isinstance(r, dict) for r in v)]
        if lists:
            value = lists[-1]
        else:
            value = []
            for match in re.finditer(r"\{[^{}]*\}", text):
                try:
                    item = json.loads(match[0])
                except ValueError:
                    continue
                if set(item) == {"id", "net", "score", "tag"}:
                    value.append(item)
        wrong = [
            dict(position=i + 1, observed=row, expected=gold["rows"][i] if i < 20 else None)
            for i, row in enumerate(value)
            if i >= 20 or row != gold["rows"][i]
        ]
        if any(list(row) != ["id", "net", "score", "tag"] for row in value):
            notes.append("部分记录字段顺序不符合题目要求")
        return dict(
            content="WRONG_NUMBERS_OR_ORDER"
            if wrong
            else "CORRECT"
            if value == gold["rows"]
            else "INCOMPLETE_OR_UNPARSED",
            parsed_rows=len(value),
            first_wrong=wrong[0] if wrong else None,
            format_notes=notes,
        )
    rows = []
    for line in text.splitlines():
        cells = line.strip().strip("|").split("|")
        if len(cells) < 3 or not cells[0].strip().isdigit() or not cells[1].strip().isdigit():
            continue
        i, a = int(cells[0]), int(cells[1])
        if len(cells) == 5:
            values = [final_integer(cell) for cell in cells[2:]]
        else:
            body = "|".join(cells[2:])
            values = [final_integer(cells[2].split(";")[0].split("；")[0])]
            for variable in ("新S", "新R"):
                match = re.search(variable + r"\s*=(.*?)(?:[;；|]|$)", body)
                values.append(final_integer(match[1]) if match else None)
        if 1 <= i <= 24 and all(v is not None for v in values):
            rows.append([i, a, *values])
    # 同一i在推演及最终表中重复时，以最后一份明确结果核对。
    by_index = {row[0]: row for row in rows}
    values = [by_index[i] for i in sorted(by_index)]
    wrong = [dict(observed=row, expected=gold["rows"][row[0] - 1]) for row in values if row != gold["rows"][row[0] - 1]]
    sum_lines = [line for line in text.splitlines() if re.search(r"总和|新R.{0,8}和|R.{0,8}总|总计|Σ|sum", line)]
    sums = [final_integer(line.replace("**", "").replace("`", "")) for line in sum_lines]
    sums = [value for value in sums if value is not None]
    observed_sum = sums[-1] if sums else None
    return dict(
        content="WRONG_NUMBERS"
        if wrong or (observed_sum is not None and observed_sum != gold["new_r_sum"])
        else "CORRECT"
        if values == gold["rows"] and observed_sum is not None
        else "INCOMPLETE_OR_UNPARSED",
        parsed_rows=len(values),
        first_wrong=wrong[0] if wrong else None,
        observed_sum=observed_sum,
        expected_sum=gold["new_r_sum"],
        format_notes=notes,
    )
