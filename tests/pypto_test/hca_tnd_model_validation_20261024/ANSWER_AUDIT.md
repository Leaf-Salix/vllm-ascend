# 已有B16答案的题意核对

2026-10-24用户明确：以decode答案是否符合要求为准，不要求对齐Native token。

仅分析已有192token输出，不新跑设备，不将Native当标准答案。16个问题各重复16个rank，不能当成256个独立质量样本。

| 问题 | 类型 | Native | HCA MPMD |
|---|---|---|---|
| h131072_q00 | python_trace | NOT_SCORED | NOT_SCORED |
| h131072_q01 | transition_trace | NOT_SCORED | NOT_SCORED |
| h131072_q02 | python_trace | WRONG_NUMBERS | WRONG_NUMBERS |
| h131072_q03 | ledger_transform | WRONG_NUMBERS | WRONG_NUMBERS |
| h131072_q04 | transition_trace | NOT_SCORED | NOT_SCORED |
| h131072_q05 | easy_repeat | PREFIX_ONLY | PREFIX_ONLY |
| h131072_q06 | python_trace | NOT_SCORED | NOT_SCORED |
| h131072_q07 | transition_trace | NOT_SCORED | NOT_SCORED |
| h131072_q08 | python_trace | WRONG_NUMBERS | NOT_SCORED |
| h131072_q09 | ledger_transform | NOT_SCORED | NOT_SCORED |
| h131072_q10 | transition_trace | NOT_SCORED | NOT_SCORED |
| h131072_q11 | easy_repeat | PREFIX_ONLY | PREFIX_ONLY |
| h131072_q12 | python_trace | WRONG_NUMBERS | NOT_SCORED |
| h131072_q13 | transition_trace | NOT_SCORED | NOT_SCORED |
| h131072_q14 | python_trace | WRONG_NUMBERS | NOT_SCORED |
| h131072_q15 | ledger_transform | NOT_SCORED | NOT_SCORED |

WRONG_NUMBERS表示已生成的明确数值错误；PREFIX_ONLY只表示生成前缀符合要求；
NOT_SCORED表示当前没有可可靠解析的完整数值行，不等于正确或错误。
长解释/未完成输出不计算整题准确率。此轮记录不能证明HCA导致了新增答案错误，也不能宣布答案质量验收通过。

## h131072_q02首条明确数值错误

- native: `{"observed": [1, 20, 92, 63], "expected": [1, 9, 5, 24]}`
- hca: `{"observed": [1, 20, 100, 71], "expected": [1, 9, 5, 24]}`

## h131072_q03首条明确数值错误

- native: `{"observed": {"id": "f1", "net": 4739, "score": 1008, "tag": "C"}, "expected": {"id": "n0", "net": 3069, "score": 965, "tag": "A"}}`
- hca: `{"observed": {"id": "f1", "net": 4739, "score": 1007, "tag": "C"}, "expected": {"id": "n0", "net": 3069, "score": 965, "tag": "A"}}`

## h131072_q08首条明确数值错误

- native: `{"observed": [1, 22, 135, 40], "expected": [1, 13, 175, 156]}`
- hca: `null`

## h131072_q12首条明确数值错误

- native: `{"observed": [1, 23, 100, 50], "expected": [1, 15, 15, 35]}`
- hca: `null`

## h131072_q14首条明确数值错误

- native: `{"observed": [1, 24, 178, 8], "expected": [1, 17, 207, 234]}`
- hca: `null`
