# SPDX-License-Identifier: Apache-2.0
"""保存七档原始候选同轮三方五窗，包含回退，不替换成最终区间选择的数值。"""

import json
from pathlib import Path

from report_key_block import summarize

HERE=Path(__file__).resolve().parent


def main():
    jobs=json.loads((HERE/'SCORE_SPLIT_TASKS.json').read_text())
    reports={tag:dict(**summarize(HERE/f'score_split_store_{tag}_v1',info['task']),source=info['source'])
             for tag,info in jobs.items()}
    (HERE/'SCORE_SPLIT_ALL_PERFORMANCE.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
    for tag,r in reports.items():
        print(tag,r.get('median_us',r['status']))


if __name__=='__main__':
    main()
