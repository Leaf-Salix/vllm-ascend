# SPDX-License-Identifier: Apache-2.0
"""仅汇总完整终态通过的三方五窗，不用CPU证据或局部计时代替性能。"""

import json
from pathlib import Path

from report_key_block import summarize

HERE=Path(__file__).resolve().parent
TASKS={16:'task_20261020_050812_247206811437',28:'task_20261020_050736_244883023864'}


def main():
    results={str(b):summarize(HERE/f'score_split_store_h131072_b{b}_v1',task) for b,task in TASKS.items()}
    (HERE/'SCORE_SPLIT_STORE_PERFORMANCE.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    for b,r in results.items():
        print(b,r.get('median_us',r['status']))


if __name__=='__main__':
    main()
