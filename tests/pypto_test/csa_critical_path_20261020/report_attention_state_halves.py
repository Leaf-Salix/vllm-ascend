# SPDX-License-Identifier: Apache-2.0
"""完整三方五窗终态汇总；尚未完成的任务不计性能，保留逐窗配对。"""

import json
from pathlib import Path

from report_key_block import summarize

HERE = Path(__file__).resolve().parent
TASKS = {16: 'task_20261020_043934_14322076168', 28: 'task_20261020_043919_14214894072'}


def main():
    results={str(batch):summarize(HERE/f'attention_state_h131072_b{batch}_v1',task)
             for batch,task in TASKS.items()}
    (HERE/'ATTENTION_STATE_HALVES_PERFORMANCE.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    for batch,result in results.items():
        print(batch,result.get('median_us',result['status']))


if __name__=='__main__':
    main()
