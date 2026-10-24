# SPDX-License-Identifier: Apache-2.0
"""正常EOS任务的执行完整性检查；答案正确性另由题意判分，不将结束视为正确。"""
from collections import Counter
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def main(root):
    config = read(root / 'configuration.json')
    assert config['batch'] == 16 and config['decode_tokens'] == 2048 and config['ignore_eos'] is False
    side = config['backend']
    ranks, canonical, preemptions = [], None, 0
    for rank in range(16):
        data = read(root / f'rank{rank}.mixed.json')
        assert data['sampling'] == dict(temperature=0.0, ignore_eos=False, max_tokens=2048)
        assert len(data['requests']) == 16 and data['backend'] == side
        implementations = data['attention_implementations'][0]
        hca = [r for r in implementations if r['kind'] == 'hca']
        csa = [r for r in implementations if r['kind'] == 'csa']
        assert len(hca) == 20 and len(csa) == 21
        assert all(r['implementation'] == 'native' for r in csa)
        assert all(r['implementation'] == ('pto' if side == 'pto' else 'native') for r in hca)
        events = {r['key']: r for r in (json.loads(line) for line in
                   (root / f'request_stats_rank{rank}.jsonl').read_text().splitlines())}
        assert len(events) == 16
        reasons, tokens = Counter(), 0
        for row in data['requests']:
            count = len(row['output_token_ids'])
            assert row['finished'] and 0 < count <= 2048
            assert row['finish_reason'] in ('stop','length')
            if row['finish_reason'] == 'length':
                assert count == 2048
            event = events[row['key']]
            assert event['output_tokens'] == count, (rank,row['key'],event['output_tokens'],count)
            assert event['preemptions'] == 0
            preemptions += event['preemptions']
            reasons[row['finish_reason']] += 1
            tokens += count
        observation = data['observation'][0]
        if side == 'pto':
            routes = [r for r in observation['hca_execution_routes'] if r['requested_mode'] == 'FULL'
                      and r['max_seq_len'] >= 131072 and r['actual_tokens'] > 0]
            assert routes and all(r['all_hca_graph_replayed'] and not r['python_resubmitted'] for r in routes)
        values = [(r['key'],r['output_token_ids'],r['finish_reason']) for r in data['requests']]
        if canonical is None:
            canonical = values
        ranks.append(dict(rank=rank, finish_reasons=dict(reasons), output_tokens=tokens,
                          same_as_rank0=values == canonical, memory=observation['memory']))
    result = dict(status='EXECUTION_PASS_QUALITY_NOT_SCORED', backend=side, requests=256,
                  output_tokens=sum(r['output_tokens'] for r in ranks), preemptions=preemptions,
                  all_dp_outputs_equal=all(r['same_as_rank0'] for r in ranks), ranks=ranks)
    (root/'quality_execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'ranks'},ensure_ascii=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
