# SPDX-License-Identifier: Apache-2.0
"""在同一批次提交不同问题，保存逐请求token与真实接受长度分布。"""
import json
import os
import time
from pathlib import Path


def wait_for_generation_peers(output, rank, offset, timeout=300):
    """在客户端等齐，保持EngineCore继续服务其他DP的EP通信；不能把worker堵在RPC barrier。"""
    (output / f'mixed_done_{offset}_rank{rank}').touch(exist_ok=False)
    deadline = time.monotonic() + timeout
    while True:
        pending = [i for i in range(16) if not (output / f'mixed_done_{offset}_rank{i}').exists()]
        if not pending:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f'混合请求完成等待超时，未完成DP={pending}')
        time.sleep(0.1)


def run_batches(llm, args, cases):
    from offline_pd.batch import generate_aligned_batch
    from offline_pd.run import spec_decode_metrics, write_json
    from vllm import SamplingParams

    chosen = cases if args.save_extended_bank else cases[:args.batch]
    if not args.save_extended_bank and len(chosen) != args.batch:
        raise ValueError('混合验证必须有足够的不同请求填满batch')
    records = []
    respect_eos = getattr(args, 'respect_eos', False)
    implementations = llm.collective_rpc('offline_attention_implementations')
    for offset in range(0, len(chosen), args.batch):
        chunk = chosen[offset:offset + args.batch]
        prompts, params = [], []
        for case in chunk:
            tokens = json.loads((args.bank / case['tokens']).read_text())
            prompts.append({'prompt_token_ids': tokens[:-1] if args.save_extended_bank else tokens})
            params.append(SamplingParams(temperature=0, max_tokens=args.decode_tokens, ignore_eos=not respect_eos,
                                         extra_args={'kv_transfer_params': {'offline_key': case['key']}}))
        llm.collective_rpc('offline_begin_observation')
        outputs = generate_aligned_batch(llm, prompts, params)
        # 低接受率时各DP的结束轮数可能不同。先完成的客户端不能提前结束观察并关闭
        # EngineCore，否则还在生成的DP会失去EP通信参与者。本屏障不在forward或采样内。
        wait_for_generation_peers(args.output, args.rank, offset)
        observation = llm.collective_rpc('offline_end_observation')
        for case, output in zip(chunk, outputs):
            records.append({'key': case['key'], 'kind': case['question_kind'],
                            'request_id': output.request_id,
                            'output_token_ids': list(output.outputs[0].token_ids),
                            'output_text': output.outputs[0].text,
                            'finish_reason': output.outputs[0].finish_reason,
                            'stop_reason': output.outputs[0].stop_reason, 'finished': output.finished})
        write_json(args.output / f'rank{args.rank}.mixed.json', {
            'rank': args.rank, 'backend': args.backend, 'variant': os.environ.get('PTO_CSA_VARIANT'),
            'pto_attention': args.pto_attention, 'attention_implementations': implementations,
            'batch': args.batch, 'decode_tokens': args.decode_tokens, 'requests': records,
            'worker_runtime_config': args.worker_runtime_config,
            'atomic_add': os.environ.get('VLLM_ASCEND_PTO_CSA_ATOMIC_ADD'),
            'pto_runtime': os.environ.get('PTO_CSA_RUNTIME'), 'deterministic': args.deterministic,
            'hccl_deterministic': os.environ.get('HCCL_DETERMINISTIC'),
            'observation': observation, 'spec_decode': spec_decode_metrics(llm),
            'sampling': {'temperature': 0.0, 'ignore_eos': not respect_eos, 'max_tokens': args.decode_tokens},
            'tokenizer_eos_token_id': llm.get_tokenizer().eos_token_id})


def collect(root, bank, batch, tokens, mean_range=(3.0, 4.0), respect_eos=False):
    plan = json.loads((bank / 'plan.json').read_text())
    expected = [c['key'] for c in plan['cases'][:batch]]
    rows, errors, distributions = [], [], [0] * 6
    proposed_total = 0
    full_five_distribution = [0] * 6
    preemptions = []
    for rank in range(16):
        data = json.loads((root / f'rank{rank}.mixed.json').read_text())
        events = [json.loads(line) for line in (root / f'request_stats_rank{rank}.jsonl').read_text().splitlines()]
        by_key = {r['key']: r for r in events}
        if len(events) != batch or len(by_key) != batch or [r['key'] for r in data['requests']] != expected:
            raise ValueError(f'rank{rank}: 请求缺失/重复/顺序错误')
        counts = [0, 0, 0]
        per_pos = [0] * 5
        for request in data['requests']:
            record = by_key[request['key']]
            preemptions.append(record.get('preemptions'))
            # vLLM input_processor会给内部ID追加UUID；输出仍使用外部数字ID。
            if (record['request_id'] != request['request_id']
                    and not record['request_id'].startswith(request['request_id'] + '-')):
                raise ValueError(f'rank{rank}/{request["key"]}: 输出与调度器请求错配')
            histogram = [0] * 6
            for proposed, accepted in record['events']:
                if not 0 <= accepted <= proposed <= 5:
                    raise ValueError('非法草稿计数')
                histogram[accepted] += 1
                if proposed == 5:
                    full_five_distribution[accepted] += 1
                counts[0] += 1
                counts[1] += proposed
                proposed_total += proposed
                counts[2] += accepted
                for i in range(accepted):
                    per_pos[i] += 1
            actual_tokens = len(request['output_token_ids'])
            valid_length = 0 < actual_tokens <= tokens if respect_eos else actual_tokens == tokens
            if not valid_length or record['output_tokens'] != actual_tokens or not sum(histogram):
                raise ValueError(f'rank{rank}/{request["key"]}: token或草稿事件缺失')
            mean = sum(i * value for i, value in enumerate(histogram)) / sum(histogram)
            distributions = [a + b for a, b in zip(distributions, histogram)]
            rows.append({'rank': rank, 'key': request['key'], 'kind': request['kind'],
                         'accepted_length_histogram': histogram, 'accepted_per_draft': mean,
                         'drafts': len(record['events'])})
        metrics = data['spec_decode']
        actual = [metrics['counters'][k] for k in ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens')]
        if counts != actual or per_pos != metrics['num_accepted_tokens_per_pos']:
            errors.append(f'rank{rank}: 逐请求计数与原生Prometheus计数不一致')
    mean = sum(i * value for i, value in enumerate(distributions)) / sum(distributions)
    return {'qualified': not errors and all(full_five_distribution) and mean_range[0] <= mean <= mean_range[1],
            'criterion': '同批不同请求，实际接受长度0..5均覆盖；主批均值3..4/目标约3.8，其他batch允许1..5',
            'mean_range': list(mean_range),
            'mean_accepted': mean, 'accepted_length_histogram': distributions,
            'full_five_draft_histogram': full_five_distribution,
            'num_drafts': sum(distributions), 'num_draft_tokens': proposed_total,
            'num_accepted_tokens': sum(i * value for i, value in enumerate(distributions)),
            'acceptance_rate': sum(i * value for i, value in enumerate(distributions)) / proposed_total,
            'request_mean_min': min(r['accepted_per_draft'] for r in rows),
            'request_mean_max': max(r['accepted_per_draft'] for r in rows),
            'total_preemptions': (sum(preemptions) if all(v is not None for v in preemptions) else None),
            'requests_with_preemption': (sum(v > 0 for v in preemptions)
                                        if all(v is not None for v in preemptions) else None),
            'errors': errors, 'requests': rows}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--tokens', type=int, default=192)
    parser.add_argument('--respect-eos', action='store_true')
    parser.add_argument('--mean-range', nargs=2, type=float, default=(3.0, 4.0),
                        help='主批3 4；其他batch可指定1 5，不限制每个请求的平均值')
    args = parser.parse_args()
    report = collect(args.root, args.bank, args.batch, args.tokens, tuple(args.mean_range), args.respect_eos)
    (args.root / 'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'requests'}, ensure_ascii=False))
