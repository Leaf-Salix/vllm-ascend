"""五组独立排序参考：检查四路Top-K的tie、耗尽前缀和二/三路尾块。"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('CSA_TOPK_PROBE_OUTPUT', str(Path(__file__).resolve().parent)))
SOURCE = Path(os.environ.get(
    'CSA_TOPK_PROBE_SOURCE', '/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-topk-fourway-e58ddc94'))
ROOT_IN_UB = os.environ.get('CSA_TOPK_PROBE_ROOT_IN_UB', '0') == '1'
sys.path.insert(0, str(SOURCE / 'tests/pypto_test'))
from dsv4_csa_env import activate

os.environ['VLLM_ASCEND_ENABLE_NZ'] = '2'
os.environ['PTO_CSA_VARIANT'] = 'performance'
os.environ['VLLM_ASCEND_PTO_CSA_ATOMIC_ADD'] = '0'
activate()
pl = importlib.import_module('pypto.language')
RunConfig = importlib.import_module('pypto.runtime').RunConfig
merge_top512_roots = importlib.import_module(
    'vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_indexer').merge_top512_roots


# Select the ABI in Python: a runtime DSL branch cannot merge a void-return
# helper with a Tile-return helper while keeping the old probe reproducible.
if ROOT_IN_UB:
    @pl.jit.inline
    def probe_root(arena: pl.Tensor, root: pl.Scalar[pl.INDEX], count: pl.Scalar[pl.INDEX]) -> pl.Tile:
        return merge_top512_roots(arena, root, count)
else:
    @pl.jit.inline
    def probe_root(arena: pl.Tensor, root: pl.Scalar[pl.INDEX], count: pl.Scalar[pl.INDEX]) -> pl.Tile:
        merge_top512_roots(arena, root, count)
        return pl.load(arena, [root, 0], [1, 1024])


@pl.jit
def merge_probe(arena: pl.InOut[pl.Tensor[[50, 1024], pl.FP32]],
                output: pl.InOut[pl.Tensor[[7, 1024], pl.FP32]]):
    for case in pl.spmd(5, name_hint='topk_merge_probe'):
        root = case * 10
        pairs = probe_root(arena, root, (case + 1) * 2)
        pl.store(pairs, [case + 1, 0], output)
    return arena, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-only', action='store_true')
    args = parser.parse_args()
    config = RunConfig(platform='a2a3', save_kernels=True,
                       save_kernels_dir=str(ROOT / 'merge_compiled'))
    compiled = merge_probe.compile(config=config)
    compiled.load()
    if args.compile_only:
        print('COMPILE_PASS merge probe; no device execution')
        return

    import torch

    device = int(os.environ['TASK_DEVICE'])
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(20260928)
    arena = torch.empty(50, 1024, dtype=torch.float32)
    expected = []
    names = ('two_lists', 'random_ties', 'all_equal_three_way_tail',
             'dominant_list_two_way_tail', 'masked_tails')
    for case in range(5):
        count = (case + 1) * 2
        scores = torch.randint(0, 17, (10, 512), generator=generator).float()
        if case == 2:
            scores.fill_(7)
        elif case == 3:
            scores += torch.arange(10).view(10, 1) * 32
        elif case == 4:
            scores[:, 128:] = -3e38
        values, order = scores.sort(dim=1, descending=True, stable=True)
        indices = order.int() + torch.arange(10).view(10, 1) * 512 + case * 5120
        packed = arena[case * 10: (case + 1) * 10]
        packed[:, ::2] = values
        packed.view(torch.int32)[:, 1::2] = indices
        # Independent global stable sort: newer list first for equal score,
        # and the supplied order within each sorted list remains unchanged.
        valid_values = values[:count].flip(0).flatten()
        valid_indices = indices[:count].flip(0).flatten()
        selected = torch.argsort(valid_values, descending=True, stable=True)[:512]
        gold = torch.empty(1024, dtype=torch.float32)
        gold[::2] = valid_values[selected]
        gold.view(torch.int32)[1::2] = valid_indices[selected]
        expected.append(gold)
    initial = arena.clone()
    output = torch.full((7, 1024), -12345.0)
    compiled(arena, output, config=RunConfig(platform='a2a3', device_id=device))
    results = []
    for case, (name, gold) in enumerate(zip(names, expected)):
        torch.testing.assert_close(output[case + 1].view(torch.int32), gold.view(torch.int32), rtol=0, atol=0)
        root = case * 10
        if ROOT_IN_UB:
            torch.testing.assert_close(arena[root].view(torch.int32), initial[root].view(torch.int32), rtol=0, atol=0)
        else:
            torch.testing.assert_close(arena[root].view(torch.int32), gold.view(torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(arena[root + 1:root + 10].view(torch.int32),
                                   initial[root + 1:root + 10].view(torch.int32), rtol=0, atol=0)
        results.append({'case': name, 'half_count': (case + 1) * 2, 'status': 'PASS'})
    assert torch.all(output[[0, 6]] == -12345.0), 'output guard rows changed'
    (ROOT / 'merge_report.json').write_text(json.dumps(
        {'status': 'PASS', 'device': device, 'cases': results, 'guards': 'PASS'}, indent=2) + '\n')
    print('PASS', names)


if __name__ == '__main__':
    main()
