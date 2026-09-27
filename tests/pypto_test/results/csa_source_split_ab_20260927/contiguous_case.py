"""Diagnostic only: give both sides consecutive Indexer physical pages at initialization.

This measures the benefit available when allocation guarantees contiguous pages;
it does not claim the production block allocator already provides this guarantee.
"""

import sys
from pathlib import Path


def main():
    source = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(source / 'tests/pypto_test'))
    import dsv4_csa_single_layer as case

    original_fixture, original_write = case.make_fixture, case.write_json

    def fixture(*args, **kwargs):
        import dsv4_csa_native_case as allocation
        import torch

        from vllm_ascend.worker.block_table import BlockTable

        original_allocate, original_add = allocation.allocate_native_cache, BlockTable.add_row
        current_indexer = False

        def allocate(group, *a, **kw):
            nonlocal current_indexer
            current_indexer = bool(getattr(group['spec'], 'scale_dim', 0))
            return original_allocate(group, *a, **kw)

        def add(table, block_ids, row):
            # The original fixture is reverse ordered. Only Indexer changes.
            if current_indexer:
                block_ids = list(reversed(block_ids))
                assert all(b == block_ids[0] + i for i, b in enumerate(block_ids))
            return original_add(table, block_ids, row)

        allocation.allocate_native_cache, BlockTable.add_row = allocate, add
        try:
            result = original_fixture(*args, **kwargs)
        finally:
            allocation.allocate_native_cache, BlockTable.add_row = original_allocate, original_add
        if '--timing-iters' not in sys.argv:
            group = result['groups']['indexer']
            scale = group['views'][1]
            rows = torch.arange(scale.numel(), dtype=torch.int64, device=scale.device)
            scale.copy_((0.00390625 + ((rows * 37) % 251).float() / 16384).to(scale.dtype).reshape(scale.shape))
            group['initial'] = group['allocation'].cpu()
        return result

    def write(path, report):
        if Path(path).name == 'report.json':
            report['accuracy_fixture'] = {
                'indexer_page_table': 'consecutive physical pages per request; initialization only',
                'scale': '251 values per physical row' if '--timing-iters' not in sys.argv else '0.01',
                'limit': 'diagnostic allocation; no production lifecycle guarantee',
            }
        original_write(path, report)

    case.make_fixture, case.write_json = fixture, write
    if '--timing-iters' in sys.argv:
        import pressure_case

        pressure_case.main()
    else:
        sys.argv.pop(1)
        case.main()


if __name__ == '__main__':
    main()
