"""Native-only A3 cache layout check; no PyPTO, model loading or performance claim."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.ops.load_library(str(args.library))
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    generator = torch.Generator().manual_seed(20260927)
    batch, queries, heads, dim, page_rows = 2, 6, 64, 128, 32
    columns, pages, guard = 257, 515, 128
    total_bytes = pages * (page_rows * dim + page_rows * 2)
    allocations = [torch.full((total_bytes + 2 * guard,), 37, dtype=torch.int8, device=device)
                   for _ in range(2)]
    old_key = allocations[0].as_strided((pages, 32, 1, 128), (4160, 128, 128, 1), guard)
    old_scale = allocations[0].view(torch.float16).as_strided((pages, 32, 1, 1), (2080, 1, 1, 1), (guard + 4096) // 2)
    raw = allocations[1][guard:-guard]
    new_key = raw[:pages * 4096].view(pages, 32, 1, 128)
    new_scale = raw[pages * 4096:].view(torch.float16).view(pages, 32, 1, 1)
    keys = torch.randint(-64, 64, old_key.shape, dtype=torch.int8, generator=generator)
    scales = (torch.rand(old_scale.shape, generator=generator) * 0.01 + 0.001).half()
    for key, scale in ((old_key, old_scale), (new_key, new_scale)):
        key.copy_(keys)
        scale.copy_(scales)
    query = torch.randint(-64, 64, (batch * queries, heads, dim), dtype=torch.int8, generator=generator).to(device)
    query_scale = (torch.rand((batch * queries, heads), generator=generator) * 0.01 + 0.001).half().to(device)
    weights = torch.rand((batch * queries, heads), generator=generator).half().to(device)
    qlens = torch.tensor([6, 12], dtype=torch.int32, device=device)
    klens = torch.tensor([8198, 4102], dtype=torch.int32, device=device)
    table_cpu = (torch.randperm(pages - 1, generator=generator) + 1).view(batch, columns).to(torch.int32)
    table = table_cpu.to(device)
    metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
        num_heads_q=heads, num_heads_k=1, head_dim=dim, query_quant_mode=0, key_quant_mode=0,
        actual_seq_lengths_query=qlens, actual_seq_lengths_key=klens,
        batch_size=batch, max_seqlen_q=queries, max_seqlen_k=8198,
        layout_query='TND', layout_key='PA_BSND', sparse_count=512, cmp_ratio=4, device=device)
    logical_slots = [(0, 2047), (0, 2048), (1, 1023), (1, 1024)]
    slots = torch.tensor([[int(table_cpu[b, row // page_rows]), row % page_rows]
                          for b, row in logical_slots], dtype=torch.int32, device=device)
    key_update = torch.randint(-64, 64, (4, 1, dim), dtype=torch.int8, generator=generator).to(device)
    scale_update = (torch.rand((4, 1, 1), generator=generator) * 0.01 + 0.001).half().to(device)

    def invoke(key, scale):
        torch.ops._C_ascend.npu_scatter_nd_update_v2(key, slots, key_update)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(scale, slots, scale_update)
        return torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
            query, key, weights, query_scale, scale.squeeze(-2),
            actual_seq_lengths_query=qlens, actual_seq_lengths_key=klens,
            block_table=table, metadata=metadata, layout_query='TND', layout_key='PA_BSND',
            sparse_count=512, cmp_ratio=4)[0]

    records = []

    def check(label, outputs):
        torch.npu.synchronize()
        output_cpu = [t.cpu() for t in outputs]
        checks = {'key_exact': torch.equal(old_key.cpu(), new_key.cpu()),
                  'scale_bits_exact': torch.equal(old_scale.cpu().view(torch.int16), new_scale.cpu().view(torch.int16)),
                  'topk_exact': torch.equal(*output_cpu),
                  'guards_intact': all(bool(torch.all(torch.cat((a[:guard], a[-guard:])).cpu() == 37)) for a in allocations),
                  'topk_not_empty': bool((output_cpu[0] >= 0).any())}
        records.append({'phase': label, **checks})
        assert all(checks.values()), records[-1]

    caches = ((old_key, old_scale), (new_key, new_scale))
    outputs = [invoke(*cache) for cache in caches]
    check('eager', outputs)
    graphs, graph_outputs = [], []
    for cache in caches:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_outputs.append(invoke(*cache))
        graphs.append(graph)
    for step in range(2):
        # Change device buffer contents at fixed addresses between graph replays.
        key_update.copy_(torch.randint(-64, 64, (4, 1, dim), dtype=torch.int8, generator=generator))
        scale_update.copy_((torch.rand((4, 1, 1), generator=generator) * 0.01 + 0.001).half())
        for graph in graphs:
            graph.replay()
        check(f'graph_replay_{step}', graph_outputs)
    report = {'status': 'PASS', 'scope': 'Native scatter + QLI; B2/S6, mixed 8K/4K, nonconsecutive physical pages, same backing allocation with planar key/scale views; no full-model or performance claim',
              'old_strides': [list(t.stride()) for t in caches[0]],
              'new_strides': [list(t.stride()) for t in caches[1]],
              'checks': records, 'library': str(args.library)}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
