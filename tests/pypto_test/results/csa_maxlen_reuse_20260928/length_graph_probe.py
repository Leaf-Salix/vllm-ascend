"""Exercise the actual merge with changing lengths in one captured graph."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--compile-only", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.source / "tests/pypto_test"))
os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"
os.environ["PTO_CSA_VARIANT"] = "performance"
os.environ["VLLM_ASCEND_PTO_CSA_ATOMIC_ADD"] = "0"
importlib.import_module("dsv4_csa_env").activate()
pl = importlib.import_module("pypto.language")
RunConfig = importlib.import_module("pypto.runtime").RunConfig
indexer = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_indexer")
indexer_topk_query_merge = indexer.indexer_topk_query_merge
TOPK_ARENA_ROWS = indexer.TOPK_ARENA_ROWS


@pl.jit
def length_graph_probe(
    positions: pl.Tensor[[96], pl.INT64],
    lengths: pl.Tensor[[16], pl.INT32],
    pairs: pl.Tensor[[TOPK_ARENA_ROWS, 1024], pl.FP32],
    scores: pl.Out[pl.Tensor[[96, 512], pl.FP32]],
    indices: pl.Out[pl.Tensor[[96, 512], pl.INT32]],
):
    max_count = 0
    for batch in pl.range(16):
        max_count = pl.max(max_count, pl.read(lengths, [batch]) // 4)
    with pl.spmd(48, name_hint="length_graph_actual_merge"):
        indexer_topk_query_merge(positions, lengths, max_count, pairs, scores, indices, True)
    return scores, indices


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    if args.compile_only:
        compiled = length_graph_probe.compile(config=RunConfig(
            platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output / "compiled")))
        compiled.load()
        print("COMPILE_PASS length-change graph probe; actual candidate merge; no device execution")
        return

    import pypto.torch
    import torch

    torch.set_num_threads(4)
    device = int(os.environ["TASK_DEVICE"])
    torch.npu.set_device(device)
    pypto.torch.init(device=device, platform="a2a3")
    device_name = f"npu:{device}"
    # Each root contains 512 sorted, unique pairs; later roots outrank earlier
    # ones. For these two lengths the real balanced plans have 12 and 6 roots.
    pairs = torch.zeros((TOPK_ARENA_ROWS, 1024), dtype=torch.float32)
    roots = torch.arange(TOPK_ARENA_ROWS, dtype=torch.int32) % 64
    lanes = torch.arange(512, dtype=torch.int32)
    pairs[:, ::2] = roots[:, None] * 1024 + (512 - lanes)
    pairs.view(torch.int32)[:, 1::2] = roots[:, None] * 512 + lanes
    pairs = pairs.to(device_name)
    positions = torch.full((96,), 32769 * 4 - 1, dtype=torch.int64, device=device_name)
    lengths = torch.full((16,), 32769 * 4, dtype=torch.int32, device=device_name)
    score_guard = torch.full((98, 512), -12345.0, device=device_name)
    index_guard = torch.full((98, 512), -12345, dtype=torch.int32, device=device_name)
    scores, indices = score_guard[1:-1], index_guard[1:-1]

    def run():
        return length_graph_probe(positions, lengths, pairs, scores, indices)

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    checks = []
    for count, last_root in ((32769, 11), (16385, 5), (32769, 11)):
        lengths.fill_(count * 4)
        positions.fill_(count * 4 - 1)
        scores.fill_(float("nan"))
        indices.fill_(-777)
        graph.replay()
        torch.npu.synchronize()
        expected_scores = (last_root * 1024 + 512 - lanes).float().expand(96, -1)
        expected_indices = (last_root * 512 + lanes).expand(96, -1)
        torch.testing.assert_close(scores.cpu(), expected_scores, rtol=0, atol=0)
        torch.testing.assert_close(indices.cpu(), expected_indices, rtol=0, atol=0)
        assert bool((score_guard[[0, 97]].cpu() == -12345).all())
        assert bool((index_guard[[0, 97]].cpu() == -12345).all())
        checks.append({"compressed_length": count, "expected_last_root": last_root, "status": "PASS"})
    report = {"status": "PASS", "source": str(args.source), "device": device, "replays": checks,
              "scope": "actual merge; same-address length/position A-B-A; scalar parameter updates on replay",
              "limits": "synthetic sorted roots; not Score arithmetic or Native/model accuracy"}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
