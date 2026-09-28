"""只覆盖新排序边界和tie：独立值/索引参考，并逐bit比较原4096排序。"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path)
parser.add_argument("--compile-only", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.source / "tests/pypto_test"))
from dsv4_csa_env import activate  # noqa: E402

os.environ["VLLM_ASCEND_ENABLE_NZ"] = "2"
os.environ["PTO_CSA_VARIANT"] = "performance"
os.environ["VLLM_ASCEND_PTO_CSA_ATOMIC_ADD"] = "0"
activate()
pl = importlib.import_module("pypto.language")
RunConfig = importlib.import_module("pypto.runtime").RunConfig
indexer_topk_half_leaf = importlib.import_module(
    "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_indexer"
).indexer_topk_half_leaf


@pl.jit
def sort_probe(
    scores: pl.Tensor[[12, 8192], pl.FP32],
    counts: pl.Tensor[[10], pl.INT32],
    output: pl.InOut[pl.Tensor[[12, 1024], pl.FP32]],
):
    for case in pl.spmd(10, name_hint="topk_half_leaf_probe"):
        valid = pl.cast(pl.read(counts, [case]), pl.INDEX)
        indexer_topk_half_leaf(scores, output, case + 1, case * 8192, valid, case + 1)
    return output


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    compiled = sort_probe.compile(
        config=RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output / "compiled"))
    )
    compiled.load()
    if args.compile_only:
        print("COMPILE_PASS ten-case sorting probe; no device execution")
        return

    import torch

    torch.set_num_threads(4)
    device = int(os.environ["TASK_DEVICE"])
    torch.npu.set_device(device)
    generator = torch.Generator().manual_seed(20260928)
    lengths = (2048, 2049, 2560, 2561, 3072, 3073, 3072, 2560, 3072, 2561)
    names = ("old_2048", "new_2049", "new_2560", "new_2561", "new_3072", "old_3073",
             "all_equal_3072", "random_ties_2560", "right_dominant_3072", "left_dominant_2561")
    scores = torch.full((12, 8192), -3e38, dtype=torch.float32)
    for case, length in enumerate(lengths):
        row = torch.randn(length, generator=generator)
        if case == 6:
            row.fill_(7)
        elif case == 7:
            row = torch.randint(0, 17, (length,), generator=generator).float()
        elif case == 8:
            row[2048:] += 100
        elif case == 9:
            row[:2048] += 100
        scores[case + 1, :length] = row
    initial_scores = scores.clone()
    output = torch.full((12, 1024), -12345.0)
    compiled(scores, torch.tensor(lengths, dtype=torch.int32), output,
             config=RunConfig(platform="a2a3", device_id=device))
    reference = None
    if args.reference:
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    checks = []
    for case, (name, length) in enumerate(zip(names, lengths)):
        pairs = output[case + 1]
        values = pairs[::2]
        indices = pairs.view(torch.int32)[1::2].long() - case * 8192
        torch.testing.assert_close(values, scores[case + 1, :length].sort(descending=True).values[:512],
                                   rtol=0, atol=0)
        assert bool(((indices >= 0) & (indices < length)).all()), name
        assert indices.unique().numel() == 512, name
        torch.testing.assert_close(values, scores[case + 1, indices], rtol=0, atol=0)
        if reference is not None:
            torch.testing.assert_close(pairs.view(torch.int32), reference[case + 1].view(torch.int32),
                                       rtol=0, atol=0)
        checks.append({"name": name, "valid_count": length, "status": "PASS"})
    torch.testing.assert_close(scores, initial_scores, rtol=0, atol=0)
    assert bool((output[[0, 11]] == -12345.0).all()), "guard rows changed"
    torch.save(output, args.output / "pairs.pt")
    report = {"status": "PASS", "device": device, "source": str(args.source), "cases": checks,
              "guards": "PASS", "baseline_bit_comparison": "PASS" if reference is not None else "reference"}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
