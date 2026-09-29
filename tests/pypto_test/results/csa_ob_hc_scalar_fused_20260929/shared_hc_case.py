"""检查共享HC单行入口的尾行、逐项算术、图输入更新和输出保护区。"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "tests/pypto_test"))
    from dsv4_csa_env import activate

    activate()
    import pypto.torch
    import torch
    import torch_npu  # noqa: F401

    module = importlib.import_module("vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.hc_post")
    if args.source.resolve() not in Path(module.__file__).resolve().parents:
        raise ValueError("Shared HC was not imported from the frozen private source")
    torch.set_num_threads(4)
    torch.manual_seed(1024)
    torch.npu.set_device(args.device)
    torch_npu.npu.set_deterministic_level(1)
    pypto.torch.init(device=args.device, platform="a2a3", aicpu_thread_num=4,
                     ring_heap=tuple(mb * 1024**2 for mb in (256, 128, 256, 32)),
                     ring_task_window=4096)
    tokens, hidden, streams = 18, module.D, module.HC_MULT
    x = torch.randn(tokens, hidden).to(torch.bfloat16)
    residual = torch.randn(tokens, streams, hidden).to(torch.bfloat16)
    post = torch.rand(tokens, streams)
    comb = torch.rand(tokens, streams * streams)
    cpu_inputs = [x, residual, post, comb]
    inputs = [value.to(f"npu:{args.device}") for value in cpu_inputs]
    storage = torch.full((tokens + 2, streams, hidden), -123, dtype=torch.bfloat16,
                         device=f"npu:{args.device}")
    output = storage[1:-1]

    def golden(value):
        # Separate mul then add, in the same four-residual order as the original HC.
        result = torch.empty(tokens, streams, hidden, dtype=torch.bfloat16)
        for out_h in range(streams):
            row = value.float() * post[:, out_h:out_h + 1]
            for in_h in range(streams):
                row = row + residual[:, in_h].float() * comb[:, in_h * streams + out_h:in_h * streams + out_h + 1]
            result[:, out_h] = row.to(torch.bfloat16)
        return result

    outputs, checks = {}, {}

    def check(name, value):
        torch.npu.synchronize()
        actual = output.cpu()
        expected = golden(value)
        if not torch.equal(actual, expected):
            delta = (actual.float() - expected.float()).abs()
            raise AssertionError(f"{name}: {(actual != expected).sum().item()} mismatches, max {delta.max().item()}")
        host_storage = storage.cpu()
        if not torch.all(host_storage[0] == -123) or not torch.all(host_storage[-1] == -123):
            raise AssertionError(f"{name}: output guard overwritten")
        for index, (device_value, original) in enumerate(zip(inputs, [value, *cpu_inputs[1:]])):
            if not torch.equal(device_value.cpu(), original):
                raise AssertionError(f"{name}: input {index} modified")
        outputs[name] = actual
        checks[name] = {"status": "PASS", "cpu_golden_zero_tolerance": True,
                        "output_guards": "PASS", "readonly_inputs": "PASS"}

    kernel = module.hc_post_test
    kernel(*inputs, output)
    check("eager", x)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        kernel(*inputs, output)
    output.fill_(-123)
    graph.replay()
    check("graph", x)
    updated = (x.float() + 0.125).to(torch.bfloat16)
    inputs[0].copy_(updated)
    output.fill_(-123)
    graph.replay()
    check("graph_updated_input", updated)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, args.output / "outputs.pt")
    report = {"status": "PASS", "source": str(args.source), "module": module.__file__,
              "device": args.device, "cann": os.environ["ASCEND_HOME_PATH"],
              "tokens": tokens, "hidden": hidden, "streams": streams, "seed": 1024,
              "checks": checks, "scope": "Shared HC only; not full precision CSA or model acceptance"}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("PASS shared HC T18, eager/graph/updated-input, exact CPU golden and guards", flush=True)


if __name__ == "__main__":
    main()
