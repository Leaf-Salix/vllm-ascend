# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Validate paired inputs and summarize saved precision artifacts, without NPU."""

import argparse
import json
from pathlib import Path

import torch
from fixture import difference


def summarize(root):
    summary = {}
    for context in ("8186", "131066"):
        folder = root / context
        records = {p.parent.name: json.loads(p.read_text()) for p in folder.glob("*/result.json")}
        expected = {
            "control",
            "qa",
            "q_mm",
            "q_rms",
            "q_pair",
            "kv_mm",
            "kv_rms",
            "kv_pair",
            "oa32_group",
            "oa16_group",
            "oa32_global",
            "oa16_global",
            "oa16_global_intsum",
        }
        if set(records) != expected:
            raise RuntimeError(
                f"Incomplete matrix {context}: missing={expected - set(records)}, extra={set(records) - expected}"
            )
        for name, record in records.items():
            if "error" in record:
                raise RuntimeError(f"{context}/{name}: {record['error']}")
        reference = records["control"]["result"]
        control = torch.load(folder / "control/stages.pt", map_location="cpu", weights_only=True)
        rows = {}
        for name, record in sorted(records.items()):
            result = record["result"]
            assert result["weight_unchanged"] and result["graph_replay"] and result["guard"]
            assert record["runtime"]["deterministic_level"] == 1
            assert record["runtime"]["HCCL_DETERMINISTIC"] == "true"
            stages = torch.load(folder / name / "stages.pt", map_location="cpu", weights_only=True)
            metrics = dict(result["metrics"])
            assert all(metric["finite"] for metric in metrics.values()), f"Non-finite stage: {context}/{name}"
            if record["mode"] == "qkv":
                assert reference["inputs"] == result["inputs"], f"Unpaired inputs: {context}/{name}"
                assert reference["native_hashes"] == result["native_hashes"], f"Native drift: {context}/{name}"
                # Q-only factors cannot write KV/compressor/indexer cache differently.
                if name in ("qa", "q_mm", "q_rms", "q_pair"):
                    for cache in control["cache"]:
                        assert torch.equal(control["cache"][cache]["pto"], stages["cache"][cache]["pto"]), cache
                if name in ("kv_mm", "kv_rms", "kv_pair"):
                    for key in ("qr", "qr_scale", "q"):
                        assert torch.equal(control["pto"][key], stages["pto"][key]), key
                    for cache in ("compressed", "main_state", "inner_state", "index_key", "index_scale"):
                        assert torch.equal(control["cache"][cache]["pto"], stages["cache"][cache]["pto"]), cache
                delta = {
                    key: difference(stages["pto"][key], control["pto"][key])
                    for key in ("qr", "q", "raw_kv", "heads", "output")
                }
                rows[name] = dict(metrics=metrics, versus_control=delta, independent_inputs_verified=True)
            else:
                assert result["weights_sha256"] == reference["inputs"]["weights"], "O-proj weight drift"
                assert result["heads_sha256"] == reference["native_hashes"]["heads"], "O-proj heads drift"
                quant = stages["pto"]["quant"].float().reshape(-1, 8, 1024)
                scales = stages["pto"]["scale"].T.unsqueeze(-1)
                actual = (quant * scales).reshape(-1, 8192)
                expected = stages["native"]["quant"].float() * stages["native"]["scale"].reshape(-1, 1)
                metrics["dequantized_activation"] = difference(actual, expected)
                rows[name] = dict(metrics=metrics, fixed_heads_and_weights_verified=True)
        # Only these edges isolate one factor; diagonals of the 2x2 are not causal comparisons.
        edges = {}
        for a, b, factor in [
            ("q_mm", "q_pair", "Q RMS BF16 after Q-B BF16"),
            ("kv_mm", "kv_pair", "KV RMS BF16 after KV projection BF16"),
            ("oa32_group", "oa16_group", "OA BF16 at group quant"),
            ("oa32_global", "oa16_global", "OA BF16 at global quant"),
            ("oa32_group", "oa32_global", "quant scope at FP32 OA"),
            ("oa16_group", "oa16_global", "quant scope at BF16 OA"),
            ("oa16_global", "oa16_global_intsum", "INT32 reduction"),
        ]:
            if a in records and b in records:
                sa = torch.load(folder / a / "stages.pt", map_location="cpu", weights_only=True)
                sb = torch.load(folder / b / "stages.pt", map_location="cpu", weights_only=True)
                edges[f"{a}->{b}"] = dict(
                    factor=factor, output_difference=difference(sb["pto"]["output"], sa["pto"]["output"])
                )
        summary[context] = dict(rows=rows, single_factor_edges=edges)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(result, indent=2))
    for context, group in result.items():
        print(context)
        for name, row in group["rows"].items():
            print(
                name,
                {
                    key: round(row["metrics"][key]["relative_l2"] * 100, 6)
                    for key in ("qr", "q", "raw_kv", "heads", "output")
                },
            )
