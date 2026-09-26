"""CPU 比较既有 Native P bank 的 TP 副本有效前缀。

P 始终使用 Native，DSA KV 跨 TP 复制；D TP1 使用 tp0 副本。
此工具只定位输入数据差异，不将结果当作 PTO 或整模型验收，也不写回 bank 或计算 hash。
H4095 已有确定性 bank；是否复查按当前问题决定，不自动重扫全部 bank。

用法：加载 env-dsv4-0251rc1.sh 后，指定 --bank 与 --output。
可复用 bank 的路径见 DSV4_FLASH_CSA_OFFLINE_PD.md。
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "offline_pd"))


LAYER_RE = re.compile(r"layers\.(\d+)\.")


def layer_of(name):
    """取张量名里的层号；draft（mtp.*）返回 None。"""
    found = LAYER_RE.search(name)
    return int(found.group(1)) if found else None


def kind_of(name):
    """把张量名归到 swa／attn／state／indexer_state／indexer_key 之一。"""
    if "indexer.compressor.state_cache" in name:
        return "indexer_state"
    if "indexer.k_cache" in name:
        return "indexer_key"
    if "compressor.state_cache" in name:
        return "state"
    if "swa_cache" in name:
        return "swa"
    if "attn__" in name:
        return "attn"
    return "other"


def compare(reference, other, entry, history, ratio):
    """比较一份张量的有效前缀，返回差异的位置与量级。

    reference/other 都已经过 prefix_tensor 处理，所以前缀外的行已置零，
    这里看到的差异都在有效区内。
    """
    import torch

    from prefix import prefix_tensor

    blocks, block_size = entry["logical_blocks"], entry["block_size"]
    left = prefix_tensor(reference, blocks, block_size, history, ratio)
    right = prefix_tensor(other, blocks, block_size, history, ratio)
    same = torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    result = {"pages": len(blocks), "block_size": block_size,
              "logical_blocks": list(blocks), "bitwise_equal": same}
    if same:
        return result

    # 逐页定位：哪些页有差异、每页差多少个元素。
    flat_left = left.reshape(len(blocks), -1)
    flat_right = right.reshape(len(blocks), -1)
    unequal = flat_left.view(torch.uint8) != flat_right.view(torch.uint8)
    per_page = unequal.reshape(len(blocks), -1).any(dim=-1)
    differing = [index for index, flag in enumerate(per_page.tolist()) if flag]
    result["differing_page_indices"] = differing
    result["differing_pages"] = len(differing)
    result["differing_logical_blocks"] = [blocks[i] for i in differing]
    result["confined_to_last_page"] = differing == [len(blocks) - 1]

    # 行级定位只对最后一页做，量小且最能说明是不是尾部效应。
    last = len(blocks) - 1
    if last in differing:
        rows_left = left[last].reshape(block_size, -1)
        rows_right = right[last].reshape(block_size, -1)
        row_diff = (rows_left.view(torch.uint8) != rows_right.view(torch.uint8)
                    ).reshape(block_size, -1).any(dim=-1).tolist()
        hit = [index for index, flag in enumerate(row_diff) if flag]
        result["last_page_differing_rows"] = hit
        result["last_page_row_span"] = [min(hit), max(hit)] if hit else []

    if left.dtype.is_floating_point:
        a = left.to(torch.float64)
        b = right.to(torch.float64)
        delta = (a - b).abs()
        scale = torch.maximum(a.abs(), b.abs())
        # 只在两边都非零处看相对误差，避免 0 除放大成 inf。
        mask = scale > 0
        relative = (delta[mask] / scale[mask]) if mask.any() else delta.new_zeros(1)
        result.update(
            differing_elements=int((delta > 0).sum().item()),
            total_elements=int(delta.numel()),
            max_abs_diff=float(delta.max().item()),
            max_rel_diff=float(relative.max().item()) if mask.any() else 0.0,
        )
    else:
        # int8 indexer key：比的是量化后的码字，差一个码字就说明选中的内容变了。
        unequal_count = int((left != right).sum().item())
        result.update(differing_elements=unequal_count, total_elements=int(left.numel()),
                      max_abs_diff=float((left.to(int) - right.to(int)).abs().max().item()),
                      max_rel_diff=None)
    return result


def run_case(bank, key, history, expected, limit):
    from safetensors.torch import load_file

    replicas = {}
    for tp in range(4):
        folder = bank / key / f"tp{tp}"
        replicas[tp] = (json.loads((folder / "manifest.json").read_text()),
                        load_file(str(folder / "cache.safetensors")))
    base_manifest, base_tensors = replicas[0]

    findings = {}
    for tp in range(1, 4):
        manifest, tensors = replicas[tp]
        per_tensor = {}
        for name, ratio in expected.items():
            entry = base_manifest["entries"][name]
            if manifest["entries"][name]["logical_blocks"] != entry["logical_blocks"]:
                per_tensor[name] = {"error": "logical block set differs"}
                continue
            outcome = compare(base_tensors[name], tensors[name], entry, history, ratio)
            if not outcome["bitwise_equal"]:
                per_tensor[name] = outcome
        findings[f"tp{tp}"] = per_tensor

    return summarize(findings, limit)


def summarize(findings, limit):
    """把逐张量结果压成可读的分布：按种类、按层、按是否只在尾页。"""
    report = {"per_replica": {}}
    for replica, per_tensor in findings.items():
        by_kind, by_layer, tail_only = {}, {}, 0
        worst = []
        for name, outcome in per_tensor.items():
            if "error" in outcome:
                continue
            kind = kind_of(name)
            by_kind[kind] = by_kind.get(kind, 0) + 1
            layer = layer_of(name)
            if layer is not None:
                by_layer[layer] = by_layer.get(layer, 0) + 1
            if outcome.get("confined_to_last_page"):
                tail_only += 1
            worst.append((outcome.get("max_rel_diff") or 0.0, name, outcome))
        worst.sort(key=lambda item: item[0], reverse=True)
        layers = sorted(by_layer)
        report["per_replica"][replica] = {
            "differing_tensors": len(per_tensor),
            "by_kind": dict(sorted(by_kind.items())),
            "differing_layers": layers,
            "first_differing_layer": layers[0] if layers else None,
            "tensors_confined_to_last_page": tail_only,
            "by_layer": {str(k): by_layer[k] for k in layers},
            "worst_by_relative_difference": [
                {"name": name, **{k: v for k, v in outcome.items()
                                  if k not in ("logical_blocks", "differing_page_indices",
                                               "last_page_differing_rows")}}
                for _, name, outcome in worst[:limit]],
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", help="只比较某一个 case，默认比较 plan 里的第一个")
    parser.add_argument("--top", type=int, default=8, help="每个副本列出多少条最大相对差异")
    args = parser.parse_args()

    from prefix import cache_contract

    plan = json.loads((args.bank / "plan.json").read_text())
    config = json.loads((Path(plan["model"]) / "config.json").read_text())
    expected = cache_contract(config)

    cases = plan["cases"]
    chosen = next((c for c in cases if c["key"] == args.case), None) if args.case else cases[0]
    if chosen is None:
        raise SystemExit(f"case {args.case} not in plan")

    report = {"bank": str(args.bank), "case": chosen["key"], "history": chosen["history"],
              "tensors_in_contract": len(expected),
              "note": "P 侧为 Native TP4；DSA 的 KV 跨 TP 复制而非切分，D 侧只读 tp0",
              **run_case(args.bank, chosen["key"], chosen["history"], expected, args.top)}

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"replica_diff_{chosen['key']}.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({replica: {k: v for k, v in value.items()
                                if k != "worst_by_relative_difference"}
                      for replica, value in report["per_replica"].items()},
                     indent=2, ensure_ascii=False), flush=True)
    print(f"\n完整报告：{target}", flush=True)


if __name__ == "__main__":
    main()
