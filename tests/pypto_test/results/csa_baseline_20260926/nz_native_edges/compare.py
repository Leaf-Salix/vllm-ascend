"""本阶段两组边界用例各自 ND/NZ 的八类状态对照；跨实现精度不在本脚本判定。"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from dsv4_csa_validation import compare_tensor  # noqa: E402

REQUIRED = frozenset(("x_out", "idx_topk", "swa.0", "compressed.0", "state.0",
                      "indexer.0", "indexer.1", "indexer_state.0"))


def compare_layouts():
    result = {"status": "PASS", "scope": "两版各一个尾块/上下文边界；不代表全部档位或跨实现验收", "cases": {}}
    for case in ("short_b1", "long_b5"):
        reports = [json.loads((ROOT / f"{case}_mode{mode}/report.json").read_text()) for mode in (0, 2)]
        settings = ("batch", "history", "seed", "variant", "deterministic_level", "pto_reduction")
        if any(reports[0][key] != reports[1][key] for key in settings):
            raise ValueError(f"{case} 两档配置不一致")
        for mode, report in zip((0, 2), reports):
            if report["status"] != "MEASURED" or report["effective_weight_nz_mode"] != mode:
                raise ValueError(f"{case} 未完成或实际 NZ mode 不符")
            if report["graph"]["status"] != "PASS":
                raise ValueError(f"{case} 图重放失败")
            for key in ("native_self", "pto_self"):
                if any(check["status"] != "PASS" for check in report[key].values()):
                    raise ValueError(f"{case}/{mode} 固定规约同初态不一致")
            for key in ("native_guards", "pto_guards"):
                if any(check["status"] != "PASS" for row in report[key] for check in row.values()):
                    raise ValueError(f"{case}/{mode} metadata 或保护区失败")
        states = [torch.load(ROOT / f"{case}_mode{mode}/states.pt", map_location="cpu", weights_only=True)
                  for mode in (0, 2)]
        entry = {key: reports[0][key] for key in settings}
        for side in ("native", "pto"):
            if any(set(state[side]) != REQUIRED for state in states):
                raise ValueError(f"{case}/{side} 缺少必需状态")
            entry[side] = {key: compare_tensor(states[1][side][key], states[0][side][key], 0, 0)
                           for key in sorted(REQUIRED)}
            if any(check["status"] != "PASS" for check in entry[side].values()):
                result["status"] = "FAIL"
        result["cases"][case] = entry
    (ROOT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["status"])
    if result["status"] != "PASS":
        raise SystemExit(1)


def diagnose_short():
    perf = torch.load(ROOT / "short_b1_mode2/states.pt", map_location="cpu", weights_only=True)
    prec = torch.load(ROOT / "short_b1_precision_mode2/states.pt", map_location="cpu", weights_only=True)
    if any(set(state[side]) != REQUIRED for state in (perf, prec) for side in ("native", "pto")):
        raise ValueError("短上下文对照缺少必需状态")
    native = {key: compare_tensor(prec["native"][key], value, 0, 0) for key, value in perf["native"].items()}
    if any(value["status"] != "PASS" for value in native.values()):
        raise ValueError("两版的 Native 基线不同，不能归因版本区别")
    result = {
        "status": "MEASURED",
        "scope": "B1/H255、mode=2、atomic=0，同一正式层与合成输入；"
                 "两版 Native 八类状态精确相同；PTO 误差来源尚未定位",
        "native_baseline_equality": native, "variants": {},
    }
    for name, state in (("performance", perf), ("precision", prec)):
        actual, reference = state["pto"]["x_out"].float(), state["native"]["x_out"].float()
        entry = compare_tensor(state["pto"]["x_out"], state["native"]["x_out"], 0, 0)
        entry.update(
            token_max_abs=(actual - reference).abs().flatten(1).max(1).values.tolist(),
            token_rmse=(actual - reference).square().flatten(1).mean(1).sqrt().tolist(),
            native_absmax=reference.abs().max().item(), native_rms=reference.square().mean().sqrt().item(),
        )
        result["variants"][name] = entry
    (ROOT / "short_precision_diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("MEASURED: 短上下文两版差异尚待归因")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-diagnostic", action="store_true", help="只比较保留的 B1 短上下文两版状态")
    args = parser.parse_args()
    diagnose_short() if args.short_diagnostic else compare_layouts()


if __name__ == "__main__":
    main()
