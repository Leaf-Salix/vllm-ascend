# SPDX-License-Identifier: Apache-2.0
"""用来源已确认的旧输入构造单卡 case；不把丢失的原始初态/别名假装恢复出来。"""

import argparse
import importlib
import json
from pathlib import Path

from dsv4_csa_env import activate
from dsv4_csa_replay import argument_roles, capture_tensors, save_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True,
                        help="明确声明 source_variant、source_weight_nz_mode、weight_layouts 和 evidence")
    parser.add_argument("--state-mode", choices=("zero", "captured_post"), required=True,
                        help="zero 构造零历史状态；captured_post 把旧执行后状态显式当作新 case 初态")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    declaration = json.loads(args.declaration.read_text())
    layouts = declaration["weight_layouts"]
    if set(layouts) != {"wq_a", "wq_b", "wo_a", "wo_b"} or any(v not in ("ND", "NZ") for v in layouts.values()):
        parser.error("必须明确四张权重的实际 ND/NZ 来源布局")
    if not declaration.get("evidence") or declaration.get("source_variant") not in ("precision", "performance"):
        parser.error("必须提供来源版本及日志/记录证据；不能根据当前开关猜测")
    mode = declaration.get("source_weight_nz_mode")
    if mode not in (0, 1, 2) or (mode == 0 and any(v != "ND" for v in layouts.values())):
        parser.error("来源 mode 与布局声明不一致")
    legacy_meta = json.loads((args.legacy_dir / "csa_args_meta.json").read_text())
    if "schema_version" in legacy_meta:
        parser.error("已有 schema 的快照直接使用回放入口，不通过旧数据构造器处理")
    activate()
    import torch
    from vllm_ascend.ops.pypto.variant import variant_package

    roots = importlib.import_module(f"{variant_package()}.decode_csa")
    roles = argument_roles(roots._decode_csa_tp1_layer)
    if list(roles) != legacy_meta["param_names"]:
        raise ValueError("旧根 ABI 与当前不一致，不能自动迁移")
    values = torch.load(args.legacy_dir / "csa_args.pt", map_location="cpu", weights_only=True)
    values = {name: values[name] for name in roles}
    for name, role in roles.items():
        if role == "out" or (role == "inout" and args.state_mode == "zero"):
            values[name] = torch.zeros_like(values[name])
    source = {"state_timing": "constructed_initial", "variant": declaration["source_variant"],
              "weight_nz_mode": mode, "legacy_directory": str(args.legacy_dir.resolve()),
              "legacy_state_timing": "after_call", "state_mode": args.state_mode,
              "alias_policy": "旧快照已独立 clone；新 case 使用独立入参，不声称保留 Native 原始别名",
              "evidence": declaration["evidence"],
              "scope": "构造的单卡诊断 case；旧输出不作为参考，不能用于宣称原整模型正确性"}
    meta, payload = capture_tensors(values, roles, layouts, source)
    meta.update({key: legacy_meta[key] for key in ("layer_index", "layer_name", "tokens")})
    save_snapshot(args.output, meta, payload)
    print(json.dumps({"output": str(args.output), "tokens": meta["tokens"],
                      "state_mode": args.state_mode, "scope": source["scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
