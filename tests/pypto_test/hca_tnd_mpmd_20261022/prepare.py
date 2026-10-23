# SPDX-License-Identifier: Apache-2.0
"""提交前冻结 HCA 算子、共同数学依赖与测试入口，不复制 SDK 或二进制。"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--legacy-signature-compat", action="store_true")
    parser.add_argument("--precision-probe", action="store_true")
    parser.add_argument("--output-variant", type=Path, help="只在冻结包替换output模块，正式源码不变")
    parser.add_argument("--q-projection-variant", type=Path,
                        help="在冻结TND包内替换Q_B，保留公共HCA模块供同进程控制组使用")
    parser.add_argument("--attention-variant", type=Path, help="仅替换冻结TND包内的attention模块")
    parser.add_argument("--attention-probe", action="store_true", help="生成仅attention的独立诊断入口")
    parser.add_argument("--attention-probe-long", action="store_true", help="独立探针调用共享工厂的K512分支")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("实验名必须是单段目录名")
    target = HERE / "experiments" / args.name
    target.mkdir(parents=True, exist_ok=False)
    source = REPO / "vllm_ascend/ops/pypto"
    operator = target / "operator"
    operator.mkdir()
    for path in source.glob("*.py"):
        shutil.copy2(path, operator / path.name)
    for name in ("deepseek_v4_flash_csa", "deepseek_v4_flash_hca", "deepseek_v4_flash_hca_tnd"):
        (operator / name).mkdir()
        for path in (source / name).glob("*.py"):
            shutil.copy2(path, operator / name / path.name)
    if args.output_variant:
        output_module = operator / "deepseek_v4_flash_hca_tnd/output.py"
        shutil.copy2(output_module, output_module.with_name("output_base.py"))
        shutil.copy2(args.output_variant.resolve(), output_module)
    if args.attention_variant:
        shutil.copy2(args.attention_variant.resolve(), operator / "deepseek_v4_flash_hca_tnd/attention.py")
    if args.q_projection_variant:
        # 控制组会从同一Python进程导入公共HCA依赖，因此不能覆盖那个公共模块。
        # 候选仅改TND自身的引用，控制入口仍使用原来的公共QKV/Q_B实现。
        tnd = operator / "deepseek_v4_flash_hca_tnd"
        shutil.copy2(args.q_projection_variant.resolve(), tnd / "q_projection_streamed.py")
        shutil.copy2(operator / "deepseek_v4_flash_hca/qkv_proj_rope.py", tnd / "qkv_proj_rope.py")
        entry = tnd / "entry.py"
        entry_text = entry.read_text()
        old_import = "from ..deepseek_v4_flash_hca.qkv_proj_rope import qkv_proj_rope"
        if entry_text.count(old_import) != 1:
            raise ValueError("TND的QKV引用已改变，须重新核对候选与控制隔离")
        entry.write_text(entry_text.replace(old_import, "from .qkv_proj_rope import qkv_proj_rope"))
        if "from .q_projection import qkv_proj_rope_tall" in entry.read_text():
            entry.write_text(entry.read_text().replace(
                "from .q_projection import qkv_proj_rope_tall",
                "from .qkv_proj_rope import qkv_proj_rope as qkv_proj_rope_tall",
            ))
    shutil.copy2(HERE / "run_case.py", target / "run_case.py")
    shutil.copy2(HERE / "compressor_probe.py", target / "compressor_probe.py")
    shutil.copy2(HERE / "profile_eager.py", target / "profile_eager.py")
    shutil.copy2(HERE / "metadata_sequence.py", target / "metadata_sequence.py")
    shutil.copy2(HERE / "precision_probe.py", target / "precision_probe.py")
    shutil.copy2(HERE / "service_probe.py", target / "service_probe.py")
    shutil.copy2(HERE / "host_probe.py", target / "host_probe.py")
    shutil.copy2(HERE / "attention_probe.py", target / "attention_probe.py")
    if args.precision_probe:
        from precision_probe import generate_entry

        generate_entry(operator / "deepseek_v4_flash_hca_tnd")
    if args.attention_probe or args.attention_probe_long:
        from attention_probe import generate_entry

        generate_entry(operator / "deepseek_v4_flash_hca_tnd", long_attention=args.attention_probe_long)
    if args.legacy_signature_compat:
        path = operator / "deepseek_v4_flash_hca/o_proj_hc_post.py"
        text = path.read_text()
        old = "o_r_i8_pad, wo_b, partials, g, col_g, proj_b_t_rows, q_tid, ROW_TILE,"
        if text.count(old) != 1:
            raise ValueError("旧 HCA 签名已经改变，必须重新检查参考")
        path.write_text(text.replace(old, "o_r_i8_pad, wo_b, partials, g, col_g, proj_b_t_rows, q_tid, t_dim, ROW_TILE,"))
        # 公共新版前端把 valid_length 推断为 INDEX，后端要求经整数类型转 FP32。
        # 该长度始终在 [0,128]，INT32 桥接不改变数值或原调度。
        path = operator / "deepseek_v4_flash_hca/decode_sparse_attn_hca.py"
        text = path.read_text()
        if text.count("pl.cast(valid_length, pl.FP32)") != 3:
            raise ValueError("旧 HCA valid_length 转换点已变化")
        path.write_text(text.replace("pl.cast(valid_length, pl.FP32)",
                                     "pl.cast(pl.cast(valid_length, pl.INT32), pl.FP32)"))
    manifest = {
        "source_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "includes_uncommitted_hca_tnd": True,
        "legacy_signature_compat": args.legacy_signature_compat,
        "precision_probe": args.precision_probe,
        "output_variant": str(args.output_variant.resolve()) if args.output_variant else None,
        "q_projection_variant": str(args.q_projection_variant.resolve()) if args.q_projection_variant else None,
        "attention_variant": str(args.attention_variant.resolve()) if args.attention_variant else None,
        "attention_probe": args.attention_probe,
        "attention_probe_long": args.attention_probe_long,
        "shared_toolchain": json.loads((HERE.parent / "csa_shared_toolchain/manifest.json").read_text()),
        "files": [{"path": str(p.relative_to(target)), "bytes": p.stat().st_size}
                  for p in sorted(target.rglob("*.py"))],
        "scope": "HCA TND attention 半层；无 CSA TND 实验源码，无权重、SDK 或编译产物",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
