# SPDX-License-Identifier: Apache-2.0
"""从已验证v04接入包派生首token诊断；不改变算子或共享框架。"""
import argparse
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--operator", type=Path, help="可选：已完成单卡诊断的私有算子目录")
    args = parser.parse_args()
    assert Path(args.name).name == args.name
    origin = HERE / "experiments/v04_graph_service"
    target = HERE / "experiments" / args.name
    source = target / "source"
    shutil.copytree(origin / "source", source, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if args.operator is not None:
        for module in ("deepseek_v4_flash_csa", "deepseek_v4_flash_hca", "deepseek_v4_flash_hca_tnd"):
            shutil.copytree(args.operator / module, source / "vllm_ascend/ops/pypto" / module,
                            dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(HERE / "precision_observer.py", source / HERE.relative_to(HERE.parents[2]) / "precision_observer.py")
    runner = source / "tests/pypto_test/offline_pd/run.py"
    text = runner.read_text()
    old = '"worker_extension_cls": "hca_tnd_model_validation_20261024.observer.HCAObserver"'
    assert text.count(old) == 1
    text = text.replace(old, '"worker_extension_cls": "hca_tnd_model_validation_20261024.precision_observer.PrecisionObserver"')
    old = '    if not prefill and args.backend == "pto":\n'
    assert text.count(old) == 1
    text = text.replace(old, '    if not prefill:  # 诊断两侧使用相同模型壳，HCA调用在observer中显式选择。\n')
    old = '+ args.decode_tokens + 32,'
    assert text.count(old) == 1
    text = text.replace(old, '+ 192 + 32,  # 只生成1token，但保持原B16模型及KV容量。')
    runner.write_text(text)
    manifest = json.loads((origin / "manifest.json").read_text())
    manifest.update(source=str(source), parent_snapshot=str(origin), precision_diagnostic=True,
                    scope="相同模型壳，首轮S1 logits/逐层数据；Native控制必须与历史Native首token核对",
                    decode_tokens=1, capacity_decode_tokens=192)
    if args.operator is not None:
        manifest["precision_operator"] = str(args.operator.resolve())
    (target / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(source)


if __name__ == "__main__":
    main()
