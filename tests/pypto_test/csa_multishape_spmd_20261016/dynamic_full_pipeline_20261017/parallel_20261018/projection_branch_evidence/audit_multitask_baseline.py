#!/usr/bin/env python3
"""只读核对历史905.08与本轮880.06的来源；不启动NPU、不计算文件散列。

比较原始样本、权重加载记录、CSA源文件和保存的编译产物。IR只归一化
源码目录及自动inline编号，不能用归一化后的相等替代完整语义或二进制证明。
性能因果仍需受控实验；本脚本不从跨轮差异推导编译器或硬件收益。
"""

import json
import re
from pathlib import Path


def normalize_generated(text):
    text = re.sub(
        r'/data/pyptouser/qinchuanyu/pto-eager/(?:[^"\n]*?)/deepseek_v4_flash_csa/',
        "CSA_SOURCE/",
        text,
    )
    return re.sub(r"inline\d+", "inlineID", text)


def main():
    here = Path(__file__).resolve().parent
    history_root = here.parents[2]
    repo = history_root.parents[2]
    old_path = history_root / "b16_cached_row_deps_v1/result.json"
    new_path = here / "h131072_b16_branch4_v1/result.json"
    old, new = (json.loads(path.read_text()) for path in (old_path, new_path))
    old_stats = old["l1_incore"]["dynamic_pto"]
    new_stats = new["l1"]["statistics_us"]["dynamic_pto"]
    frozen = repo / "tests/pypto_test/csa_static_spmd_feasibility_20261015/reference_operator_source"
    current = repo / "vllm_ascend/ops/pypto"
    source_equal, source_different = [], []
    for path in sorted((frozen / "deepseek_v4_flash_csa").glob("*.py")):
        relative = path.relative_to(frozen)
        target = current / relative
        result = source_equal if target.read_bytes() == path.read_bytes() else source_different
        result.append(str(relative))
    old_build = next((old_path.parent / "build_output").glob("_decode_csa_tp1_layer_*"))
    new_build = next((new_path.parent / "build_output").glob("_decode_csa_tp1_layer_*"))
    generated = {}
    for subdir, suffix in (("ptoas", ".pto"), ("kernels", ".cpp"), ("orchestration", ".cpp")):
        old_files = {str(p.relative_to(old_build / subdir)): p
                     for p in (old_build / subdir).rglob("*" + suffix)}
        new_files = {str(p.relative_to(new_build / subdir)): p
                     for p in (new_build / subdir).rglob("*" + suffix)}
        equal, different = [], []
        for name in sorted(old_files.keys() & new_files.keys()):
            a = normalize_generated(old_files[name].read_text())
            b = normalize_generated(new_files[name].read_text())
            (equal if a == b else different).append(name)
        generated[subdir] = {
            "suffix": suffix,
            "normalized_equal": equal,
            "normalized_different": different,
            "only_old": sorted(old_files.keys() - new_files.keys()),
            "only_new": sorted(new_files.keys() - old_files.keys()),
        }
    window_keys = ("incore_span_us", "clock_freq_hz", "core_records", "task_count", "cores")
    report = {
        "old_result": str(old_path), "new_result": str(new_path),
        "old_task": "task_20261017_044912_40225339550",
        "new_task": "task_20261018_114949_268532326529",
        "old_stats": old_stats, "new_stats": new_stats,
        "median_difference_us": old_stats["median_us"] - new_stats["median_us"],
        "median_change_percent": (new_stats["median_us"] / old_stats["median_us"] - 1) * 100,
        "old_environment": old["environment"], "new_environment": new["environment"],
        "weight_load_records_equal": old["weights"] == new["weights"],
        "source_equal": source_equal, "source_different": source_different,
        "generated_normalization": "仅源码路径与inline数字编号；保留操作、类型、地址、常量和顺序",
        "generated_comparison": generated,
        "old_windows": [{k: w[k] for k in window_keys} for w in old["l1_windows"]
                        if w["side"] == "dynamic_pto"],
        "new_windows": [{k: w[k] for k in window_keys} for w in new["l1"]["windows"]
                        if w["side"] == "dynamic_pto"],
        "old_measurement": "两角色AB/BA；5次预热与20次Event采样后，3次L1采样",
        "new_measurement": "三角色轮换；3次图预热后，5次L1采样",
        "conclusion": "源码及规范化编排相同；编译环境、部分生成核代码和采样流程不同，无法归因25.02微秒变化",
        "causal_experiment_performed": False,
        "new_npu_measurement_performed": False,
        "old_baseline_preserved": True,
        "performance_acceptance": "每轮单SPMD只与该轮原分task比较，不跨轮替换分母",
    }
    destination = here / "MULTITASK_905_VS_880_AUDIT.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "source_equal_count": len(source_equal),
                      "source_different": source_different,
                      "weights_equal": report["weight_load_records_equal"],
                      "generated_counts": {k: {"equal": len(v["normalized_equal"]),
                                                "different": len(v["normalized_different"])}
                                           for k, v in generated.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
