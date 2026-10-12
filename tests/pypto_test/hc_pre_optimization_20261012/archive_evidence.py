"""归档本轮 HC_pre 小型文本证据；大快照及二进制只记录本地位置。"""
import json
import shutil
from pathlib import Path


DESTINATION = Path(__file__).resolve().parent
TEST_ROOT = DESTINATION.parent
ROOT_NAMES = (
    "hc_pre_goal20_20261011", "hc_pre_stage1_merge_20261011",
    "hc_pre_tail_perf_20261011", "hc_pre_tail_vectors_20261011",
)
TEXT_SUFFIXES = {".py", ".sh", ".json", ".md", ".csv", ".log", ".txt"}
LIMIT = 1024 * 1024
copied = []
retained = []


def archive(source, destination):
    size = source.stat().st_size
    if size > LIMIT:
        retained.append({"path": str(source), "bytes": size, "reason": "大型文本保留本地"})
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied.append({"source": str(source), "archive": str(destination.relative_to(DESTINATION)), "bytes": size})


for name in ROOT_NAMES:
    source_root = TEST_ROOT / "results" / name
    target_root = DESTINATION / "evidence" / name
    retained.append({"path": str(source_root), "reason": "完整原始工作目录；权重、输出快照、编译目录及原始采集留在本地"})
    for source in sorted(source_root.iterdir()):
        if source.is_file() and source.suffix in TEXT_SUFFIXES:
            archive(source, target_root / source.name)
        elif source.is_dir() and not source.name.startswith(("jit_", "cache_", "build_")):
            # 仅收取已结束运行的顶层报告与独立诊断表，不深入编译或采集缓存。
            if source.name == "full14_abba_20261011":
                retained.append({"path": str(source), "reason": "用户取消的旧全档 ABBA；不发布未完成性能结果"})
                continue
            for item in sorted(source.iterdir()):
                if item.is_file() and item.suffix in TEXT_SUFFIXES:
                    archive(item, target_root / source.name / item.name)
                elif item.is_dir() and item.name not in {"ascend", "profile", "dfx", "traces", "build"}:
                    report = item / "report.json"
                    if report.is_file():
                        archive(report, target_root / source.name / item.name / report.name)
        elif source.is_dir() and source.name.startswith("build_"):
            report = source / "report.json"
            if report.is_file():
                archive(report, target_root / source.name / report.name)

goal_root = TEST_ROOT / "results" / ROOT_NAMES[0]
summary = json.loads((goal_root / "latest14_20261012_summary.json").read_text())
for row in summary["rows"]:
    for field, suffix in (("pto_trace", "PTO_L1"), ("native_trace", "Native_PyTorch")):
        source = Path(row[field])
        name = f'HCpre_B{row["batch_size"]:02d}_S{row["seqlen"]}_T{row["tokens"]:03d}_{suffix}_3steps.json'
        archive(source, DESTINATION / "latest14_traces" / name)

for label in ("summary.json", "summary.csv", "manifest.json"):
    archive(goal_root / f"latest14_20261012_{label}", DESTINATION / f"latest14_{label}")
(DESTINATION / "file_manifest.json").write_text(json.dumps({
    "scope": "仅归档本轮 HC_pre；不包含 CSA 会话文件，不计算 hash。",
    "files": copied, "local_retained": retained,
    "count": len(copied), "bytes": sum(item["bytes"] for item in copied),
}, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"files": len(copied), "bytes": sum(item["bytes"] for item in copied)}, ensure_ascii=False))
