# SPDX-License-Identifier: Apache-2.0
"""冻结本轮公共 PTO helper 与多 task 源码，避免其它会话的更新混入配对测试。

只复制 Python 源文件，记录路径、大小及 Git 提交；不复制编译物、不做 hash。
正式单 SPMD 参考与候选另有独立源码包，运行时仍沿用已有 source_loader。
"""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    source = repo / "vllm_ascend/ops/pypto"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-name", default="frozen_pypto",
                        help="本目录下的新快照名；不覆盖已有实验")
    parser.add_argument("--record-name", default="SHARED_SOURCE_FREEZE.json",
                        help="本目录下的对应记录名")
    parser.add_argument("--from-record", action="store_true",
                        help="从已提交的路径记录和对应Git对象恢复；不使用当前工作树")
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.snapshot_name, args.record_name)):
        raise ValueError("快照和记录必须是本目录下的单级名称")
    target = here / args.snapshot_name
    record = here / args.record_name
    if args.from_record:
        manifest = json.loads(record.read_text())
        if any(name.endswith(".py") for name in manifest["tracked_modifications"]):
            raise RuntimeError("记录包含未提交Python修改，不能只从Git恢复")
        files = {}
        for row in manifest["files"]:
            relative = Path(row["path"])
            git_path = "vllm_ascend/ops/pypto/" + relative.as_posix()
            files[relative] = subprocess.check_output(
                ["git", "-C", str(repo), "show", manifest["repository_head"] + ":" + git_path]
            )
            if len(files[relative]) != row["bytes"]:
                raise RuntimeError("Git源文件大小与记录不符：" + str(relative))
    else:
        if record.exists():
            raise RuntimeError("已有冻结记录；恢复请传 --from-record，禁止覆盖历史")
        files = {p.relative_to(source): p.read_bytes() for p in source.rglob("*.py")}
    if target.exists():
        raise RuntimeError("冻结目录已存在，禁止覆盖已使用的测试源码")
    target.mkdir()
    for relative, content in files.items():
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
    # 仅检查复制期间是否存在并发修改，不计算或保存散列清单。
    if not args.from_record and any(
        (source / relative).read_bytes() != content for relative, content in files.items()
    ):
        raise RuntimeError("复制期间公共源码改变，不能将此快照用于同轮比较")
    if args.from_record:
        print(json.dumps({"restored_files": len(files), "from": manifest["repository_head"]}))
        return
    manifest = {
        "repository_head": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source": str(source),
        "snapshot": str(target),
        "tracked_modifications": subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--name-only", "--", "vllm_ascend/ops/pypto"],
            text=True,
        ).splitlines(),
        "files": [{"path": str(name), "bytes": len(value)} for name, value in sorted(files.items())],
        "includes_binaries": False,
        "hashes_computed": False,
    }
    record.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"files": len(files), "bytes": sum(map(len, files.values()))}))


if __name__ == "__main__":
    main()
