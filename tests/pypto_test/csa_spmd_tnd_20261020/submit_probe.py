# SPDX-License-Identifier: Apache-2.0
"""Freeze the shell launcher before task-submit; record the returned task ID."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

METADATA_HISTORY_SLACK = 128


def validate_fixture_arguments(arguments):
    """Reject known harness-contract mistakes before reserving an NPU."""
    if "--metadata-matrix" in arguments and any(
        flag in arguments
        for flag in ("--service-route", "--framework-graph", "--service-metadata-replay", "--service-timing")
    ):
        raise ValueError("metadata矩阵与服务诊断需要独立fixture任务；提交前拒绝混用")
    if not any(flag in arguments for flag in ("--metadata-matrix", "--service-metadata-replay")):
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--history", type=int)
    parser.add_argument("--table-history", type=int)
    parser.add_argument("--query-tokens", type=int, default=5)
    parser.add_argument("--query-lengths")
    parser.add_argument("--token-capacity", type=int)
    parser.add_argument("--request-capacity", type=int)
    parser.add_argument("--metadata-matrix", action="store_true")
    parser.add_argument("--b1-metadata-matrix", action="store_true")
    parser.add_argument("--service-metadata-replay", action="store_true")
    args, _ = parser.parse_known_args(arguments)
    queries = (
        [int(value) for value in args.query_lengths.split(",")]
        if args.query_lengths
        else [args.query_tokens] * args.batch
    )
    token_capacity = args.token_capacity if args.token_capacity is not None else sum(queries)
    request_capacity = args.request_capacity if args.request_capacity is not None else args.batch
    if args.metadata_matrix:
        if args.b1_metadata_matrix:
            valid = (args.batch, queries, token_capacity, request_capacity) == (1, [6], 6, 1)
            expected = "B1、S6、T容量6、请求容量1"
        else:
            # The uniform S1 case leaves capacity-6 padding tokens. Each
            # dummy request can absorb at most six, in addition to six active.
            required_requests = 6 + (token_capacity - 6 + 5) // 6
            valid = (
                (args.batch, queries) == (6, [6] * 6)
                and 41 <= token_capacity <= 72
                and request_capacity == required_requests
            )
            expected = f"B6、各S6、41<=T容量<=72、请求容量{required_requests}（覆盖S1的dummy尾部）"
        if not valid:
            raise ValueError(f"metadata矩阵的初始捕获输入须为{expected}；各变长场景由矩阵内部生成")
        if (
            args.history is None
            or args.table_history is None
            or args.table_history < args.history + METADATA_HISTORY_SLACK
        ):
            raise ValueError("metadata矩阵须提供history及至少history+128的table-history；提交前拒绝不完整fixture")
    if args.service_metadata_replay and queries == list(reversed(queries)):
        raise ValueError("该服务metadata用例要求query长度反转后发生变化；B1请使用标准framework-graph检查")


def main():
    here = Path(__file__).resolve().parent
    if len(sys.argv) < 2:
        raise SystemExit("usage: submit_probe.py OUTPUT [probe arguments...]")
    validate_fixture_arguments(sys.argv[2:])
    output = Path(sys.argv[1]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    launcher = output.with_name(output.name + "_launcher.sh")
    source = (here / "run_probe.sh").read_bytes()
    with launcher.open("xb") as stream:
        stream.write(source)
    command = shlex.join(["bash", str(launcher), str(output), *sys.argv[2:]])
    completed = subprocess.run(
        ["task-submit", "--device", "auto", "--max-time", "3600", command],
        capture_output=True,
        text=True,
        check=True,
    )
    task_id = completed.stdout.strip()
    if not task_id.startswith("task_") or len(task_id.splitlines()) != 1:
        raise RuntimeError(f"Unexpected task-submit response: {task_id}")
    path = here / "jobs.json"
    jobs = json.loads(path.read_text())
    jobs.append(
        {
            "task_id": task_id,
            "output": str(output),
            "last_observed_status": "submitted",
            "arguments": sys.argv[2:],
            "launcher": str(launcher),
            "launcher_sha256": hashlib.sha256(source).hexdigest(),
            "command": command,
        }
    )
    path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n")
    print(task_id)


if __name__ == "__main__":
    main()
