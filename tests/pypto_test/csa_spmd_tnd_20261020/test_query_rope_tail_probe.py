# SPDX-License-Identifier: Apache-2.0
"""Accept the device option appended by task-submit without initializing an NPU."""

import pytest
from query_rope_tail_probe import parse_args


@pytest.mark.parametrize("cpu_only", [False, True])
def test_task_submit_device_argument(tmp_path, cpu_only):
    output = tmp_path / "unused"
    arguments = ["--output", str(output), "--device", "3"]
    if cpu_only:
        arguments.append("--cpu-only")
    args = parse_args(arguments)
    assert args.device == 3 and args.cpu_only is cpu_only
    assert args.output == output and not output.exists()
