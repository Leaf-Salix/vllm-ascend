# SPDX-License-Identifier: Apache-2.0
"""真实CPU进程树验证：rank及其worker继承同一组，首差清理不遗漏后代。"""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def test_owned_group_cleanup_includes_rank_workers(tmp_path):
    spec = importlib.util.spec_from_file_location("threeway_launch_test", Path(__file__).with_name("launch.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    script = tmp_path / "tree.py"
    receipt = tmp_path / "pids.json"
    script.write_text("""import json, os, subprocess, sys, time
from pathlib import Path
if len(sys.argv) == 3 and sys.argv[1] == "rank":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(sys.argv[2]).write_text(json.dumps([os.getpid(), child.pid, os.getpgrp()]))
    time.sleep(60)
elif len(sys.argv) == 3 and sys.argv[1] == "parent":
    roots = [Path(sys.argv[2] + str(i)) for i in range(2)]
    children = [subprocess.Popen([sys.executable, __file__, "rank", str(p)], start_new_session=False) for p in roots]
    while not all(p.exists() for p in roots): time.sleep(.01)
    Path(sys.argv[2]).write_text(json.dumps([json.loads(p.read_text()) for p in roots]))
    time.sleep(60)
else:
    raise SystemExit("explicit parent/rank mode required")
""")
    process = subprocess.Popen([sys.executable, str(script), "parent", str(receipt)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        rows = json.loads(receipt.read_text())
        assert all(group == process.pid for _, _, group in rows)
        started = time.monotonic()
        module.stop_group(process, grace=0.2)
        assert time.monotonic() - started < 2
        for parent, child, _ in rows:
            for pid in (parent, child):
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
                ).stdout.strip()
                assert not state or state.startswith("Z"), (pid, state)
    finally:
        module.stop_group(process, grace=0.2)
    assert process.poll() is not None and process.pid != os.getpgrp()
