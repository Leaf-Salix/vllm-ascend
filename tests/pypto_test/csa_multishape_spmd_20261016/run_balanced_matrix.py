# SPDX-License-Identifier: Apache-2.0
"""队列内按卡顺序执行两个规格，控制同时编译数量；不同卡可各跑一组。"""

import argparse
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cases", required=True, help="history:batch,history:batch")
parser.add_argument("--device", type=int, help="接收 task-submit 自动追加的设备号")
args = parser.parse_args()
if "TASK_DEVICE" not in os.environ:
    raise RuntimeError("必须通过 task-submit 运行")
if args.device is not None and args.device != int(os.environ["TASK_DEVICE"]):
    raise RuntimeError("命令行与队列分配的设备不一致")
root = Path(__file__).resolve().parent
failed = False
for value in args.cases.split(","):
    history, batch = map(int, value.split(":"))
    name = f"balanced_h{history}_b{batch}_v1"
    command = ["bash", str(root / "run_device_compare.sh"), str(batch), str(history),
               str(root / name), "--balanced-schedule"]
    print(name, "开始", flush=True)
    completed = subprocess.run(command, check=False)
    failed |= completed.returncode != 0
    print(json.dumps({"case": name, "exit_code": completed.returncode}, ensure_ascii=False), flush=True)
raise SystemExit(1 if failed else 0)
