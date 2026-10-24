# SPDX-License-Identifier: Apache-2.0
"""CPU回归：验证CLI到每个DP子进程的实际传参，不启动Worker或NPU。"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class LaunchConfigurationTest(unittest.TestCase):
    def test_hccl_buffer_reaches_every_child(self):
        source = Path(__file__).resolve().parent.parent / "offline_pd/run.py"
        spec = importlib.util.spec_from_file_location("offline_launch_test", source)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for command, explicit, expected, ranks in (
            ("decode", None, 1800, 16),
            ("decode", 768, 768, 16),
            ("prefill", None, 1024, 4),
        ):
            with self.subTest(command=command, explicit=explicit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bank = root / "bank"
                bank.mkdir()
                (bank / "audit.json").write_text(json.dumps({"status": "PASS"}))
                argv = [str(source), command, "--bank", str(bank), "--output", str(root / "out")]
                model = root / "formal-checkpoint"
                argv += ["--model", str(model), "--host", "127.0.0.1", "--nic", "lo"]
                if explicit:
                    argv += ["--hccl-buffer-mb", str(explicit)]
                children = []

                def start(cmd, _children=children, **kwargs):
                    _children.append((cmd, kwargs["env"]))
                    return Mock(pid=12345, returncode=0, poll=Mock(return_value=0), wait=Mock(return_value=0))

                with (
                    patch.object(runner.sys, "argv", argv),
                    patch.dict(runner.os.environ, {"TASK_DEVICE": ",".join(map(str, range(16)))}),
                    patch.object(runner.subprocess, "Popen", side_effect=start),
                    patch.object(runner.os, "killpg"),
                    patch.object(runner.signal, "signal"),
                ):
                    runner.main()
                self.assertEqual(len(children), ranks)
                for cmd, env in children:
                    self.assertEqual(env["HCCL_BUFFSIZE"], str(expected))
                    self.assertEqual(cmd[cmd.index("--hccl-buffer-mb") + 1], str(expected))
                    self.assertEqual(cmd[cmd.index("--model") + 1], str(model))
                    self.assertEqual(env["HCCL_IF_IP"], "127.0.0.1")
                    self.assertEqual(env["HCCL_SOCKET_IFNAME"], "lo")


if __name__ == "__main__":
    unittest.main()
