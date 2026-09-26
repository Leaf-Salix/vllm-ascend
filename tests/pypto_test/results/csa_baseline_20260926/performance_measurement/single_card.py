# SPDX-License-Identifier: Apache-2.0
"""只验证性能观测器在真实 ACL Graph 重放中能获得新的设备时间戳。"""

import json
from pathlib import Path
from types import SimpleNamespace

import dsv4_csa_env

dsv4_csa_env.activate()

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
from offline_pd.observer import OfflineCSAObserver  # noqa: E402

torch.npu.set_device(0)
output = Path(__file__).resolve().parent
left = torch.ones((96, 512), dtype=torch.bfloat16, device="npu")
right = torch.ones((512, 512), dtype=torch.bfloat16, device="npu")
torch.mm(left, right)
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    result = torch.mm(left, right)
graph.replay()
torch.npu.synchronize()
observer = OfflineCSAObserver()
observer.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=0))
observer.model_runner = SimpleNamespace(execute_model=lambda _: graph.replay())
scheduled = SimpleNamespace(total_num_scheduled_tokens=96, num_scheduled_tokens=dict.fromkeys(range(16), 6))
observer.offline_begin_steady(4, 96, 16)
for _ in range(24):
    observer.model_runner.execute_model(scheduled)
steady = observer.offline_end_steady()
observer.offline_begin_profile(str(output / "trace"), 1, 3, 96, 16, 0)
for _ in range(4):
    observer.model_runner.execute_model(scheduled)
profile = observer.offline_end_profile()
report = {"scope": "单卡矩阵乘图仅检查测量工具；不代表 CSA 或整模型性能", "steady": steady, "profile": profile}
(output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
assert steady["sufficient"] and profile["sufficient"]
assert bool((result == 512).all())
print("单卡观测器通过：20 个新设备时间戳、3 个完整 Level0 重放窗口")
