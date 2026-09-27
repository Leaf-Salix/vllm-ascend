"""单卡小图验证同轮forward事件的采集、窗口选择和钩子恢复。"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from offline_pd.observer import OfflineCSAObserver  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(args.device)
    torch.manual_seed(1024)
    x = torch.randn((256, 256), device="npu", dtype=torch.bfloat16)
    weight = torch.randn_like(x) * 0.03

    def compute():
        value = x
        for _ in range(8):
            value = torch.mm(value, weight)
        return value

    for _ in range(3):
        compute()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = compute()

    class Runner:
        def __init__(self):
            self._dsa_positions_cpu_buf = torch.arange(48)

        def _model_forward(self):
            graph.replay()
            return output

        def execute_model(self, scheduler_output):
            return self._model_forward()

    runner = Runner()
    original_execute, original_forward = runner.execute_model, runner._model_forward
    observer = OfflineCSAObserver()
    observer.model_runner = runner
    observer.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=0))
    schedule = SimpleNamespace(total_num_scheduled_tokens=48, num_scheduled_tokens={i: 6 for i in range(8)})
    observer.offline_begin_profile(str(args.output / "trace"), 2, 3, 48, 8, 0, True)
    for _ in range(6):
        runner.execute_model(schedule)
    profile = observer.offline_end_profile()
    restored = runner.execute_model == original_execute and runner._model_forward == original_forward
    observation = profile["profile_forward"]
    assert restored and profile["sufficient"] and observation["sufficient"], profile
    assert observation["step_indices"] == [2, 3, 4], observation
    assert observation["positions_cpu"] == [list(range(48))] * 3, observation
    # The original no-profiler observer must still work after the optional hook.
    observer.offline_begin_forward(1, 48, 8, 2)
    for _ in range(3):
        runner.execute_model(schedule)
    forward = observer.offline_end_forward()
    assert forward["sufficient"], forward
    result = {"status": "PASS", "restored": restored, "profile": profile, "subsequent_forward": forward,
              "scope": "单卡BF16小图，仅验证诊断采集；不提供CSA/EP16性能或数值结论"}
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("PASS", observation["samples_us"], "us", flush=True)


if __name__ == "__main__":
    main()
