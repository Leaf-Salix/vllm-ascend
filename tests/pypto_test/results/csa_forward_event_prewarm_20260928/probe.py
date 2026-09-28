"""单卡验证预创建事件重记能计量图重放；不用于比较性能。"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from offline_pd.event_mode import get_event_work_mode, set_event_work_mode
from offline_pd.forward_timing import TIMING_EVENT_SETUP, prewarm_timing_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--mode", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    set_event_work_mode(args.mode)
    value = torch.full((128, 128), 2.0, device=f"npu:{args.device}")
    for _ in range(3):
        value + 1
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = value + 1
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    events = prewarm_timing_events(10)
    setup_last_stamp = events[-1][1].recorded_time()
    value.fill_(5)
    for begin, end in events:
        begin.record()
        graph.replay()
        end.record()
    events[-1][1].synchronize()
    samples_us = [begin.elapsed_time(end) * 1000 for begin, end in events]
    stamps = [begin.recorded_time() for begin, _ in events]
    assert all(sample > 0 for sample in samples_us), samples_us
    assert setup_last_stamp < stamps[0], (setup_last_stamp, stamps)
    assert all(a < b for a, b in zip(stamps, stamps[1:])), stamps
    torch.testing.assert_close(output.cpu(), torch.full((128, 128), 6.0), atol=0, rtol=0)
    result = {"status": "PASS", "scope": "事件重记与图重放功能验证，不是性能比较",
              "device": args.device, "event_work_mode": get_event_work_mode(),
              "timing_event_setup": TIMING_EVENT_SETUP, "setup_last_stamp": setup_last_stamp,
              "start_stamps": stamps, "samples_us": samples_us, "torch_npu": torch_npu.__version__}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
