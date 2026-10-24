# SPDX-License-Identifier: Apache-2.0
"""16卡功能验收：真实HCA路由、图重放、满批S6、完整token及零抢占。"""
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def main(root):
    config = read(root / "configuration.json")
    batch, side = config["batch"], config["backend"]
    acceptance = read(root / "acceptance.json")
    assert acceptance["qualified"] and acceptance["total_preemptions"] == 0
    rows = []
    for rank in range(16):
        data = read(root / f"rank{rank}.mixed.json")
        assert data["rank"] == rank and data["batch"] == batch and data["decode_tokens"] == 192
        assert data["backend"] == side and data["pto_attention"] == "hca"
        assert len(data["requests"]) == batch and all(len(r["output_token_ids"]) == 192 for r in data["requests"])
        (implementations,) = data["attention_implementations"]
        csa = [r for r in implementations if r["kind"] == "csa"]
        hca = [r for r in implementations if r["kind"] == "hca"]
        assert len(csa) == 21 and len(hca) == 20
        assert all(r["implementation"] == "native" for r in csa)
        if side == "pto":
            assert all(r["implementation"] == "pto" and r["runtime_class"].endswith("deepseek_v4_flash_hca_tnd.graph_service") for r in hca)
        else:
            assert all(r["implementation"] == "native" for r in hca)
        (runtime,) = data["worker_runtime_config"]
        assert runtime["runtime_environment"]["HCCL_BUFFSIZE"] == str(config["hccl_buffer_mb"])
        assert runtime["deterministic_level"] == 1 and not runtime["dynamic_eplb"]
        assert runtime["cann_event_work_mode"] == 1
        compilation = runtime["decode_optimizations"]["ascend_compilation_config"]
        assert not compilation["enable_static_kernel"] and compilation["enable_npugraph_ex"]
        (observation,) = data["observation"]
        occupancy = observation["decode_occupancy"]
        full_steps = sum(r["steps"] for r in occupancy if r["actual_requests"] == batch and r["actual_tokens"] == batch*6)
        assert full_steps > 0, (rank, "缺少真实满批S6，不能只看padding容量", occupancy)
        if side == "pto":
            assert len(observation["hca_layers"]) == 20
            assert all(r["graph_counts"]["allowed"] > 0 for r in observation["hca_layers"])
            routes = [r for r in observation["hca_execution_routes"] if r["requested_mode"] == "FULL" and r["actual_tokens"] > 0]
            assert routes and any(r["actual_requests"] == batch for r in routes)
            assert all(r["all_hca_graph_replayed"] and not r["python_resubmitted"] for r in routes), (rank, routes)
        rows.append(dict(rank=rank, actual_full_s6_steps=full_steps, observation=observation))
    result = dict(status="PASS", scope="真实HCA接入及图重放功能；输出token与Native的对照另行验收",
                  backend=side, batch=batch, requests=16*batch, output_tokens=16*batch*192,
                  acceptance={k:v for k,v in acceptance.items() if k != "requests"}, ranks=rows)
    (root / "functional_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != "ranks"},ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
