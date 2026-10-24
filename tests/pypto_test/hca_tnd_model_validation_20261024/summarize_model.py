# SPDX-License-Identifier: Apache-2.0
"""汇总已通过的B16/B24模型路由证据，严格区分业务重放、空闲图和答案质量。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    rows = []
    for batch in (16, 24):
        root = HERE / f"runs/model_b{batch}_v04"
        config = read(root / "configuration.json")
        check = read(root / "functional_check.json")
        assert check["status"] == "PASS" and config["HCA"] == "mpmd_tnd" and config["CSA"] == "native"
        per_rank, outputs = [], []
        for rank in range(16):
            data = read(root / f"rank{rank}.mixed.json")
            observation = data["observation"][0]
            routes = observation["hca_execution_routes"]
            real = [r for r in routes if r["requested_mode"] == "FULL" and r["actual_tokens"] > 0
                    and r["max_seq_len"] >= 131072]
            assert real and all(r["all_hca_graph_replayed"] and not r["python_resubmitted"] for r in real)
            assert len(observation["hca_layers"]) == 20
            def count(predicate):
                return sum(r["steps"] for r in real if predicate(r))
            per_rank.append(dict(rank=rank, real_full_steps=count(lambda r: True),
                full_batch_s6_steps=count(lambda r: r["actual_requests"] == batch and r["actual_tokens"] == batch*6),
                tail_steps=count(lambda r: r["actual_requests"] < batch),
                padded_tail_steps=count(lambda r: r["actual_requests"] < batch and r["padded_tokens"] > r["actual_tokens"]),
                small_graph_steps=count(lambda r: r["actual_requests"] == 1 and r["padded_tokens"] == 6),
                eager_steps=sum(r["steps"] for r in routes if r["requested_mode"] == "NONE" and r["max_seq_len"] >= 131072),
                idle_short_full_steps=sum(r["steps"] for r in routes if r["requested_mode"] == "FULL" and r["max_seq_len"] < 8192),
                memory=observation["memory"], layers=observation["hca_layers"]))
            outputs.append([(r["key"], r["output_token_ids"]) for r in data["requests"]])
        assert all(values == outputs[0] for values in outputs), "副本不同不得合并问答展示"
        comparison = read(HERE / f"B{batch}_COMPARISON.json")
        rows.append(dict(batch=batch, requests=check["requests"], output_tokens=check["output_tokens"],
                         source=str(root), source_commit=config["source_commit"], toolchain=config["toolchain"],
                         functional="PASS", cross_dp_tokens_equal=True, ranks=per_rank,
                         native_token_equal_requests=comparison["token_equal_requests"],
                         mean_accepted=check["acceptance"]["mean_accepted"], preemptions=0))
    audit_path = HERE / "PRODUCTION_BINDING_AUDIT.json"
    promoted = audit_path.exists() and read(audit_path)["status"] == "PASS"
    report = dict(status="FUNCTIONAL_PASS_ANSWER_QUALITY_SEPARATE", variants=rows,
                  scope="只替换20层C128 HCA；21层CSA、SWA、FFN仍Native。静态编译/SuperKernel关闭的功能任务，不测性能。",
                  math="正式FP32融合；未采用BF16数值对齐候选", production_promoted=promoted,
                  production_source_audit="PRODUCTION_BINDING_AUDIT.json" if promoted else None,
                  acceptance="Native token一致仅为诊断，答案质量按题意独立验收；192token可能未完成答案")
    (HERE / "MODEL_INTEGRATION_SUMMARY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for row in rows:
        rank = row['ranks'][0]
        print(row['batch'], {k:rank[k] for k in ('real_full_steps','full_batch_s6_steps','tail_steps','padded_tail_steps','small_graph_steps','eager_steps')})


if __name__ == "__main__":
    main()
