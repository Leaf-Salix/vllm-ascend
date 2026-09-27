"""单卡同输入/同历史的请求副本诊断；仅复用现有整层执行器，不改生产路径。"""

import sys
from pathlib import Path

SOURCE = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(SOURCE / "tests/pypto_test"))
import dsv4_csa_single_layer as runner  # noqa: E402


def install():
    make_fixture = runner.make_fixture
    collect_state = runner.collect_state
    run = runner.run
    snapshots = []

    def duplicate_fixture(config, attention, batch, history, seed, device):
        fixture = make_fixture(config, attention, batch, history, seed, device)
        hidden = fixture["hidden"]
        source_hidden = hidden[:6].clone()
        for request in range(batch):
            hidden[6 * request:6 * (request + 1)].copy_(source_hidden)
        for group in fixture["groups"].values():
            per_request, remainder = divmod(group["pages"] - 1, batch)
            if remainder or per_request < 1:
                raise ValueError("fixture 页分配不满足独占请求等长分段合同")
            for view in group["views"]:
                first = view[1:1 + per_request].clone()
                for request in range(1, batch):
                    begin = 1 + request * per_request
                    view[begin:begin + per_request].copy_(first)
            group["initial"] = group["allocation"].cpu()
        return fixture

    def summarize(values, batch):
        values = values.reshape(batch, 6, -1)
        different = values[1:] != values[:1]
        return {
            "compared_elements": different.numel(),
            "different_elements": int(different.sum()),
            "compared_token_rows": (batch - 1) * 6,
            "different_token_rows": int(different.any(dim=-1).sum()),
            "max_abs": float((values[1:].float() - values[:1].float()).abs().max()),
        }

    def collect(fixture, output, topk):
        state = collect_state(fixture, output, topk)
        batch = fixture["tokens"] // 6
        snapshots.append({
            "x_out": summarize(state["x_out"], batch),
            "topk_ordered": summarize(state["idx_topk"], batch),
            "topk_set": summarize(state["idx_topk"].sort(dim=-1).values, batch),
        })
        return state

    def diagnose(args, report):
        run(args, report)
        labels = ["native_eager_0", "native_eager_1", "pto_eager_0", "pto_eager_1"]
        if args.timing_iters:
            labels.extend(["native_graph_last", "pto_graph_last"])
        if len(snapshots) != len(labels):
            raise ValueError(f"unexpected collect sequence: {len(snapshots)} vs {labels}")
        report["duplicate_requests"] = {
            "scope": "单卡正式layer4权重；所有请求6个输入及逻辑历史相同、物理页独立；不代表真实模型路由或性能",
            "operator_source": str(SOURCE),
            "snapshots": dict(zip(labels, snapshots)),
        }

    runner.make_fixture = duplicate_fixture
    runner.collect_state = collect
    runner.run = diagnose


if __name__ == "__main__":
    install()
    runner.main()
