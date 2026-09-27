# SPDX-License-Identifier: Apache-2.0
"""Create a reproducible cache bank, run P4x4, then Native/PTO D1x16.

The plan and audit commands are CPU only. Both execution commands require a
16-device task-submit allocation. Logs/artifacts belong to --bank/--output.
"""

import argparse
import contextlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

FORMAL_MODEL = "/data/model/DeepSeek-V4-Flash-0731-w8a8"
# release 正式配置里第一个 compress_ratio==4 的 target 层，泳道采集默认取它。
FIRST_TARGET_CSA_LAYER = 2


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_plan(bank):
    plan = json.loads((bank / "plan.json").read_text())
    if plan["model"] != FORMAL_MODEL:
        raise ValueError("Only the formal ModelSlim checkpoint is authorized")
    return plan


def make_plan(args):
    from tokenizers import Tokenizer

    bank = args.bank.resolve()
    bank.mkdir(parents=True, exist_ok=True)
    if (bank / "plan.json").exists():
        raise FileExistsError(bank / "plan.json")
    model = Path(FORMAL_MODEL)
    index = json.loads((model / "quant_model_weights.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 75 or any(not (model / shard).is_file() for shard in shards):
        raise ValueError("The complete formal 75-shard ModelSlim checkpoint is required")
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    cases = []
    for history in map(int, args.histories.split(",")):
        if history < 128:
            raise ValueError("Use at least 128 historical tokens for the SWA fixture")
        for variant in range(4):
            text = (f"Document {variant}: This is a deterministic systems engineering workload. "
                    "Explain paged attention, distributed inference and cache consistency. "
                    "缓存需要保留上下文次序、压缩状态和索引。请基于材料继续分析。\n")
            base = tokenizer.encode(text, add_special_tokens=False).ids
            tokens = (base * ((history + 1 + len(base) - 1) // len(base)))[:history + 1]
            key = f"h{history}_v{variant}"
            write_json(bank / f"{key}.tokens.json", tokens)
            cases.append({"key": key, "history": history, "p_dp_rank": variant,
                          "tokens": f"{key}.tokens.json"})
    write_json(bank / "plan.json", {
        "schema": 1, "model": FORMAL_MODEL,
        "weight_shards": len(shards), "weight_bytes": sum((model / s).stat().st_size for s in shards),
        "prefill": {"tp": 4, "dp": 4, "ep": 16},
        "decode": {"tp": 1, "dp": 16, "ep": 16, "speculative_tokens": 5},
        "layout": {"weight_nz_mode": 0, "enable_kv_nz": False, "block_size": 32,
                   "indexer_kv_dtype": "int8", "dtype": "bfloat16"},
        "decode_batches_per_rank": [1, 4, 8, 16, 24, 32, 40],
        "boundary": "P computes H tokens; D loads h(H) and recomputes token H from H+1 prompt tokens",
        "cases": cases,
    })
    print(f"Prepared {len(cases)} P fixtures in {bank}")


PREFIX_DATA_POLICY = (
    "跨副本的有效前缀数据差异记为报告项，不致命。D 侧 TP1 只读 tp0（connector 里 tp 固定为 0），"
    "且 DSA 的 KV 是 MLA 压缩潜变量、跨 TP 复制而非切分，每个副本都是完整一份；"
    "几何／布局／覆盖／history 这些 D 真正依赖的检查仍然致命。"
    "相对误差 |a-b|/max(|a|,|b|) 在近零值上会饱和到 2.0 附近，不能用来判断是不是舍入，"
    "所以量级以 ULP 距离为准，并同时给出中位数与最大值。"
)


def summarize_prefix_differences(per_replica):
    """把逐张量的差异压成可读摘要，并保留量级最大的若干条。"""
    summary = {"per_replica": {}, "policy_is_non_fatal": True}
    for replica, entries in per_replica.items():
        if not entries:
            summary["per_replica"][replica] = []
            continue
        ranked = sorted(entries, key=lambda item: item.get("ulp_median") or 0.0, reverse=True)
        summary["per_replica"][replica] = ranked
        medians = [item["ulp_median"] for item in entries if "ulp_median" in item]
        summary.setdefault("aggregate", {})[replica] = {
            "differing_tensors": len(entries),
            "ulp_median_max": max(medians) if medians else None,
            "max_abs_diff": max((item.get("max_abs_diff") or 0.0) for item in entries),
        }
    return summary


def ulp_gap(left, right):
    """两个同 dtype 浮点张量的 ULP 距离：按位重解释成有序整数后作差。

    近零值跨零号翻转时该距离会饱和到很大，所以判定要同时看中位数，
    不能只看最大值。
    """
    import torch

    if left.dtype == torch.bfloat16:
        wide, floor = torch.int16, -32768
    elif left.dtype == torch.float32:
        wide, floor = torch.int32, -2147483648
    else:
        return None
    def ordered(value):
        raw = value.view(wide).to(torch.int64)
        return torch.where(raw < 0, torch.tensor(floor, dtype=torch.int64) - raw, raw)
    return (ordered(left) - ordered(right)).abs()


def replica_difference(name, reference, other):
    """量化一份张量在两个 TP 副本之间的差异，供 audit 作为报告项记录。

    只在已知两者不逐 bit 相同时调用。相对误差用 |a-b|/max(|a|,|b|) 会在近零值上
    饱和到 2.0 附近，不能用来判断是不是舍入，所以这里以 ULP 距离为准。
    """
    import torch

    gap = ulp_gap(reference, other)
    unequal = reference.view(torch.uint8) != other.view(torch.uint8)
    entry = {"name": name, "dtype": str(reference.dtype).rsplit(".", 1)[-1],
             "differing_bytes": int(unequal.sum().item()), "total_bytes": int(unequal.numel())}
    if gap is None:
        # int8 indexer key：比的是量化码字，差一个码字就说明选中的内容变了。
        differs = reference != other
        entry.update(differing_elements=int(differs.sum().item()),
                     max_codeword_delta=int((reference.to(int) - other.to(int)).abs().max().item()))
        return entry
    hit = gap > 0
    counted = gap[hit].to(torch.float64)
    entry.update(
        differing_elements=int(hit.sum().item()), total_elements=int(gap.numel()),
        ulp_median=float(counted.median().item()), ulp_max=float(counted.max().item()),
        max_abs_diff=float((reference.to(torch.float64) - other.to(torch.float64)).abs().max().item()),
        magnitude_max=float(reference.abs().max().to(torch.float64).item()),
    )
    return entry


def audit(args):
    import torch
    from prefix import cache_contract, prefix_tensor
    from safetensors.torch import load_file

    plan = read_plan(args.bank)
    config = json.loads((Path(FORMAL_MODEL) / "config.json").read_text())
    expected = cache_contract(config)
    results, payloads = [], []
    for case in plan["cases"]:
        replicas = [json.loads((args.bank / case["key"] / f"tp{tp}" / "manifest.json").read_text())
                    for tp in range(4)]
        base = replicas[0]
        errors, raw_differences, prefix_differences = [], {}, {}
        reference_raw, reference_prefix = None, None
        for tp, other in enumerate(replicas):
            payload = args.bank / case["key"] / f"tp{tp}" / "cache.safetensors"
            if not payload.is_file():
                errors.append(f"tp{tp}: missing payload")
                continue
            payloads.append({"path": str(payload.relative_to(args.bank)), "bytes": payload.stat().st_size})
            if other["history"] != case["history"]:
                errors.append(f"tp{tp}: prefix mismatch")
            tensors = load_file(str(payload))
            if set(tensors) != set(expected) or set(other["entries"]) != set(expected):
                errors.append(f"tp{tp}: incomplete target/draft cache coverage")
                continue
            prefixes = {}
            raw_differences[f"tp{tp}"] = []
            prefix_differences[f"tp{tp}"] = []
            for name, ratio in expected.items():
                entry, value = other["entries"][name], tensors[name]
                if (list(value.shape[1:]) != entry["shape"] or str(value.dtype) != entry["dtype"]
                        or entry.get("compress_ratio", ratio) != ratio):
                    errors.append(f"tp{tp}: payload geometry mismatch: {name}")
                fields = ("logical_blocks", "group", "block_size", "shape", "dtype", "stride")
                if any(entry[k] != base["entries"].get(name, {}).get(k) for k in fields):
                    errors.append(f"tp{tp}: replica layout mismatch: {name}")
                prefix = prefix_tensor(value, entry["logical_blocks"], entry["block_size"], case["history"], ratio)
                prefixes[name] = prefix
                if reference_raw is not None:
                    if not torch.equal(value.view(torch.uint8), reference_raw[name].view(torch.uint8)):
                        raw_differences[f"tp{tp}"].append(name)
                    if not torch.equal(prefix.view(torch.uint8), reference_prefix[name].view(torch.uint8)):
                        # 报告项，不致命。理由见下方 prefix_data_differences 的说明。
                        prefix_differences[f"tp{tp}"].append(
                            replica_difference(name, reference_prefix[name], prefix))
            if reference_raw is None:
                reference_raw, reference_prefix = tensors, prefixes
        results.append({"key": case["key"], "tensors": len(base["entries"]), "errors": errors,
                        "raw_payload_differences": raw_differences,
                        "prefix_data_differences": summarize_prefix_differences(prefix_differences)})
    passed = all(not r["errors"] for r in results)
    write_json(args.bank / "audit.json", {"status": "PASS" if passed else "FAIL", "cases": results,
               "comparison": "fatal: geometry/layout/coverage/history; reported: cross-replica prefix data",
               "prefix_data_policy": PREFIX_DATA_POLICY,
               "payloads": payloads})
    if not passed:
        raise RuntimeError("Offline cache audit failed; see audit.json")
    reported = sum(len(entries) for result in results
                   for entries in result["prefix_data_differences"]["per_replica"].values())
    print(f"PASS: {len(results)} cases, four TP replicas, all target/draft groups"
          + (f"; {reported} cross-replica prefix differences reported (non-fatal)" if reported else ""))


def rank_values(given, dp, fallback):
    """把按 rank 给的一组值补齐到 dp 个，不足部分用最后一个值补。"""
    if not given:
        return [fallback] * dp
    values = list(given)
    return [values[index] if index < len(values) else values[-1] for index in range(dp)]


def rank_request_counts(args, dp):
    """每个 DP rank 实际提交多少条请求。

    D01～D05 要的是各 rank 负载不均衡——(4,40)、(40,4)、(8,24)、(16,32)、
    (0,4)、(0,40)——而 --batch 只能给所有 rank 同一个值。--rank-batches 按
    rank 逐个指定实际提交数，不足部分按最后一个值补齐。

    max_num_seqs 仍统一取 --batch（各 rank 的容量必须一致，否则捕获档位会
    因 rank 而异），变的只是实际提交的请求条数。所以 --batch 要给成各 rank
    里的最大值。
    """
    counts = rank_values(args.rank_batches, dp, args.batch)
    if any(count < 0 or count > args.batch for count in counts):
        raise ValueError(f"--rank-batches 的每一项都必须在 0..{args.batch} 之间，得到 {counts}")
    return counts


def stagger_limits(batch, limit):
    """让各请求在不同步数结束，使活跃 batch 逐档下降。

    默认所有请求同一个 max_tokens，会一起结束，活跃 batch 几乎不变——实测
    256 次 metadata build 里只有收尾的 8 次带补位。要覆盖 G04（档位切换）和
    G06（请求退出与位置重排），需要 batch 真的一档一档往下走。

    返回从 limit 递减到约 limit/batch 的一组上限，最长的那条保持 limit，
    这样整轮时长不变、可与非错开的轮次直接比较。
    """
    if batch <= 1:
        return [limit]
    step = max(1, limit // batch)
    return [max(1, limit - index * step) for index in range(batch)]


def spec_decode_metrics(llm, baseline=None):
    """从 Prometheus 快照取 DSpark 的真实接受统计（T2.4／T4.4）。

    走 llm.get_metrics() 这条公开出口，而不是钻 EngineCore：SpecDecodingStats 由
    scheduler 聚合，而 scheduler 与 worker 不在同一进程，worker extension 的
    collective_rpc 够不着它。LLM 构造时已设 disable_log_stats=False，否则取不到。

    自然接受长度按 num_accepted_tokens / num_drafts 算，不注入任何假定值：
    每次 draft 除了被接受的草稿 token，还必定产出 1 个由目标模型给出的 token，
    所以每步实际推进为该比值加一。per_pos 是逐位置接受数，用来看接受在草稿
    序列上的分布，而不是只看一个平均值。
    """
    counters, per_pos = {}, None
    for metric in llm.get_metrics():
        if metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            per_pos = list(getattr(metric, "values", []) or [])
        elif metric.name.startswith("vllm:spec_decode_"):
            counters[metric.name[len("vllm:spec_decode_"):]] = getattr(metric, "value", None)
    if baseline is not None:
        # 同一次加载扫描多个 batch 时，只统计本档预热和两轮采集的增量。
        # 第一次生成前指标可能尚未注册；已有非零计数则不允许缺失逐位置快照。
        before = baseline["counters"]
        if before.keys() - counters.keys():
            raise ValueError("DSpark 累计计数在扫描期间缺失")
        for name, value in counters.items():
            previous = before.get(name, 0)
            if any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 or n != int(n)
                   for n in (value, previous)) or value < previous:
                raise ValueError(f"DSpark 累计计数重置或无效：{name}")
            counters[name] = value - previous
        old_pos = baseline["num_accepted_tokens_per_pos"]
        if old_pos is None and any(before.values()):
            raise ValueError("已有 DSpark 计数，但缺少本档开始前的逐位置快照")
        if per_pos is not None:
            old_pos = [0] * len(per_pos) if old_pos is None else old_pos
            if len(old_pos) != len(per_pos) or any(
                type(n) not in (int, float) or not math.isfinite(n) or n < 0 or n != int(n)
                for n in [*old_pos, *per_pos]
            ) or any(new < old for new, old in zip(per_pos, old_pos)):
                raise ValueError("DSpark 逐位置累计计数重置或无效")
            per_pos = [new - old for new, old in zip(per_pos, old_pos)]
    drafts = counters.get("num_drafts") or 0
    draft_tokens = counters.get("num_draft_tokens") or 0
    accepted = counters.get("num_accepted_tokens") or 0
    report = {"counters": counters, "num_accepted_tokens_per_pos": per_pos}
    if drafts:
        report["accepted_per_draft"] = accepted / drafts
        report["advance_per_step"] = accepted / drafts + 1.0
        report["draft_tokens_per_draft"] = draft_tokens / drafts
    if draft_tokens:
        report["acceptance_rate"] = accepted / draft_tokens
    if per_pos and drafts:
        report["acceptance_rate_per_pos"] = [value / drafts for value in per_pos]
    # 取不到就如实留空，由读者判断，不要用名义 num_speculative_tokens 顶替。
    report["sufficient"] = bool(drafts)
    return report


def generate_round(llm, args, case, limit, stagger=False):
    """每轮都用同一个 offline_key 重新提交，缓存由 connector 从同一 bank 初态恢复。"""
    from vllm import SamplingParams

    tokens = json.loads((args.bank / case["tokens"]).read_text())

    def make(max_tokens):
        return SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True,
                              extra_args={"kv_transfer_params": {"offline_key": case["key"]}})

    submitted = args.rank_batch if args.rank_batch is not None else args.batch
    limits = stagger_limits(submitted, limit) if stagger else None
    params = [make(value) for value in limits] if limits else make(limit)
    start = time.perf_counter()
    if submitted == 0 and args.command not in ("performance", "moe-routing"):
        # 注意语义：这里直接返回，**引擎根本没被调用**，所以它不是 D04 要的
        # "rank 参与 DP 但没有调度到请求"那条路径，只是"这个 rank 本轮不出题"。
        # 真实空 rank 要用 --rank-decode-tokens 让某个 rank 的请求提前跑完，
        # 其余 rank 继续推进，那时它的引擎才会在有 DP 协调的前提下空转。
        return {"elapsed_seconds": 0.0, "max_tokens_per_request": [], "output_token_ids": []}
    prompts = [{"prompt_token_ids": tokens}] * submitted
    if args.command in ("performance", "moe-routing"):
        from offline_pd.batch import generate_aligned_batch

        result = generate_aligned_batch(llm, prompts, params)
    else:
        result = llm.generate(prompts, params, use_tqdm=False)
    return {"elapsed_seconds": time.perf_counter() - start,
            "max_tokens_per_request": limits or [limit] * submitted,
            "output_token_ids": [list(r.outputs[0].token_ids) for r in result]}


def diagnostic_runs(args):
    """模型容量固定为 --batch；扫描时每档重新指定真实请求数及独立输出目录。"""
    for batch in args.sweep_batches or [args.batch]:
        current = argparse.Namespace(**vars(args))
        current.max_num_seqs = args.batch
        if args.sweep_batches:
            current.batch = current.rank_batch = batch
            current.output = args.output.parent / f"b{batch}" / args.backend
        yield current


def token_budget(args, plan):
    """DSpark 并行出 n 验 n+1；容量内每条请求还须预留 n-1 个草稿槽位。"""
    if args.max_num_batched_tokens is not None:
        return args.max_num_batched_tokens
    if args.command == "prefill":
        return 2048
    draft_tokens = plan["decode"]["speculative_tokens"]
    return max(256, args.batch * ((draft_tokens + 1) + (draft_tokens - 1)))


def diagnose(args, llm, cases):
    """先预热掉首次编译和缓存冷读，再单独开一次诊断窗口。

    profile 采 PyTorch/NPU 数据用于 Native/PTO 结构对照，swimlane 采一次真实 CSA
    调用的 PTO DFX 记录。两者都带同步和采集开销，各自单独运行，不互相混用，
    也不作为稳态延迟或吞吐结论。
    """
    # 稳态构成直接取生产入口的 S，不在测试里另写一份常量。
    from vllm_ascend import envs as ascend_envs
    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.service_config import QUERY_TOKENS

    case = cases[0]
    # 按 rank 的 decode 步数：让某个 rank 提前跑完，其余继续推进，
    # 那个 rank 的引擎才会在有 DP 协调的前提下空转（D04 / T1.6）。
    if args.rank_decode_token is not None:
        args.decode_tokens = args.rank_decode_token
    expected_tokens = args.batch * QUERY_TOKENS
    if args.command == "moe-routing" and args.rank_batch is not None:
        expected_tokens = args.rank_batch * QUERY_TOKENS
    stats_before = spec_decode_metrics(llm) if args.sweep_batches else None
    warmup = [generate_round(llm, args, case, args.warmup_tokens)["elapsed_seconds"]
              for _ in range(args.warmup_rounds)]
    common = {"command": args.command, "backend": args.backend, "rank": args.rank,
              "batch": args.batch, "max_num_seqs": args.max_num_seqs,
              "max_num_batched_tokens": args.max_num_batched_tokens,
              "submitted": args.rank_batch if args.rank_batch is not None else args.batch,
              "key": case["key"], "history": case["history"],
              "warmup_rounds": args.warmup_rounds, "warmup_tokens": args.warmup_tokens,
              "requested_steady_cycles": args.steady_cycles,
              "warmup_elapsed_seconds": warmup, "expected_tokens": expected_tokens,
              "weight_nz_mode": args.weight_nz_mode, "graph_mode": args.graph_mode,
              "capture_sizes": args.capture_sizes,
              "batch_admission": ("pause_enqueue_dp_barrier_resume"
                                  if args.command in ("performance", "moe-routing") else "streaming_generate"),
              "deterministic": args.deterministic,
              "hccl_deterministic": os.environ.get("HCCL_DETERMINISTIC", "false"),
              "variant": os.environ.get("PTO_CSA_VARIANT", "precision"),
              "atomic_add": str(ascend_envs.VLLM_ASCEND_PTO_CSA_ATOMIC_ADD),
              "eplb_enabled": False,
              "dynamic_eplb_env": os.environ.get("DYNAMIC_EPLB", "false"),
              "expert_map_record_env": os.environ.get("EXPERT_MAP_RECORD", "false"),
              "worker_runtime_config": args.worker_runtime_config,
              "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")}
    if args.command == "performance":
        # 当前主口径仅测 decode forward；完整周期由独立 steady 诊断入口保留。
        # _model_forward 之外的 metadata/logits/采样/草稿不计入本轮性能判断。
        llm.collective_rpc("offline_begin_observation")
        llm.collective_rpc("offline_begin_forward", args=(
            args.warmup_steps, expected_tokens, args.batch, args.steady_cycles, args.forward_host_diagnostics))
        common["forward_host_diagnostics"] = args.forward_host_diagnostics
        common["stage"] = "measuring_forward"
        write_json(args.output / f"rank{args.rank}.performance.json", common)
        try:
            steady_output = generate_round(llm, args, case, args.decode_tokens)
        finally:
            common["steady_window"] = llm.collective_rpc("offline_end_forward")
            write_json(args.output / f"rank{args.rank}.performance.json", common)
        common["steady_output_token_ids"] = steady_output["output_token_ids"]
        common["stage"] = "measuring_layer_intervals"
        if not common["steady_window"] or not all(w["sufficient"] for w in common["steady_window"]):
            write_json(args.output / f"rank{args.rank}.performance.json", common)
            raise RuntimeError(f"约定档位的稳态设备样本不足 {args.steady_cycles} 个，"
                               "或事件时间戳无效；不启动层区间采集")
    if args.command in ("profile", "performance"):
        level = 0 if args.command == "performance" else 1
        started = llm.collective_rpc("offline_begin_profile", args=(
            str((args.output / "trace").resolve()), args.profile_start_step,
            args.profile_steps, expected_tokens, args.batch, level, args.profile_forward_events))
        measured = generate_round(llm, args, case, args.decode_tokens)
        window = llm.collective_rpc("offline_end_profile")
        common.update({
            "decode_tokens": args.decode_tokens,
            "profiler": {"activities": ["CPU", "NPU"], "profiler_level": f"Level{level}",
                         "with_stack": False, "with_modules": False,
                         "record_shapes": False, "profile_memory": False},
            "started": started, "window": window,
            "measured_elapsed_seconds": measured["elapsed_seconds"],
            "output_token_ids": measured["output_token_ids"],
            "scope": "窗口含采集与同步开销，仅用于Native/PTO结构对照，不是稳态性能结论",
        })
        if args.command == "performance":
            common["csa_observation"] = llm.collective_rpc("offline_end_observation")
            common["stage"] = "measured"
            common["scope"] = ("steady_window 仅记录 warmup 后 _model_forward 的设备耗时；"
                               "独立 Level0 trace 用于 HC_pre→HC_post 设备首末区间，"
                               "须解析各 rank/层，不累加并发 kernel 时间。")
    elif args.command == "moe-routing":
        started = llm.collective_rpc("offline_begin_moe_routing", args=(
            args.warmup_steps, expected_tokens, common["submitted"], args.compare_samples))
        measured = generate_round(llm, args, case, args.decode_tokens)
        window = llm.collective_rpc("offline_end_moe_routing")
        common.update(started=started, window=window, output_token_ids=measured["output_token_ids"],
                      scope="图内复制路由整数、图外同步读取的诊断；本轮不报告性能")
    elif args.command == "hostprofile":
        started = llm.collective_rpc("offline_begin_host_profile", args=(
            str((args.output / "host").resolve()), args.profile_start_step,
            args.profile_steps, expected_tokens, args.batch))
        measured = generate_round(llm, args, case, args.decode_tokens)
        window = llm.collective_rpc("offline_end_host_profile")
        common.update({
            "decode_tokens": args.decode_tokens, "started": started, "window": window,
            "measured_elapsed_seconds": measured["elapsed_seconds"],
            "output_token_ids": measured["output_token_ids"],
            "scope": "cProfile放大Python调用开销，只用于主机侧相对归因，不与设备耗时相加",
        })
    elif args.command == "bitcompare":
        started = llm.collective_rpc("offline_begin_bitcompare",
                                     args=(args.swimlane_layer, expected_tokens,
                                           args.compare_samples, args.compare_mode))
        measured = generate_round(llm, args, case, args.decode_tokens)
        common.update({
            "decode_tokens": args.decode_tokens, "layer_index": args.swimlane_layer,
            "compare_samples": args.compare_samples, "compare_mode": args.compare_mode,
            "started": started,
            "window": llm.collective_rpc("offline_end_bitcompare"),
            "measured_elapsed_seconds": measured["elapsed_seconds"],
            "output_token_ids": measured["output_token_ids"],
            "scope": "同一步里先跑PTO再跑Native并逐bit比对该层输出；output留Native值，"
                     "故本轮的token轨迹是Native的，不代表PTO独立运行的结果",
        })
    elif args.command == "steady":
        started = llm.collective_rpc("offline_begin_steady", args=(
            args.warmup_steps, expected_tokens, args.batch, args.steady_cycles))
        common.update({"decode_tokens": args.decode_tokens, "warmup_steps": args.warmup_steps,
                       "started": started, "stage": "measuring",
                       "scope": "约定档位的图外事件同时记录 execute_model 和连续完整 decode 周期；"
                                "主机时间另列，不把调度 token 速率当成输出吞吐"})
        # 先落一份带 stage 的记录：这一轮多次被外部信号在 decode 中途打断，
        # 而收尾才写盘导致什么都拿不到。哪怕只走到这里，也要留下证据。
        write_json(args.output / f"rank{args.rank}.{args.command}.json", common)
        try:
            measured = generate_round(llm, args, case, args.decode_tokens)
            common.update({"measured_elapsed_seconds": measured["elapsed_seconds"],
                           "output_token_ids": measured["output_token_ids"]})
        finally:
            # 即使 generate 被打断，也把已采到的单步耗时取回来。取不回来也不能让
            # 异常吃掉后面的写盘——上一轮就是 RPC 在引擎被杀时抛错，导致 16 份
            # 记录全停在 stage=measuring。
            try:
                common["window"] = llm.collective_rpc("offline_end_steady")
            except BaseException as exc:  # noqa: BLE001 - 收尾尽力而为
                common["window"] = None
                common["window_error"] = f"{type(exc).__name__}: {exc}"
            common["stage"] = "measured" if "measured_elapsed_seconds" in common else "interrupted"
            write_json(args.output / f"rank{args.rank}.{args.command}.json", common)
    elif args.command == "padding-capture":
        # 补位只在收尾阶段自然出现，所以按完整 decode_tokens 跑，让尾部请求陆续结束。
        started = llm.collective_rpc("offline_begin_padding_capture", args=(
            str((args.output / "padding").resolve()), args.swimlane_layer))
        measured = generate_round(llm, args, case, args.decode_tokens, stagger=args.stagger)
        window = llm.collective_rpc("offline_end_padding_capture")
        common.update({
            "decode_tokens": args.decode_tokens, "layer_index": args.swimlane_layer,
            "stagger": args.stagger, "max_tokens_per_request": measured["max_tokens_per_request"],
            "started": started, "window": window,
            "measured_elapsed_seconds": measured["elapsed_seconds"],
            "output_token_ids": measured["output_token_ids"],
            "scope": "只读取metadata并多调一次Native自己的compact生产器量形状，不改变本步计算结果",
        })
    elif args.command == "argdump":
        # 把一次真实 CSA 调用的根入参落盘，之后单卡 bench 反复回放，
        # 不必为每次 kernel 改动再起一次 16 卡整模型。
        target = (args.output / "csa_args").resolve()
        started = llm.collective_rpc("offline_begin_argdump",
                                     args=(args.swimlane_layer, str(target), expected_tokens))
        measured = generate_round(llm, args, case, args.warmup_tokens)
        window = llm.collective_rpc("offline_end_argdump")
        common.update({
            "decode_tokens": args.warmup_tokens, "layer_index": args.swimlane_layer,
            "started": started, "window": window, "argdump_dir": str(target),
            "measured_elapsed_seconds": measured["elapsed_seconds"],
            "scope": "只落盘入参，不改变本步计算结果",
        })
    else:
        started = llm.collective_rpc("offline_begin_swimlane",
                                     args=(args.swimlane_layer, expected_tokens))
        measured = generate_round(llm, args, case, args.warmup_tokens)
        window = llm.collective_rpc("offline_end_swimlane")
        common.update({
            "decode_tokens": args.warmup_tokens, "swimlane_layer": args.swimlane_layer,
            "started": started, "window": window,
            "measured_elapsed_seconds": measured["elapsed_seconds"],
            "output_token_ids": measured["output_token_ids"],
            "scope": "DFX诊断窗口带边界同步开销，只用于查看任务依赖，不参与耗时对比",
        })
    common["spec_decode"] = spec_decode_metrics(llm, baseline=stats_before)
    common["spec_decode_scope"] = "本档预热和采集轮次"
    if stats_before is not None:
        common["spec_decode_baseline"] = stats_before
    write_json(args.output / f"rank{args.rank}.{args.command}.json", common)
    if args.command in ("profile", "performance") and (
        not common["window"] or not all(w["sufficient"] for w in common["window"])
    ):
        raise RuntimeError("设备 trace 未覆盖指定档位的连续完整窗口；详见已落盘的观测分布")
    if args.command == "bitcompare" and (
        not common["window"] or any(item.get("status") == "FAIL" for item in common["window"])
    ):
        raise RuntimeError("CSA 数值诊断样本不足或包含无效张量；详见已落盘的 bitcompare 记录")
    if args.command == "moe-routing" and (
        not common["window"] or not all(item["sufficient"] for item in common["window"])
    ):
        raise RuntimeError("路由诊断未取得约定的满档 step 和完整主模型层")


def profile_export(args):
    """离线解析采集到的 PROF 原始数据，生成带 device kernel 的 trace_view.json。

    解析是纯 CPU 工作，放在采集任务之外执行：既不占用设备队列，也便于按需重跑。
    原始数据保持不变，同一批数据重复解析得到同样的导出文件。
    """
    from torch_npu.profiler.profiler import analyse

    root = (args.output / "trace").resolve()
    wanted = None if args.profile_ranks == "all" else {f"rank{r}" for r in args.profile_ranks.split(",")}
    exported = []
    for directory in sorted(root.glob("rank*"), key=lambda p: int(p.name.removeprefix("rank"))):
        if wanted is not None and directory.name not in wanted:
            continue
        captures = sorted(directory.glob("*_ascend_pt"))
        if len(captures) != 1:
            raise ValueError(f"Expected one capture under {directory}, found {len(captures)}")
        capture = captures[0]
        output = capture / "ASCEND_PROFILER_OUTPUT"
        if not output.is_dir():
            analyse(profiler_path=str(capture), max_process_number=args.analyse_processes)
        trace = output / "trace_view.json"
        if not trace.is_file():
            raise RuntimeError(f"Offline analysis produced no trace_view.json under {capture}")
        exported.append({"rank": directory.name, "capture": str(capture), "trace_view": str(trace),
                         "trace_view_bytes": trace.stat().st_size,
                         "exported_files": sorted(p.name for p in output.iterdir())})
    if not exported:
        raise ValueError(f"No capture matched --profile-ranks {args.profile_ranks} under {root}")
    report = {"command": "profile-export", "root": str(root), "exported": exported,
              "scope": "仅离线解析已采集数据，不改变原始PROF记录，也不产生新的执行结果"}
    write_json(args.output / "profile_export.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def profile_compare(args):
    """对照两次采集：同一 bank/case/batch 与同一 step 窗口，逐个 device kernel 列差异。

    只汇总解析出来的实际 kernel 记录，不另立分类口径。窗口内含 EP 全互联的等待时间，
    也含采集与同步开销，因此这里给的是结构对照，不是稳态延迟或吞吐结论。
    """
    import csv

    rank = f"rank{args.profile_ranks}"

    def load(backend):
        root = args.output / backend
        meta = json.loads((root / f"{rank}.profile.json").read_text())
        if meta["backend"] != backend or meta["command"] != "profile":
            raise ValueError(f"{root} does not hold a {backend} profile capture")
        entry = next(item for item in json.loads((root / "profile_export.json").read_text())["exported"]
                     if item["rank"] == rank)
        totals, counts, cores = {}, {}, {}
        with (Path(entry["trace_view"]).parent / "kernel_details.csv").open() as handle:
            for row in csv.DictReader(handle):
                name, duration = row["Name"], float(row["Duration(us)"])
                totals[name] = totals.get(name, 0.0) + duration
                counts[name] = counts.get(name, 0) + 1
                cores[row["Accelerator Core"]] = cores.get(row["Accelerator Core"], 0.0) + duration
        return {"meta": meta, "entry": entry, "totals": totals, "counts": counts, "cores": cores}

    native, pto = load("native"), load("pto")
    for key in ("key", "history", "batch", "expected_tokens", "decode_tokens"):
        if native["meta"][key] != pto["meta"][key]:
            raise ValueError(f"Captures differ in {key}; they are not comparable")
    windows = {side["meta"]["backend"]: side["meta"]["window"][0] for side in (native, pto)}
    if {w["profiled_steps"] for w in windows.values()} != {args.profile_steps}:
        raise ValueError(f"Both captures must cover {args.profile_steps} steady steps: {windows}")
    kernels = []
    for name in sorted(native["totals"].keys() | pto["totals"].keys()):
        left, right = native["totals"].get(name, 0.0), pto["totals"].get(name, 0.0)
        kernels.append({"name": name, "native_us": round(left, 1), "pto_us": round(right, 1),
                        "delta_us": round(right - left, 1),
                        "native_calls": native["counts"].get(name, 0),
                        "pto_calls": pto["counts"].get(name, 0)})
    kernels.sort(key=lambda item: abs(item["delta_us"]), reverse=True)
    report = {
        "command": "profile-compare", "rank": rank,
        "fixture": {key: native["meta"][key] for key in ("key", "history", "batch", "expected_tokens")},
        "window": {side: {"steady_step_indices": [s["steady_step_index"] for s in window["window"]],
                          "scheduled_tokens": [s["scheduled_tokens"] for s in window["window"]],
                          "requests": [s["requests"] for s in window["window"]]}
                   for side, window in windows.items()},
        "device_totals_us": {side["meta"]["backend"]: round(sum(side["totals"].values()), 1)
                             for side in (native, pto)},
        "device_kernel_records": {side["meta"]["backend"]: sum(side["counts"].values())
                                  for side in (native, pto)},
        "accelerator_core_us": {side["meta"]["backend"]: {k: round(v, 1) for k, v in sorted(side["cores"].items())}
                                for side in (native, pto)},
        "kernels_by_absolute_delta": kernels[:args.compare_top],
        "output_token_ids_equal": native["meta"]["output_token_ids"] == pto["meta"]["output_token_ids"],
        "trace_view": {side["meta"]["backend"]: {"path": side["entry"]["trace_view"],
                                                 "bytes": side["entry"]["trace_view_bytes"]}
                       for side in (native, pto)},
        "scope": "3个稳态decode step的结构对照；窗口含EP等待、采集与同步开销，不是稳态性能结论",
    }
    write_json(args.output / "profile_comparison.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "kernels_by_absolute_delta"},
                     indent=2, ensure_ascii=False))
    print(f"\n{'kernel':<54}{'native us':>12}{'pto us':>12}{'delta us':>12}")
    for item in kernels[:args.compare_top]:
        print(f"{item['name'][:52]:<54}{item['native_us']:>12.1f}{item['pto_us']:>12.1f}{item['delta_us']:>12.1f}")


def swimlane_export(args):
    """把 DFX 记录转成可直接打开的泳道图，任务名取自本次实际生成的 kernel_config。"""
    import ast
    import shutil

    directory = args.output / "swimlane"
    records, deps = directory / "chip_swimlane_records.json", directory / "deps.json"
    raw = json.loads(records.read_text())
    metadata = raw.get("metadata", {})
    boundaries = metadata.get("run_boundaries", [])
    if len(boundaries) != 1 or metadata.get("dropped_run_boundaries", 0):
        raise ValueError(f"Expected exactly one complete DFX window: {metadata}")
    tasks = raw.get("aicore_tasks", [])
    if not tasks or not deps.is_file():
        raise ValueError("DFX capture produced no AICore task or dependency record")
    table = args.kernel_config
    if table is None:
        # 开了 DFX 的 rank 会单独编译一份 kernel，直接从它自己的日志里取实际路径，
        # build_output 下还有其他 rank 和历史运行的同名产物，不能按时间或数量猜。
        log = (args.output / f"rank{args.swimlane_rank}.log").read_text()
        used = sorted(set(re.findall(r"build_output/_jit__decode_csa_tp1_attention_\w+", log)))
        if len(used) != 1:
            raise ValueError(f"Pass --kernel-config; rank{args.swimlane_rank} log names {len(used)} JIT builds")
        table = Path(used[0]) / "kernel_config.py"
    tree = ast.parse(table.read_text())
    tables = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "KERNELS" for target in node.targets)]
    if len(tables) != 1 or not isinstance(tables[0], ast.List):
        raise ValueError(f"Unexpected KERNELS table in {table}")
    names = {}
    for node in tables[0].elts:
        values = {ast.literal_eval(key): value for key, value in zip(node.keys, node.values)}
        names[str(ast.literal_eval(values["func_id"]))] = ast.literal_eval(values["name"])
    active = {str(k) for task in json.loads(deps.read_text())["tasks"] for k in task["kernel_ids"] if k >= 0}
    if active - names.keys():
        raise ValueError(f"Capture used kernel ids absent from {table}: {sorted(active - names.keys())}")
    shutil.copy2(table, directory / "kernel_config_source.py")
    name_map = directory / "name_map.json"
    write_json(name_map, {"callable_id_to_name": names})
    merged = directory / "merged_swimlane.json"
    # 转换器会打印任务数、执行与调度占比，存档下来，避免只靠终端回看。
    converted = subprocess.run([sys.executable, "-m", "simpler_setup.tools.swimlane_converter", str(records),
                                "--func-names", str(name_map), "-o", str(merged)],
                               check=True, capture_output=True, text=True)
    (directory / "converter_output.txt").write_text(converted.stdout + converted.stderr)
    events = json.loads(merged.read_text())["traceEvents"]
    slices = sum(event.get("ph") == "X" and event.get("cat") == "event" for event in events)
    if slices < len(tasks):
        raise ValueError(f"Converted {slices} device slices for {len(tasks)} AICore tasks")
    report = {"captured_pypto_launches": len(boundaries), "aicore_task_count": len(tasks),
              "converted_device_slices": slices, "named_callables": len(names),
              "kernel_config_source": str(table),
              "chip_swimlane_records": str(records), "deps_json": str(deps),
              "merged_swimlane": str(merged),
              "scope": "DFX诊断窗口带边界同步开销，只用于查看任务与依赖，不参与耗时对比"}
    write_json(directory / "swimlane_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def worker(args):
    # 根布局在模块导入时固定，必须先把显式 CLI 同步到环境，再导入 vLLM/PTO。
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = str(args.weight_nz_mode)
    os.environ["DYNAMIC_EPLB"] = "false"
    os.environ["EXPERT_MAP_RECORD"] = "false"
    print(f"OFFLINE_WEIGHT_NZ requested={args.weight_nz_mode} "
          f"environment={os.environ['VLLM_ASCEND_ENABLE_NZ']} rank={args.rank} "
          f"backend={args.backend}", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.platforms import current_platform

    # 父进程退出时会给整个子进程组发 SIGTERM，默认动作是立即终止，finally 不会执行，
    # 已采到的数据就全丢了。转成 SystemExit 让清理与落盘有机会跑完。
    def terminated(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminated)

    # Match `vllm serve` model/config registration before constructing LLM.
    current_platform.pre_register_and_update()

    plan = read_plan(args.bank)
    prefill = args.command == "prefill"
    args.max_num_batched_tokens = token_budget(args, plan)
    cases = [c for c in plan["cases"] if c["p_dp_rank"] == args.rank % 4]
    connector = KVTransferConfig(
        kv_connector="OfflineDSV4Connector", kv_connector_module_path="offline_pd.connector",
        kv_role="kv_producer" if prefill else "kv_consumer",
        kv_connector_extra_config={"bank": str(args.bank.resolve())},
    )
    overrides = {"sliding_window": 128}
    if not prefill and args.backend == "pto":
        overrides["architectures"] = ["PyptoCSADeepseekV4ForCausalLM"]
    llm = LLM(
        model=plan["model"], tokenizer_mode="deepseek_v4", trust_remote_code=True,
        worker_cls="offline_pd.worker.OfflineNPUWorker",
        tensor_parallel_size=4 if prefill else 1, enable_expert_parallel=True,
        dtype="bfloat16", quantization="ascend", hf_overrides=overrides,
        max_model_len=max(c["history"] for c in cases) + args.decode_tokens + 32,
        max_num_seqs=1 if prefill else args.batch,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False, enforce_eager=prefill or args.graph_mode == "eager", seed=1024,
        gpu_memory_utilization=0.9, block_size=32,
        # 清单约定的 D 侧上线口径为 FULL_DECODE_ONLY；
        # eager 只用于定位问题，其每步重入 Python 派发路径，不代表上线表现。
        # draft 保持 eager。两侧 NZ mode 均由显式 CLI 控制。
        # 捕获档位默认由 vLLM 按 max_num_seqs*6 截断默认列表得到，最大档可能小于
        # potential_max_tokens（实测档位 [1,2,4,8,16,24] 而 potential 为 30）。
        # 那会让 mc2_tokens_capacity 小于 potential，A3 上 select_moe_comm_method
        # 改选 ALLTOALL，should_skip_allreduce_across_dp_group 随之为假，
        # 最终触发 service_config.py 的 PTO DP 闸门。显式给 6 的倍数档位可避开。
        # 默认档位由 platform.py 对齐到 uniform_decode_query_len（DSpark 下即 6），
        # 这样每一档都能还原成整数条请求，PTO 可以捕获全部档位。
        # --capture-sizes 用于造"档位不是 6 的倍数"的反例场景，那时必须同时关掉对齐，
        # 否则给的 16 会被取整成 18，场景就造不出来。
        **({} if prefill or args.graph_mode == "eager"
           else {"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY",
                                        **({} if not args.capture_sizes
                                           else {"cudagraph_capture_sizes": list(args.capture_sizes)})}}),
        # Release A3 Native chooses INT8 Indexer storage in its constructor;
        # the upstream release AttentionConfig does not accept an int8 Literal.
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        # weight_nz_mode 由 --weight-nz-mode 控制：0 全 ND，1 是 vllm-ascend 的默认值，
        # Native 会把量化权重转成 FRACTAL_NZ，PTO 按匹配的根签名直接借用存储。
        additional_config={"weight_nz_mode": args.weight_nz_mode, "enable_kv_nz": False, "enable_dsa_cp": False,
                           "offline_deterministic_level": int(args.deterministic),
                           "offline_event_work_mode": args.event_work_mode,
                           **({"offline_moe_routing_tokens": (args.rank_batch or args.batch)
                               * (plan["decode"]["speculative_tokens"] + 1)}
                              if args.command == "moe-routing" else {}),
                           # 本机 CANN 9.0.0 的 libopapi.so 与已构建的 CSA 自定义算子包里都没有
                           # aclnnAddRmsNormBias。norm_quant 融合 pass 的 pattern 里直接调用
                           # npu_add_rms_norm_bias，而 PyTorch 的 pattern matcher 用
                           # tracing_mode="real" 追踪 pattern，等于真的执行一次，于是图编译在
                           # 建 pattern 阶段就崩。关掉该融合即可，属本地环境适配，不改生产代码。
                           # 注意这偏离上线口径：参考脚本所在环境有该算子，融合是开启的。
                           **({} if prefill or args.graph_mode == "eager"
                              else {"ascend_compilation_config": {
                                  "fuse_norm_quant": False,
                                  # 显式给档位时关掉对齐，让非 6 倍数的档位原样保留。
                                  **({} if not args.capture_sizes
                                     else {"align_decode_capture_sizes": False})}}),
                           **({} if not args.recompute_scheduler
                              else {"recompute_scheduler_enable": True}),
                           # T1.10：embedding TP 走组内 all_gather，要求各 rank 贡献的 token
                           # 数一致，因此即便 eager 也会引出 DP 补齐。_forward_embed_tp 的
                           # 静态缓冲按 get_potential_max_tokens() 预分配，超了直接 ValueError，
                           # 所以补齐后的 token 数是否仍在容量内是这项的关键判据。
                           **({} if not args.embedding_tp
                              else {"finegrained_tp_config": {
                                  "embedding_tensor_parallel_size": args.embedding_tp}}),
                           # EP 保留；所有 CSA 功能/性能测试均关闭动态 EPLB。
                           "eplb_config": {"dynamic_eplb": False}},
        model_loader_extra_config={"enable_multithread_load": True, "num_threads": 16},
        kv_transfer_config=connector, disable_log_stats=False,
        **({} if prefill else {"worker_extension_cls": "offline_pd.observer.OfflineCSAObserver"}),
    )
    print(f"OFFLINE_MODEL_READY role={args.command} dp={args.rank}", flush=True)
    args.worker_runtime_config = llm.collective_rpc("offline_runtime_config")
    if not args.worker_runtime_config or any(
        config["deterministic_level"] != int(args.deterministic) or config["dynamic_eplb"]
        for config in args.worker_runtime_config
    ):
        raise RuntimeError(f"Worker 实际确定性/EPLB 配置不符：{args.worker_runtime_config}")
    if args.event_work_mode is not None and any(
        config["cann_event_work_mode"] != args.event_work_mode for config in args.worker_runtime_config
    ):
        raise RuntimeError(f"模型加载后的 CANN event 模式与请求不符：{args.worker_runtime_config}")
    if args.command == "performance":
        required = max(args.sweep_batches or [args.batch]) * (plan["decode"]["speculative_tokens"] + 1)
        if any(config["scheduler"]["max_num_scheduled_tokens"] < required
               for config in args.worker_runtime_config):
            raise ValueError(f"扣除草稿预留后的调度预算不足 {required}，无法测量满档 forward："
                             f"{args.worker_runtime_config}")
    if args.layout_only:
        write_json(args.output / f"rank{args.rank}.cache_layout.json",
                   llm.collective_rpc("offline_cache_layout"))
        llm.llm_engine.engine_core.shutdown()
        return
    if args.command in ("profile", "performance", "hostprofile", "swimlane", "padding-capture", "steady", "bitcompare",
                        "argdump", "moe-routing"):
        try:
            for current in diagnostic_runs(args):
                current.output.mkdir(parents=True, exist_ok=True)
                print(f"OFFLINE_BATCH_START batch={current.batch} capacity={current.max_num_seqs} "
                      f"rank={args.rank}", flush=True)
                diagnose(current, llm, cases)
                print(f"OFFLINE_BATCH_DONE batch={current.batch} rank={args.rank}", flush=True)
        finally:
            llm.llm_engine.engine_core.shutdown()
        return
    outputs = []
    for case in cases:
        tokens = json.loads((args.bank / case["tokens"]).read_text())
        params = SamplingParams(temperature=0, max_tokens=1 if prefill else args.decode_tokens,
                                ignore_eos=True,
                                extra_args={"kv_transfer_params": {"offline_key": case["key"]}})
        prompt = {"prompt_token_ids": tokens[:-1] if prefill else tokens}
        if not prefill:
            llm.collective_rpc("offline_begin_observation")
        start = time.perf_counter()
        # decode 也要认 --rank-batches：此前这里写死 args.batch，导致按 rank 指定的
        # 提交数被忽略，所谓"不均衡负载"实际仍是均衡的（eager_pto_imbalanced 那轮
        # 16 个 rank 都提交了 40 条即为此）。max_num_seqs 仍统一取 args.batch。
        submitted = 1 if prefill else (args.rank_batch if args.rank_batch is not None else args.batch)
        result = llm.generate([prompt] * submitted, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        observation = None if prefill else llm.collective_rpc("offline_end_observation")
        outputs.append({"key": case["key"], "submitted": submitted,
                        "elapsed_including_io_seconds": elapsed,
                        "output_token_ids": [list(r.outputs[0].token_ids) for r in result],
                        "csa_observation": observation,
                        "spec_decode": None if prefill else spec_decode_metrics(llm)})
        write_json(args.output / f"rank{args.rank}.json", {"role": args.command, "backend": args.backend,
                   "rank": args.rank, "batch": args.batch, "eplb_enabled": False,
                   "worker_runtime_config": args.worker_runtime_config, "cases": outputs})
        if observation is not None and args.backend == "pto":
            # 改读捕获期的 capture_time_selection：图重放不触发 forward hook，
            # 原先那版遍历 forward_hook_counts，而各层全是空字典时 any(...) 为假、
            # 校验反而通过，等于形同虚设（b1/b4/b8 三组即因此没报错）。
            for rank_stats in observation:
                selection = (rank_stats or {}).get("capture_time_selection")
                if not selection:
                    raise RuntimeError("No capture-time CSA selection recorded; see csa_observation")
                dead = [layer for layer, counts in selection.items()
                        if not any(k.startswith("pto_") and v for k, v in counts.items())]
                if dead:
                    raise RuntimeError(f"These target CSA layers never selected PTO: {dead[:5]}; "
                                       "see csa_observation")
    # Engine shutdown tears down its owned workers; launcher checks all ranks.
    llm.llm_engine.engine_core.shutdown()


def launch(args):
    def interrupted(signum, frame):
        # 多轮任务报 exit=130（SIGINT）而父进程无任何输出，来源一直查不出来。
        # 把信号号、进程与进程组、以及收到信号时的调用栈打出来，让下一轮自证。
        import traceback
        name = signal.Signals(signum).name
        print(f"OFFLINE_SIGNAL {name}({signum}) pid={os.getpid()} pgid={os.getpgrp()} "
              f"ppid={os.getppid()}", flush=True)
        traceback.print_stack(frame)
        sys.stdout.flush()
        sys.stderr.flush()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    devices = os.environ.get("TASK_DEVICE", "").split(",")
    if len(devices) != 16 or any(not d.isdigit() for d in devices) or len(set(devices)) != 16:
        raise RuntimeError("Run through task-submit --device auto --device-num 16")
    if args.command in ("decode", "profile", "performance", "hostprofile", "swimlane", "padding-capture",
                        "steady", "bitcompare", "argdump", "moe-routing"):
        report = json.loads((args.bank / "audit.json").read_text())
        if report["status"] != "PASS":
            raise ValueError("P cache bank must pass audit before D loads it")
    if args.command == "swimlane" and args.backend != "pto":
        raise ValueError("Swimlane capture requires --backend pto")
    if args.command == "bitcompare" and args.graph_mode != "eager":
        # 与 swimlane 同因：钩子挂在 CSAServiceRuntime.__call__ 上，而 ACL Graph 下
        # decode 步是图回放、不再执行 Python forward，包装函数一次都进不去（实测
        # compared=0）。凡是挂在这个点上的诊断都必须用 eager。
        raise ValueError("Bit comparison requires --graph-mode eager")
    if args.command == "swimlane" and args.graph_mode != "eager":
        # 泳道窗口挂在 CSAServiceRuntime.__call__ 上，而 ACL Graph 下 decode 步是图回放，
        # 不再执行 Python forward，包装函数一次都进不去（实测 captured=0）。芯片泳道记录的是
        # kernel 内部各流水线的占用，属于 kernel 自身性质，与它由图回放还是 eager 下发无关，
        # 所以泳道一律用 eager 采，不跟随 decode 性能的 FULL_DECODE_ONLY 口径。
        raise ValueError("Swimlane capture requires --graph-mode eager")
    if args.command == "padding-capture" and args.graph_mode == "eager":
        # 补位只来自图模式：eager 下 cudagraph_mode 为 NONE，allow_dp_padding 随之为 False，
        # 也不注册捕获档位，结构上不产生补位请求（首轮 eager 采集 32 步 0 命中已证实）。
        # 挂载点在 Native 的 metadata builder 上，两个后端都能用。
        raise ValueError("Padding only occurs under graph mode; use --graph-mode full_decode_only")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("rank*.log")):
        raise FileExistsError("Use a fresh --output directory to preserve prior run evidence")
    if args.sweep_batches:
        for current in diagnostic_runs(args):
            if current.output.exists() and any(current.output.iterdir()):
                raise FileExistsError(f"扫描目录已存在数据：{current.output}")
    prefill = args.command == "prefill"
    tp, dp = (4, 4) if prefill else (1, 16)
    rank_counts = rank_request_counts(args, dp)
    rank_tokens = rank_values(args.rank_decode_tokens, dp, args.decode_tokens)
    if any(value != args.decode_tokens for value in rank_tokens):
        print(f"OFFLINE_RANK_DECODE_TOKENS {rank_tokens}", flush=True)
    if any(count != args.batch for count in rank_counts):
        print(f"OFFLINE_RANK_BATCHES {rank_counts}", flush=True)
    children, files = [], []
    try:
        for rank in range(dp):
            env = os.environ.copy()
            env.update({
                "VLLM_DP_SIZE": str(dp), "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": "0",
                "VLLM_DP_MASTER_IP": args.host, "VLLM_DP_MASTER_PORT": str(args.port),
                "ASCEND_RT_VISIBLE_DEVICES": ",".join(devices[rank * tp:(rank + 1) * tp]),
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_USE_V2_MODEL_RUNNER": "0",
                "VLLM_ASCEND_ENABLE_NZ": str(args.weight_nz_mode),
                "OMP_NUM_THREADS": "4", "OMP_PROC_BIND": "false",
                "HCCL_IF_IP": args.host, "HCCL_SOCKET_IFNAME": args.nic,
                "GLOO_SOCKET_IFNAME": args.nic, "TP_SOCKET_IFNAME": args.nic,
                # 采集窗口会在各 rank 上做同步和落盘，给集合通信留出等待余量。
                "HCCL_CONNECT_TIMEOUT": "120",
                "HCCL_EXEC_TIMEOUT": "1800" if args.command != "decode" else "204",
                # decode 侧 HCCL buffer 固定为 1800 MB，
                # prefill 侧为 1024。1024 在 max_num_seqs=40 时不够：算子按
                # ((maxBs*tokenNeedSizeDispatch*epWorldSize*localMoeExpertNum)
                #  + (maxBs*tokenNeedSizeCombine*(k+sharedExpertNum))) * 2 计算，
                # maxBs=240 需 1043MB，实测直接在 npu_moe_distribute_dispatch_v2
                # 报 HCCL_BUFFSIZE_EP is too SMALL（错误码 561002）。
                "HCCL_BUFFSIZE": "1024" if prefill else "1800",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                # HCCL 默认 HCCL_DETERMINISTIC=false（见 libhccl.so 的
                # "HCCL_DETERMINISTIC set by default to [false]"）。开启后集合通信保序归约。
                # 注意 libhccl.so 里还有一条 "Deterministic do not support aiv"，
                # 而这里保留 AIV 展开模式是用户明确要求的；若 HCCL 因此降级或告警，
                # 日志里会有记录，按实测结果判断，不预先改 AIV。
                **({"HCCL_DETERMINISTIC": "true"} if args.deterministic else {}),
                # 不继承父进程的 EPLB 配置，关闭重平衡和专家热度采集。
                "DYNAMIC_EPLB": "false", "EXPERT_MAP_RECORD": "false",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", ""),
            })
            env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
            # DFX 配置必须在模型加载前绑定，只有被选中的 rank 采集泳道。
            if args.command == "swimlane" and rank == args.swimlane_rank:
                env["OFFLINE_PTO_SWIMLANE_DIR"] = str((args.output / "swimlane").resolve())
            cmd = [sys.executable, str(Path(__file__).resolve()), args.command,
                   "--bank", str(args.bank.resolve()), "--output", str(args.output.resolve()),
                   "--rank", str(rank), "--batch", str(args.batch),
                   "--rank-batch", str(rank_counts[rank]),
                   "--rank-decode-token", str(rank_tokens[rank]), "--backend", args.backend,
                   "--weight-nz-mode", str(args.weight_nz_mode),
                   "--decode-tokens", str(args.decode_tokens),
                   "--warmup-rounds", str(args.warmup_rounds),
                   "--warmup-tokens", str(args.warmup_tokens),
                   "--profile-start-step", str(args.profile_start_step),
                   "--profile-steps", str(args.profile_steps),
                   "--warmup-steps", str(args.warmup_steps),
                   "--steady-cycles", str(args.steady_cycles),
                   "--compare-samples", str(args.compare_samples),
                   "--compare-mode", args.compare_mode,
                   "--swimlane-layer", str(args.swimlane_layer),
                   "--graph-mode", args.graph_mode]
            if args.recompute_scheduler:
                cmd.append("--recompute-scheduler")
            if args.profile_forward_events:
                cmd.append("--profile-forward-events")
            if args.embedding_tp:
                cmd += ["--embedding-tp", str(args.embedding_tp)]
            if args.deterministic:
                cmd.append("--deterministic")
            if args.event_work_mode is not None:
                cmd += ["--event-work-mode", str(args.event_work_mode)]
            if args.forward_host_diagnostics:
                cmd.append("--forward-host-diagnostics")
            if args.stagger:
                cmd.append("--stagger")
            if args.capture_sizes:
                cmd += ["--capture-sizes", *[str(size) for size in args.capture_sizes]]
            if args.sweep_batches:
                cmd += ["--sweep-batches", *map(str, args.sweep_batches)]
            if args.max_num_batched_tokens is not None:
                cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
            if args.layout_only:
                cmd.append("--layout-only")
            file = (args.output / f"rank{rank}.log").open("w")
            files.append(file)
            children.append(subprocess.Popen(
                cmd, env=env, stdout=file, stderr=subprocess.STDOUT, start_new_session=True))
        while any(child.poll() is None for child in children):
            failed = [(rank, p.returncode) for rank, p in enumerate(children) if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Offline {args.command} rank failed: {failed}; see {args.output}")
            time.sleep(2)
        if any(p.returncode for p in children):
            raise RuntimeError("An offline rank failed")
    finally:
        for child in children:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for file in files:
            file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "audit", "prefill", "decode", "profile", "performance",
                                            "hostprofile",
                                            "profile-export", "profile-compare",
                                            "swimlane", "swimlane-export",
                                            "padding-capture", "steady", "bitcompare", "argdump", "moe-routing"])
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--histories", default="255,4095,32767,131071,131072,131073")
    parser.add_argument("--host", default="192.168.0.106")
    parser.add_argument("--nic", default="enp23s0f3")
    parser.add_argument("--port", type=int, default=29683)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--backend", choices=["native", "pto"], default="native")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int,
                        help="worker token 总容量，包含 DSpark 草稿预留；默认覆盖 --batch 的出5验6满档")
    parser.add_argument("--sweep-batches", type=int, nargs="+",
                        help="performance 在一次模型加载内扫描实际 batch，容量固定为 --batch；"
                             "每档单独预热、采样和记录 DSpark 增量")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--layout-only", action="store_true", help="加载D模型后仅采集缓存描述符")
    parser.add_argument("--warmup-rounds", type=int, default=1, help="诊断前的预热轮数，排除首次编译与缓存冷读")
    parser.add_argument("--warmup-tokens", type=int, default=96, help="每个预热轮的生成token数")
    parser.add_argument("--embedding-tp", type=int, default=0,
                        help="开启 embedding TP 并指定组大小；0 为关闭（T1.10）")
    parser.add_argument("--compare-mode", choices=["native", "self"], default="native",
                        help="native=与Native对比；self=PTO自比对，用于确认PTO自身是否可复现")
    parser.add_argument("--compare-samples", type=int, default=3,
                        help="bitcompare采集多少个被比对的step；每个样本都要多跑一次Native，代价不低")
    parser.add_argument("--warmup-steps", type=int, default=8,
                        help="steady命令丢弃的前N个decode step，用于排除首次编译与首个恢复步骤")
    parser.add_argument("--steady-cycles", type=int, default=10,
                        help="采集 warmup 后连续满档周期数，默认 10，主结果取均值")
    parser.add_argument("--profile-start-step", type=int, default=8, help="从第几个稳态decode step开始采集")
    parser.add_argument("--profile-steps", type=int, default=3, help="采集的完整decode step数")
    parser.add_argument("--profile-forward-events", action="store_true",
                        help="在profile窗口内另记同一次forward的设备事件；仅供边界诊断，不替代无profiler计时")
    parser.add_argument("--weight-nz-mode", type=int, default=0, choices=(0, 1, 2),
                        help="vllm-ascend 的 weight_nz_mode；1 会让 Native 把量化权重转成 NZ")
    parser.add_argument("--swimlane-rank", type=int, default=0, help="采集PTO DFX泳道的DP rank")
    parser.add_argument("--swimlane-layer", type=int, default=FIRST_TARGET_CSA_LAYER,
                        help="采集泳道的target C4层序号")
    parser.add_argument("--kernel-config", type=Path, help="swimlane-export用于命名任务的JIT kernel_config.py")
    parser.add_argument("--graph-mode", choices=["full_decode_only", "eager"], default="full_decode_only",
                        help="D侧执行模式；上线口径为FULL_DECODE_ONLY，eager仅用于定位问题")
    parser.add_argument("--recompute-scheduler", action="store_true",
                        help="开启recompute_scheduler_enable；DP>1下PTO图模式需要它才能跳过DP padding")
    parser.add_argument("--rank-batches", type=int, nargs="+",
                        help="按DP rank指定各自实际提交的请求数，用于D01~D05的不均衡负载"
                             "与D04/T1.6的空rank；不足部分按最后一个值补齐。"
                             "max_num_seqs仍统一取--batch，所以--batch要给成各rank的最大值")
    parser.add_argument("--rank-decode-tokens", type=int, nargs="+",
                        help="按DP rank指定各自的decode步数。让某个rank的请求提前跑完、其余rank继续推进，\n"
                             "那个rank的引擎才会在有DP协调的前提下空转，这才是D04/T1.6要的真实空rank")
    parser.add_argument("--rank-decode-token", type=int, default=None,
                        help="内部参数：launch按--rank-decode-tokens逐个下发，不要手工指定")
    parser.add_argument("--rank-batch", type=int, default=None,
                        help="内部参数：launch按--rank-batches逐个下发给子进程，不要手工指定")
    parser.add_argument("--stagger", action="store_true",
                        help="让各请求在不同步数结束，使活跃batch逐档下降，"
                             "覆盖G04档位切换与G06请求退出；默认所有请求同时结束，"
                             "活跃batch几乎不变，只有收尾几步才产生补位")
    parser.add_argument("--deterministic", action="store_true",
                        help="开启算子级确定性(set_deterministic_level(1))与HCCL_DETERMINISTIC=true；"
                             "用于排查同一DP组内四个TP副本的缓存差异，保留AIV展开模式不变")
    parser.add_argument("--event-work-mode", type=int, choices=(0, 1),
                        help="仅用于因果诊断：显式设置进程级 CANN event 模式（0 软件 / 1 硬件）")
    parser.add_argument("--forward-host-diagnostics", action="store_true",
                        help="performance的10步另记主机入场时钟和GC事件；不修改GC或增加设备同步")
    parser.add_argument("--capture-sizes", type=int, nargs="+",
                        help="显式指定ACL Graph捕获档位。默认列表按max_num_seqs*6截断后，"
                             "最大档可能盖不住potential_max_tokens，导致MoE选ALLTOALL而非MC2、"
                             "进而让should_skip_allreduce_across_dp_group为假并触发PTO的DP闸门。"
                             "PTO只捕获6的倍数档位，所以这里也应传6的倍数")
    parser.add_argument("--profile-ranks", default="0", help="profile-export要解析的DP rank，all表示全部")
    parser.add_argument("--analyse-processes", type=int, default=16, help="离线解析使用的进程数上限")
    parser.add_argument("--compare-top", type=int, default=25, help="profile-compare列出的kernel差异条数")
    args = parser.parse_args()
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1:
        parser.error("--max-num-batched-tokens 必须大于 0")
    if args.sweep_batches:
        batches = args.sweep_batches
        if args.command != "performance" or args.rank_batches or args.rank_decode_tokens or args.stagger:
            parser.error("--sweep-batches 仅用于均衡 performance，不能混用 rank 覆盖或 stagger")
        if sorted(set(batches)) != batches or min(batches) < 1 or max(batches) > args.batch:
            parser.error("扫描 batch 须严格递增，且在 1..--batch 范围内")
        if args.rank_batch not in (None, args.batch) or args.rank_decode_token not in (None, args.decode_tokens):
            parser.error("扫描不能覆盖单 rank 的请求数或生成长度")
        query = read_plan(args.bank)["decode"]["speculative_tokens"] + 1
        if args.graph_mode != "full_decode_only" or not set(
            b * query for b in [*batches, args.batch]
        ).issubset(args.capture_sizes or []):
            parser.error("扫描须使用 full_decode_only，并显式捕获每个 batch 及容量的 S6 档位")
    if args.command in ("steady", "performance"):
        query = read_plan(args.bank)["decode"]["speculative_tokens"] + 1
        minimum = (args.warmup_steps + args.steady_cycles + 3) * query
        if args.steady_cycles < 10 or args.warmup_steps < 0 or args.decode_tokens < minimum:
            parser.error(f"稳态至少采 10 个周期；--decode-tokens 须 >= {minimum}，为收尾留余量")
    if args.layout_only and args.command != "decode":
        parser.error("--layout-only 仅适用于 decode")
    if args.profile_forward_events and args.command not in ("performance", "profile"):
        parser.error("--profile-forward-events只适用于performance/profile")
    if args.forward_host_diagnostics and args.command != "performance":
        parser.error("--forward-host-diagnostics只适用于performance")
    if args.command == "moe-routing":
        if args.graph_mode != "full_decode_only" or args.compare_samples < 1 or args.warmup_steps < 0:
            parser.error("路由诊断须使用 full_decode_only、非负 warmup_steps 和正 compare_samples")
    if args.command == "plan":
        make_plan(args)
    elif args.command == "audit":
        audit(args)
    elif args.command in ("profile-export", "profile-compare", "swimlane-export"):
        if args.output is None:
            parser.error(f"--output is required for {args.command}")
        {"profile-export": profile_export, "profile-compare": profile_compare,
         "swimlane-export": swimlane_export}[args.command](args)
    elif args.rank >= 0:
        worker(args)
    else:
        if args.output is None:
            parser.error("--output is required for prefill/decode/profile/swimlane")
        launch(args)


if __name__ == "__main__":
    main()
