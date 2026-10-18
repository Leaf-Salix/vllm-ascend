"""复用10/12已校准的图重放与全流整层kernel跨度；不另捕获外图。"""
import math
import statistics

from dsv4_csa_single_layer import collect_state, guard_checks
from dsv4_csa_validation import compare_tensor, compare_topk


def measure_graph_interval(fixture, run, output, topk, reference, *, iters, warmup, require_exact,
                           profile_dir=None):
    """图外事件包住一次整层重放；初态恢复与输出毒化均在计时区间外。"""
    import torch

    initial = {
        name: group["initial"].to(group["allocation"].device)
        for name, group in fixture["groups"].items()
    }

    def reset():
        for name, group in fixture["groups"].items():
            group["allocation"].copy_(initial[name])
        output.fill_(float("nan"))

    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    # 当前 torch_npu 图内 Event 不随重放更新时间戳，必须在图外显式 record。
    start.record()
    end.record()
    reset()
    torch.npu.synchronize()
    # npugraph_ex owns capture/replay; do not nest a manual outer graph.
    samples, timestamps = [], []
    for iteration in range(warmup + iters):
        reset()
        start.record()
        run()
        end.record()
        torch.npu.synchronize()
        stamp = start.recorded_time()
        if timestamps and stamp <= timestamps[-1]:
            raise ValueError("图计时的开始事件时间戳未更新，不能采纳重复的旧时间")
        timestamps.append(stamp)
        if iteration >= warmup:
            elapsed = start.elapsed_time(end) * 1000
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError(f"图外设备事件没有产生有效时间戳：{elapsed}")
            samples.append(elapsed)
    state = collect_state(fixture, output, topk())
    checks = {name: compare_tensor(value, reference[name], 0, 0) for name, value in state.items()}
    guards = guard_checks(fixture)
    if any(check["status"] != "PASS" for check in guards.values()):
        raise ValueError("计时图重放改写了 metadata 或 slot 外存储")
    if any(check.get("nonfinite") != 0 for check in checks.values()):
        raise ValueError("计时图重放出现非有限值或不完整输出")
    if require_exact and any(check["status"] != "PASS" for check in checks.values()):
        raise ValueError("固定规约的计时图与同初态 eager 不一致")
    selection = compare_topk(state["idx_topk"], reference["idx_topk"], (fixture["positions"].cpu() + 1) // 4)
    if selection["status"] == "FAIL":
        raise ValueError("计时图 Top-K 含越界、重复或缺失的索引")
    profile = None
    if profile_dir is not None:
        import torch_npu

        reset()
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        ) as trace:
            for _ in range(30):
                reset()
                start.record()
                run()
                end.record()
                torch.npu.synchronize()
            trace.step()
        profile = {"directory": str(profile_dir), "event_envelope_us": start.elapsed_time(end) * 1000,
                   "scope": "独立30次图重放，保留完整设备跨度样本；不混入无 profiler 采样"}
    ordered = sorted(samples)
    return {
        "samples_us": samples,
        # recorded_time 的原始计数只用于检查更新；耗时单位由 elapsed_time 给出。
        "start_timestamps_raw": timestamps[warmup:],
        "us_min": ordered[0], "us_p50": statistics.median(samples),
        "us_p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "us_max": ordered[-1],
        "eager_comparison": checks, "exact_comparison_required": require_exact,
        "topk_selection": selection, "guards": guards,
        "profile": profile,
    }


def layer_device_interval(profile_dir):
    """依设备流/图ID区分图外reset与整层kernel，保留Native层内真实拷贝。

    profiler把图外恢复也记录为kernel；小batch的一份copy使用运行时DMA，
    不出现在kernel_details中，因此不能假设固定拷贝个数，也不能按名称统删。
    输出毒化在独立的默认流、无图Model ID上，每轮一次；只剔除这个流上
    同属图外的InplaceCopy/InplaceFill。Native图内拷贝仍计入整层时间。
    """
    import csv
    from pathlib import Path
    files=list(Path(profile_dir).glob('*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
    if len(files)!=1:
        raise RuntimeError(f'需要唯一kernel_details.csv，得到{files}')
    rows=list(csv.DictReader(files[0].open()))
    rows.sort(key=lambda r:float(r['Start Time(us)']))
    copy_name='aclnnInplaceCopy_TensorMoveAiCore_TensorMove'
    fill_name='aclnnInplaceFillScalar_FillAiCore_Fill'
    fills=[r for r in rows if r['Name']==fill_name and r['Model ID']=='4294967295']
    streams={r['Stream ID'] for r in fills}
    if len(fills)!=30 or len(streams)!=1:
        raise RuntimeError('需要30次同流图外输出毒化作为重放边界证据')
    reset_stream=next(iter(streams))
    groups=[];current=[];excluded=[]
    for row in rows:
        reset=(row['Model ID']=='4294967295' and row['Stream ID']==reset_stream
               and row['Name'] in (copy_name,fill_name))
        if reset:
            excluded.append(row)
            if current: groups.append(current);current=[]
        else:
            current.append(row)
    if current: groups.append(current)
    if len(groups)!=30:
        raise RuntimeError(f'整层区间不是30段：{len(groups)}')
    counts=[len(g) for g in groups]
    if len(set(counts))!=1:
        raise RuntimeError(f'各步kernel数量不一致：{counts}')
    spans=[max(float(r['Start Time(us)'])+float(r['Duration(us)']) for r in g)
           -min(float(r['Start Time(us)']) for r in g) for g in groups]
    return {'available':True,'replays':30,'span_us':statistics.median(spans),
            'span_us_all':spans,'kernels_per_replay':counts,
            'stream_ids_per_replay':[sorted({r['Stream ID'] for r in g}) for g in groups],
            'model_ids_per_replay':[sorted({r['Model ID'] for r in g}) for g in groups],
            'span_includes_all_streams':True,
            'excluded_reset_kernels':len(excluded),'reset_stream_id':reset_stream,
            'reset_model_id':'4294967295','raw_csv':str(files[0]),
            'scope':'仅剔除图外fixture恢复和输出毒化；保留完整层计算、层内搬运及PTO调度'}
