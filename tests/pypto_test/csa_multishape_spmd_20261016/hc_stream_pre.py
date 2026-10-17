# SPDX-License-Identifier: Apache-2.0
"""T96 的 HC 前段流式候选，保持原 K256 点积和 32 项 RMS 累加顺序。

只作用于实验生成源码。每组两颗 AIV 分担 16 行输入，每次产生 K512；
AIC 用两个 K256 matmul 消费。原 4 份 K4096 投影部分和保持不变。
每个 K512 的平方和独立存到闲置 widen_tail，原 linear_reduce 阶段
再按 K=0..31 依次相加。不能换成四份局部平方和后相加。

READY/ACK 使用现有 FFTS 组内机制，两个交接槽分别编号 8/9、10/11。
ACK 限制同一事件上未消费的通知数量；末尾两次 ACK 必须全部收回。
这改变了 HC 内部依赖，旧 GM 进度模型不能单独证明新协议正确。
"""

import ast
import copy
import random


def protocol_check():
    """检查双 AIV 通知聚合、两槽复用及连续调用的事件排空。

模型假定 FFTS 的组内通知/等待符合现有 API 的语义，不模拟硬件 pipe。
真实数据可见性还需检查生成的 MTE3/MTE2 指令并做真机 exact 验证。
"""
    for seed in range(128):
        rng = random.Random(seed)
        ready, ack = set(), set()
        for generation in range(3):
            programs = []
            for lane in range(2):
                ops = []
                for chunk in range(8):
                    if chunk >= 2:
                        ops.append(("ack_wait", lane, chunk % 2))
                    ops.append(("ready_set", lane, chunk % 2))
                ops.extend(("ack_wait", lane, slot) for slot in range(2))
                programs.append(ops)
            programs.append([op for chunk in range(8)
                             for op in (("ready_wait", 2, chunk % 2), ("ack_set", 2, chunk % 2))])
            positions = [0, 0, 0]
            while any(p < len(ops) for p, ops in zip(positions, programs)):
                candidates = []
                for peer, ops in enumerate(programs):
                    if positions[peer] == len(ops):
                        continue
                    op, lane, slot = ops[positions[peer]]
                    if op == "ack_wait" and (lane, slot) not in ack:
                        continue
                    if op == "ready_wait" and not all((v, slot) in ready for v in range(2)):
                        continue
                    candidates.append(peer)
                if not candidates:
                    raise ValueError("HC 双槽协议出现死锁")
                peer = rng.choice(candidates)
                op, lane, slot = programs[peer][positions[peer]]
                if op == "ready_set":
                    if (lane, slot) in ready:
                        raise ValueError("READY 未被消费就再次发布")
                    ready.add((lane, slot))
                elif op == "ready_wait":
                    ready.difference_update((v, slot) for v in range(2))
                elif op == "ack_set":
                    if any((v, slot) in ack for v in range(2)):
                        raise ValueError("ACK 未被消费就再次发布")
                    ack.update((v, slot) for v in range(2))
                else:
                    ack.remove((lane, slot))
                positions[peer] += 1
            if ready or ack:
                raise ValueError("调用返回后仍有未排空事件")
    return {"random_interleavings": 128, "generations": 3, "no_unconsumed_event_overwrite": True,
            "all_events_drained": True, "hardware_visibility_proven_by_model": False}


def _math_interval(phase, index):
    begin = next(i for i, st in enumerate(phase.body)
                 if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
                 and st.target.id.startswith(f"stage_{index}_"))
    end = next(i for i in range(begin, len(phase.body))
               if isinstance(phase.body[i], ast.Expr)
               and isinstance(phase.body[i].value, ast.Call)
               and ast.unparse(phase.body[i].value.func) == "pl.system.fence")
    return begin, end


def transform(source, layout):
    tree = ast.parse(source)
    header = "".join(source.splitlines(keepends=True)[:tree.body[0].lineno - 1])
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    aic, aiv = functions["csa_static_aic"], functions["csa_static_aiv"]
    parameter = next(p for p in aic.args.args if p.arg == "view_2")
    if "[[96, 16384], pl.FP32]" not in ast.unparse(parameter.annotation):
        raise ValueError("HC 流式候选目前仅适用于 T96，不将块形状套到其他档")
    selected = {}
    for kind, function, wanted in (("aic", aic, ("hc_pre_linear",)),
                                  ("aiv", aiv, ("hc_widen_rms", "hc_pre_linear_reduce"))):
        phases = [s for s in function.body if isinstance(s, ast.If)]
        batches = layout["batches"][kind]
        if len(phases) != len(batches):
            raise ValueError("阶段源码与 layout 不符")
        for kernel in wanted:
            hits = [i for i, batch in enumerate(batches) if batch["kernel"] == kernel]
            if len(hits) != 1:
                raise ValueError(f"HC 阶段不是唯一的：{kernel}")
            index = hits[0]
            selected[kernel] = (index, phases[index], batches[index])

    cube_index, cube_phase, cube_batch = selected["hc_pre_linear"]
    widen_index, widen_phase, widen_batch = selected["hc_widen_rms"]
    reduce_index, reduce_phase, reduce_batch = selected["hc_pre_linear_reduce"]
    if cube_index != 0 or widen_index != 1:
        raise ValueError("当前协议要求 AIC 第一阶段及 AIV 原 HC 入口")
    if sorted((m["core"], m["logical_block"]) for m in cube_batch["members"]) != [(i, i) for i in range(24)]:
        raise ValueError("HC AIC 的24个分片必须一组一个")
    if sorted(m["logical_block"] for m in widen_batch["members"]) != list(range(12)):
        raise ValueError("原加宽不是12个8行分片")
    if any(m["wait_for_core_progress"] for m in widen_batch["members"]):
        raise ValueError("原加宽有前驱，不能提前启动新供数")
    expected_waits = {str(m["core"]): m["dispatch_seq"] for m in widen_batch["members"]}
    if any(m["wait_for_core_progress"] != expected_waits for m in cube_batch["members"]):
        raise ValueError("原 AIC 等待包含其他依赖，不能改为仅等待组内 READY")
    cube_done = {str(m["core"]): m["dispatch_seq"] for m in cube_batch["members"]}
    if any(m["wait_for_core_progress"] != cube_done for m in reduce_batch["members"]):
        raise ValueError("归约阶段必须等全部24组 AIC 完成，才能读取全部 RMS 分片")
    if sorted(m["logical_block"] for m in reduce_batch["members"]) != list(range(6)):
        raise ValueError("归约阶段必须完整覆盖6个16行块")
    # temporary_5 是原8行尾部暂存；T96无尾行，其他阶段不可使用它。
    for function in (aic, aiv):
        for phase in [s for s in function.body if isinstance(s, ast.If)]:
            if phase is not widen_phase and any(isinstance(n, ast.Name) and n.id == "temporary_5"
                                               for n in ast.walk(phase)):
                raise ValueError("RMS临时区与其他阶段重叠")
    parameter = next(p for p in aiv.args.args if p.arg == "temporary_5")
    if "[[8, 16384], pl.FP32]" not in ast.unparse(parameter.annotation):
        raise ValueError("RMS临时区描述符改变")
    # 新输入供数在全部48颗AIV执行；原12颗AIV仍按原代际发布原完成标记。
    # inv_rms 改在 linear_reduce 写入，因此检查其两个消费者都经过 reduce。
    reduce_done = {str(m["core"]): m["dispatch_seq"] for m in reduce_batch["members"]}
    for index, batch in enumerate(layout["batches"]["aiv"]):
        if batch["kernel"] not in ("split_pre_post", "comb_sinkhorn"):
            continue
        for member in batch["members"]:
            for core, done in reduce_done.items():
                covered = member["dispatch_seq"] > done if str(member["core"]) == core else (
                    member["wait_for_core_progress"].get(core, 0) >= done)
                if not covered:
                    raise ValueError("inv_rms 消费者没有等待所有归约生产者")

    begin, end = _math_interval(cube_phase, cube_index)
    original_cube = ast.unparse(ast.Module(body=cube_phase.body[begin:end], type_ignores=[]))
    if "pl.pipeline(16, stage=2" not in original_cube or "[16, 256]" not in original_cube:
        raise ValueError("原投影不是16次K256累加")
    cube_source = '''
pl.system.set_ffts(temporary_50)
hc_row = block_0 // 4 * 16
hc_split = block_0 % 4
hc_acc = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
for hc_k in pl.range(8):
    hc_col = hc_split * 4096 + hc_k * 512
    hc_weight = pl.tile.load(arg_hc_attn_fn, [0, hc_col], [32, 512], [24, 512], target_memory=pl.Mem.Mat)
    hc_weight_t = pl.tile.transpose_view(hc_weight)
    pl.system.sync_wait(8 + hc_k % 2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
    hc_input = pl.tile.load(view_2, [hc_row, hc_col], [16, 512], [16, 512], target_memory=pl.Mem.Mat)
    pl.system.sync_set(10 + hc_k % 2, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIC)
    hc_left0 = pl.tile.extract(hc_input, 0, 0, [16, 256], target_memory=pl.Mem.Left)
    hc_right0 = pl.tile.extract(hc_weight_t, 0, 0, [256, 32], target_memory=pl.Mem.Right)
    hc_acc = pl.tile.matmul_acc(hc_acc, hc_left0, hc_right0, init_cond=(hc_k == 0))
    hc_left1 = pl.tile.extract(hc_input, 0, 256, [16, 256], target_memory=pl.Mem.Left)
    hc_right1 = pl.tile.extract(hc_weight_t, 256, 0, [256, 32], target_memory=pl.Mem.Right)
    hc_acc = pl.tile.matmul_acc(hc_acc, hc_left1, hc_right1, init_cond=False)
pl.tile.store(hc_acc, [hc_split * 96 + hc_row, 0], temporary_7)
'''
    cube_phase.body[begin:end] = ast.parse(cube_source).body
    waits = [s for s in cube_phase.body if isinstance(s, ast.For)
             and isinstance(s.target, ast.Name) and s.target.id == "dependency_0"]
    if len(waits) != 1:
        raise ValueError("没有找到唯一原整块等待")
    cube_phase.body.remove(waits[0])

    producer_source = '''
pl.system.set_ffts(temporary_50)
hc_stream_row = group_id // 4 * 16 + lane_id * 8
hc_stream_split = group_id % 4
for hc_stream_k in pl.range(8):
    if hc_stream_k >= 2:
        pl.system.sync_wait(10 + hc_stream_k % 2, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
    hc_stream_part = hc_stream_split * 8 + hc_stream_k
    hc_stream_col = hc_stream_part * 512
    hc_bf16 = pl.tile.load(view_3, [hc_stream_row, hc_stream_col], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
    hc_fp32 = pl.tile.cast(hc_bf16, target_type=pl.FP32, mode='round')
    pl.tile.store(hc_fp32, [hc_stream_row, hc_stream_col], view_2)
    hc_square = pl.tile.mul(hc_fp32, hc_fp32)
    hc_reduce_tmp = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    hc_square_rows = pl.tile.row_sum(hc_square, hc_reduce_tmp)
    hc_square_line = pl.tile.reshape(hc_square_rows, [1, 8])
    pl.tile.store(hc_square_line, [0, hc_stream_part * 96 + hc_stream_row], temporary_5)
    pl.system.sync_set(8 + hc_stream_k % 2, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
for hc_drain in pl.range(2):
    pl.system.sync_wait(10 + hc_drain, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
'''
    begin, end = _math_interval(widen_phase, widen_index)
    del widen_phase.body[begin:end]
    position = aiv.body.index(widen_phase)
    aiv.body[position:position] = ast.parse(producer_source).body

    reduction_source = f'''
for hc_half in pl.range(2):
    hc_rms_row = block_{reduce_index} * 16 + hc_half * 8
    hc_rms_sum = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
    for hc_rms_part in pl.pipeline(32, stage=2):
        hc_rms_value = pl.tile.load(temporary_5, [0, hc_rms_part * 96 + hc_rms_row], [1, 8], [1, 8], target_memory=pl.Mem.Vec)
        hc_rms_sum = pl.tile.add(hc_rms_sum, hc_rms_value)
    hc_mean = pl.tile.adds(pl.tile.muls(hc_rms_sum, 6.103515625e-05), 1e-06)
    hc_rsqrt_tmp = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    hc_rsqrt = pl.tile.rsqrt(hc_mean, hc_rsqrt_tmp)
    pl.tile.store(pl.tile.reshape(hc_rsqrt, [8, 1]), [hc_rms_row, 0], temporary_4)
'''
    _, end = _math_interval(reduce_phase, reduce_index)
    reduce_phase.body[end:end] = ast.parse(reduction_source).body

    # 当前PTOAS在pipeline里的动态event_id生成C++时出现作用域错误。
    # 两槽只有偶/奇两种情况，显式选择静态ID，保持同一通知与等待协议。
    class StaticEvents(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if not isinstance(call, ast.Call) or ast.unparse(call.func) not in (
                    "pl.system.sync_wait", "pl.system.sync_set"):
                return self.generic_visit(node)
            event = call.args[0]
            if (not isinstance(event, ast.BinOp) or not isinstance(event.op, ast.Add)
                    or not isinstance(event.left, ast.Constant) or event.left.value not in (8, 10)
                    or not any(isinstance(n, ast.Name) and n.id.startswith("hc_")
                               for n in ast.walk(event.right))):
                return self.generic_visit(node)
            left, right = copy.deepcopy(node), copy.deepcopy(node)
            left.value.args[0] = ast.Constant(event.left.value)
            right.value.args[0] = ast.Constant(event.left.value + 1)
            return ast.If(test=ast.Compare(copy.deepcopy(event.right), [ast.Eq()], [ast.Constant(0)]),
                          body=[left], orelse=[right])

    tree = StaticEvents().visit(tree)
    audit = {"scope": "T96/24组，实验候选，尚未获得真机正确性或性能结论",
             "cube_stage": cube_index, "widen_stage": widen_index, "reduce_stage": reduce_index,
             "aiv_producers": 48, "aic_consumers": 24, "k256_accumulation_order": list(range(0, 4096, 256)),
             "rms_partial_add_order": list(range(32)), "rms_scratch_bytes": 32 * 96 * 4,
             "workspace_allocations_added": 0, "new_scalar_gm_signals": 0,
             "ready_ids": [8, 9], "ack_ids": [10, 11], "protocol_model": protocol_check(),
             "static_event_id_branches": True,
             "manual_event_loops_use_range": True,
             "old_whole_widen_wait_replaced_by_paired_ffts": True,
             "original_completion_records_kept": True,
             "rms_consumers_wait_for_all_reduce_members": True}
    return header + ast.unparse(ast.fix_missing_locations(tree)) + "\n", audit
