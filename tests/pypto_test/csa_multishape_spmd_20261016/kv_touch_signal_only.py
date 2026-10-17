# SPDX-License-Identifier: Apache-2.0
"""保留KV写回汇总通知，删除原样搬运和仅为该搬运服务的payload屏障。

原TMR通过1行KV自拷贝建立WAR依赖。单SPMD中等待表已经显式保存写回关系，
但汇总通知仍有用：后继只等一个汇总者，避免展开全部写回成员。
此处保留等待表、代际、信号DCCI/DSB、生产者发布与QK读前维护。
只移除零业务计算阶段的Tile load/store及其前后fence/cacheinvalid/fence。
"""

import ast


def signal_only(source, layout):
    batches = layout["batches"]["aiv"]
    matches = [(i, b) for i, b in enumerate(batches) if b["kernel"] == "kv_touch"]
    if len(matches) != 1 or len(matches[0][1]["members"]) != 1:
        raise ValueError("KV标记必须是唯一单成员")
    index, batch = matches[0]
    member = batch["members"][0]
    if not member["publish_completion"] or not member["wait_for_core_progress"]:
        raise ValueError("KV标记不是原写回汇总发布点")
    module = ast.parse(source)
    original = ast.dump(module)
    fn = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)
              and n.name == "csa_static_aiv")
    phase = next(n for n in fn.body if isinstance(n, ast.If)
                 and ast.unparse(n.test) == f"mapped_block_{index} >= 0")
    before = phase.body
    if len(before) != 15:
        raise ValueError("KV阶段包装结构改变")
    for offset in (6, 11):
        if [ast.unparse(n) for n in before[offset:offset + 3]] != [
                "pl.system.fence()", "pl.system.cacheinvalid()", "pl.system.fence()"]:
            raise ValueError("KV阶段业务缓存边界改变")
    load, store = before[9:11]
    if (not isinstance(load, ast.AnnAssign) or not isinstance(store, ast.Assign)
            or ast.unparse(load.value.func) != "pl.tile.load"
            or ast.unparse(store.value.func) != "pl.tile.store"
            or [ast.unparse(x) for x in load.value.args[1:]] != ["[0, 0]", "[1, 512]", "[1, 512]"]
            or [ast.unparse(x) for x in store.value.args[:2]] != [load.target.id, "[0, 0]"]
            or ast.dump(store.value.args[2]) != ast.dump(load.value.args[0])
            or "pl.BF16" not in ast.unparse(load.annotation)):
        raise ValueError("KV阶段不是严格的1行BF16原样读写")
    if (not isinstance(before[5], ast.For)
            or "csa_signal_wait(" not in ast.unparse(before[5])
            or not isinstance(before[-1], ast.If)
            or "csa_signal_notify(" not in ast.unparse(before[-1])):
        raise ValueError("KV信号等待/通知边界改变")
    phase.body = before[:6] + before[-1:]
    ast.fix_missing_locations(module)
    result = ast.unparse(module) + "\n"
    phase.body = before
    if ast.dump(module) != original:
        raise ValueError("修改超出了KV标记的业务操作范围")
    return result, {"stage": index, "core": member["core"], "task": member["task"],
                    "identity_bytes_removed": 1024, "pipeline_fences_removed": 4,
                    "whole_scalar_cache_operations_removed": 2,
                    "wait_loop_and_notification_exactly_preserved": True,
                    "wait_table_and_schedule_unchanged": True,
                    "all_other_math_and_cache_operations_unchanged": True,
                    "producer_publish_and_qk_read_maintenance_unchanged": True,
                    "signal_dcci_dsb_unchanged": True, "device_validated": False}
