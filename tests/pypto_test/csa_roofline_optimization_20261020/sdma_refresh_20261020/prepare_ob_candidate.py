# SPDX-License-Identifier: Apache-2.0
"""从本轮冻结完整单 SPMD 生成 O_B 预取候选，不更改正式算子或已有数学阶段。"""

import json
from pathlib import Path
import shutil


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"冻结源锚点不唯一：{old!r}")
    return text.replace(old, new, 1)


def main():
    here = Path(__file__).resolve().parent
    source = here.parent / "source_reference"
    target = here / "source_ob_prefetch"
    if target.exists():
        raise RuntimeError("候选目录已存在，拒绝覆盖")
    target.mkdir()
    for item in source.glob("*.py"):
        shutil.copy2(item, target / item.name)
    entry_path = target / "entry.py"
    text = entry_path.read_text()
    text = replace_once(
        text,
        "def _csa_core_group(\n    x: pl.Tensor[[T, 16384], pl.BF16],\n",
        "def _csa_core_group(\n    x: pl.Tensor[[T, 16384], pl.BF16],\n"
        "    wo_b_prefetch: pl.Tensor[[33554432], pl.INT8],\n",
    )
    text = replace_once(
        text, ") = vector_schedule(\n            x,\n",
        ") = vector_schedule(\n            x,\n            wo_b_prefetch,\n",
    )
    text = replace_once(
        text, "    with pl.spmd(GROUPS, sync_start=True):\n",
        "    # 仅建立原 NZ O_B 权重的完整物理字节视图，不分配、不复制，也不读取设备值。\n"
        "    # 向 AIV 传递内部派生描述符；外部根入参仍只有原 wo_b。\n"
        "    ob_prefetch_flat = pl.reshape(wo_b, [33554432])\n"
        "    with pl.spmd(GROUPS, sync_start=True):\n",
    )
    text = replace_once(
        text, ") = _csa_core_group(\n            x,\n",
        ") = _csa_core_group(\n            x,\n            ob_prefetch_flat,\n",
    )
    entry_path.write_text(text)

    path = target / "vector_schedule.py"
    text = path.read_text()
    text = replace_once(
        text, "def vector_schedule(\n    x: pl.Tensor[[T, 16384], pl.BF16],\n",
        "def vector_schedule(\n    x: pl.Tensor[[T, 16384], pl.BF16],\n"
        "    wo_b_prefetch: pl.Tensor[[33554432], pl.INT8],\n",
    )
    text = replace_once(
        text,
        '"""完整动态CSA的双AIV编排：先读整体数据流，再沿阶段注释阅读。\n',
        '"""完整动态CSA的双AIV编排：先读整体数据流，再沿阶段注释阅读。\n\n'
        "本候选仅增加输出权重预取：先完成原 Attention 并发布 packed，再由尾部\n"
        "四条 AIV 各提交 8 MiB 的只读 O_B NZ 权重。AIC 可继续原 O_A 计算；\n"
        "发起者等自己事件完成后恢复原量化和 HCpost，不改变 Attention 数学。\n"
        "新区间使用已有 BALANCED 且动态 T>=64，不按 B16/B28 等单点选择。\n",
    )
    old = "    signal = publish_phase(signal, PHASE_ATTN_VECTOR, core, CORES, generation)\n"
    block = old + """    if BALANCED and tokens >= 64 and worker >= VECTORS - 4:
        # 先发布原 packed 完成，使 AIC 可以沿原依赖进入 O_A；预取不成为它的新前驱。
        # 四个发起者来自尾部两组的双 AIV，通道由新版 runtime 的驻留身份隔离。
        # 4×8 MiB 完整覆盖 32 MiB 权重，无重叠、无越界、不触碰可写 KV/state。
        # 其余 44 条 AIV 继续原量化；发起者完成等待后仍承担自己的全部计算。
        # 新版编译器为 context 全函数保留 256B UB，不能手写 UB 地址来避让。
        ob_regions = pl.tensor.view(wo_b_prefetch, shape=[4, 8388608])
        ob_region = pl.tensor.slice(
            ob_regions, [1, 8388608], [worker - (VECTORS - 4), 0], gm_view=True
        )
        ob_context = pl.prefetch.make_context()
        ob_event = pl.prefetch.async_prefetch(ob_region, ob_context)
        ob_session = pl.prefetch.session(ob_context)
        if not pl.prefetch.wait(ob_event, ob_session):
            # 候选诊断：失败在本核独占隔离区留下标记，原保护检查必须据此失败。
            # 不把超时视为取消或成功；测试遇此标记后终止，不继续重放该工作区。
            # 预取源为整层只读权重，关闭 runtime 前保持存活。
            pl.write(signal, [core * 128 + 127], pl.const(-1, pl.INT32))
            pl.system.cacheinvalid(signal, [1], [core * 128 + 127])
            pl.system.fence()
"""
    text = replace_once(text, old, block)
    path.write_text(text)
    changed = [p.name for p in source.glob("*.py") if p.read_bytes() != (target / p.name).read_bytes()]
    assert set(changed) == {"entry.py", "vector_schedule.py"}
    report = {
        "source": str(source), "candidate": str(target), "changed": changed,
        "root_arguments_added": 0, "incore_readonly_views_added": 1,
        "math_changed": False, "new_cross_core_dependencies": 0,
        "prefetch_total_bytes": 33554432, "issuers": 4,
        "functional_passed": False, "performance_measured": False,
    }
    (here / "OB_CANDIDATE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
