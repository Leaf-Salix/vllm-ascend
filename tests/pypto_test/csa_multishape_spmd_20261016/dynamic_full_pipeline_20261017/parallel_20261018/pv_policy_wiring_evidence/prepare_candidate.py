"""制作隔离的 PV 列 tile 策略接线候选；拒绝覆盖已冻结源码。

只复制正式动态包的 33 个 Python 文件，改策略和三层 constexpr 传递；
不把 B/S 绑定为常量、不改 GM 工作区或 AIV 协议。只写本实验目录，
不会编译、启动 NPU 或修改正式仓。后续 CPU 显式分别选择 128/256。
"""

import ast
import datetime
import difflib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
CANONICAL = Path("/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1")
SOURCE = CANONICAL / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def replace(text, before, after, expected=1):
    assert text.count(before) == expected, (before, text.count(before), expected)
    return text.replace(before, after)


def main():
    before, candidate = HERE / "source_before", HERE / "source_candidate"
    assert not before.exists() and not candidate.exists(), "禁止覆盖已有冻结候选"
    before.mkdir()
    candidate.mkdir()
    sources = sorted(SOURCE.glob("*.py"))
    assert len(sources) == 33
    for path in sources:
        shutil.copyfile(path, before / path.name)
        shutil.copyfile(path, candidate / path.name)

    path = candidate / "policy.py"
    source = path.read_text()
    source = replace(source, "from dataclasses import dataclass", "from dataclasses import dataclass")
    source = replace(source, "见统一优化日志。\n\"\"\"", "见统一优化日志。\n"
        "Attention PV列tile新增待采用区间候选：长历史且Indexer使用G6时选256，\n"
        "其余仍128；字段本身默认128，显式旧策略保持原行为。这个域对应既有\n"
        "长历史G6平衡单根编排及完整函数资源布局，不按七个B/S验收点写白名单。\n"
        "已有长/短域结果分化只支撑继续验证区间，不能声称已证明性能根因；\n"
        "完整长域补测和接线验证完成前，本文件仅为私有待采用候选。PV列tile\n"
        "只改变512列的独立分块，不改变K128归约、softmax块、动态Q/L或缓存协议。\n\"\"\"")
    source = replace(source, "    hc_consumer_reduce: bool = False\n", "    hc_consumer_reduce: bool = False\n"
        "    # 默认值保留手工构造旧策略的N128行为；select_policy单独表达待验证区间。\n"
        "    pv_cols: int = 128\n\n"
        "    def __post_init__(self):\n"
        "        # 只接受两种已设计的完整列分块；拒绝bool及不能整分512列的值，\n"
        "        # 避免错误配置悄悄漏写PV尾列。这里只检查Host策略，不读设备数据。\n"
        "        if type(self.pv_cols) is not int or self.pv_cols not in (128, 256):\n"
        "            raise ValueError(\"Attention PV列tile仅支持整数128或256\")\n")
    source = replace(source, '            "HC_CONSUMER_REDUCE": self.hc_consumer_reduce,\n',
        '            "HC_CONSUMER_REDUCE": self.hc_consumer_reduce,\n            "PV_N_TILE": self.pv_cols,\n')
    source = replace(source, "        ob_rows=ob_rows,\n", "        ob_rows=ob_rows,\n"
        "        # 暂定沿用长历史G6平衡单根的编排/资源域，而非以T或B白名单选值。\n"
        "        # 这是待完整实测采用的候选边界，不是已证实的性能因果解释。\n"
        "        pv_cols=256 if long_history and group == 6 else 128,\n")
    path.write_text(source)

    path = candidate / "entry.py"
    source = path.read_text()
    source = replace(source, "    HC_CONSUMER_REDUCE: pl.constexpr,\n):", "    HC_CONSUMER_REDUCE: pl.constexpr,\n"
        "    # 仅作编译策略；尾部默认兼容未传此项的旧controls，不增加运行时标量。\n"
        "    PV_N_TILE: pl.constexpr = 128,\n):", expected=2)
    source = replace(source, "            HC_ROW_DEPENDENCIES,\n        )", "            HC_ROW_DEPENDENCIES,\n"
        "            PV_N_TILE,\n        )")
    # vector无需知道Cube的列tile；只有root调用core_group时继续转传。
    anchor = "            HC_CONSUMER_REDUCE,\n        )"
    assert source.count(anchor) == 2
    offset = source.rfind(anchor)
    source = source[:offset] + source[offset:].replace(anchor,
        "            HC_CONSUMER_REDUCE,\n            PV_N_TILE,\n        )", 1)
    source = replace(source, "    QUERY_GROUP/QK_PANEL/SCORE_TILE等仅是数学与流水策略，不固定T/B/L。\n",
        "    QUERY_GROUP/QK_PANEL/SCORE_TILE等仅是数学与流水策略，不固定T/B/L。\n"
        "    PV_N_TILE只经root→core_group→AIC传递，决定PV独立列块为128/256；\n"
        "    默认128兼容旧controls。AIV读取完整512列及环槽、FFTS和动态轴均不变。\n")
    path.write_text(source)

    path = candidate / "cube_schedule.py"
    source = path.read_text()
    source = replace(source, "    HC_ROW_DEPENDENCIES: pl.constexpr,\n):", "    HC_ROW_DEPENDENCIES: pl.constexpr,\n"
        "    # 独立列分块由策略特化；不影响本核实际query循环或信号所有权。\n"
        "    PV_N_TILE: pl.constexpr = 128,\n):")
    source = replace(source, "        worker,\n        GROUPS,\n        2,\n    )\n    signal = publish_phase(signal, PHASE_ATTN_CUBE",
        "        worker,\n        GROUPS,\n        2,\n        PV_N_TILE,\n    )\n    signal = publish_phase(signal, PHASE_ATTN_CUBE")
    # 文件总览放在原docstring里，不能在版权前增加新的说明体。
    pos = source.index('"""', source.index('"""') + 3)
    source = source[:pos] + "\nAttention PV列tile由策略传入，只特化AIC的128/256列计算；环槽仍写全512列，\nAIV协议和动态工作量不变。此私有区间候选尚待实测采用，不声称已证明性能根因。\n" + source[pos:]
    path.write_text(source)

    path = candidate / "attention_stream_stages.py"
    source = path.read_text()
    source = replace(source, "PV_N_TILE = 128\n", "")
    source = replace(source, "    EVENT_BASE: pl.constexpr = 0,\n):", "    EVENT_BASE: pl.constexpr = 0,\n"
        "    PV_N_TILE: pl.constexpr = 128,\n):", expected=1)
    source = replace(source, "    KV进一次L1，QK用转置view，PV复用原L1行；PV N128/K128保留交替累加器。",
        "    KV进一次L1，QK用转置view，PV复用原L1行；PV N128或N256/K128保留交替累加器。\n"
        "    PV_N_TILE为编译策略，默认128；不固定实际T/B/Q/L，不新增GM工作区。\n"
        "    128/256均整除完整512列，只改变独立列分块数量，不改变K归约或softmax；\n"
        "    所有列写回后才发布原PV_READY，AIV仍以原两段256列消费，不感知列tile。\n"
        "    私有策略暂按长历史G6编排/资源域选择256，尚待完整实测采用，非根因结论。")
    source = replace(source, "                # PV沿用N128/K128，保留每128候选一次softmax和query流水顺序。",
        "                # PV只特化独立输出列：128为四块，256为两块；K始终128，\n"
        "                # 每个输出元素的点积、softmax边界和query推进顺序不变。\n"
        "                # 仍交替保留前后累加器，末块store后才发布原PV_READY。")
    path.write_text(source)

    changed = []
    patch = []
    for path in sorted(candidate.glob("*.py")):
        ast.parse(path.read_text())
        old = before / path.name
        if old.read_text() != path.read_text():
            changed.append(path.name)
            patch.extend(difflib.unified_diff(old.read_text().splitlines(True), path.read_text().splitlines(True),
                fromfile="source_before/" + path.name, tofile="source_candidate/" + path.name))
    assert changed == ["attention_stream_stages.py", "cube_schedule.py", "entry.py", "policy.py"]
    (HERE / "candidate.patch").write_text("".join(patch))
    controls = dict(QUERY_GROUP=6, QK_PANEL=64, SCORE_TILE=1024, PREFETCH=1, TO_L0=True,
        BALANCED=True, ROOTS=1, OA_COLS=128, OB_ROWS=96, OUTPUT_GROUP_PIPELINE=True,
        HC_ROW_DEPENDENCIES=True, ATTENTION_QUERY_METADATA=False, HC_CONSUMER_REDUCE=False)
    for cols in (128, 256):
        (HERE / f"controls_{cols}.json").write_text(json.dumps({**controls, "PV_N_TILE": cols}, indent=2) + "\n")
    (HERE / "source_manifest.json").write_text(json.dumps({
        "canonical": str(SOURCE), "baseline_git_head": "8d46159f392a4d621a63dd20e8f84a1fe982e9f5",
        "source_frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_file_count": len(sources), "changed_files": changed,
        "status": "私有待采用候选，未修改正式，CPU/设备结论另记",
        "files": [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(candidate.glob("*.py"))]
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
