# SPDX-License-Identifier: Apache-2.0
"""移植SPMD fad8ccb64的2048/2080物理页表容量合同，不改MPMD任务划分。"""

import difflib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NAME = "aicpu_v34_physical_plan_capacity"


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


def main():
    subprocess.run(["python3", str(HERE / "prepare.py"), NAME], check=True)
    experiment = HERE / "experiments" / NAME
    operator = experiment / "operator"
    file = operator / "attention_plan_stages.py"
    text = file.read_text()
    text = replace(text, "C<=2048、", "C<=PLAN_TABLE_COLUMNS、")
    text = replace(
        text,
        "    ENABLED: pl.constexpr,\n",
        "    ENABLED: pl.constexpr,\n    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,\n",
    )
    text = replace(text, "columns <= PHYSICAL_PLAN_COLUMNS", "columns <= PLAN_TABLE_COLUMNS")
    text = replace(text, "[1, PHYSICAL_PLAN_COLUMNS]", "[1, PLAN_TABLE_COLUMNS]")
    text = replace(
        text,
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n",
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n"
        "    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,\n",
        2,
    )
    text = replace(
        text,
        "physical_mode = physical_plan_supported(pl.tensor.dim(compressed_table, 1), "
        "compressed_pages, PLAN_PHYSICAL_ROWS)",
        "physical_mode = physical_plan_supported(\n"
        "        pl.tensor.dim(compressed_table, 1), compressed_pages, PLAN_PHYSICAL_ROWS, PLAN_TABLE_COLUMNS\n"
        "    )",
        2,
    )
    file.write_text(text)
    file = operator / "attention_stream_stages.py"
    text = file.read_text()
    text = replace(text, "    ORIGINAL_COLUMNS_DYN,\n", "    ORIGINAL_COLUMNS_DYN,\n    PHYSICAL_PLAN_COLUMNS,\n")
    text = replace(
        text,
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n",
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n"
        "    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,\n",
    )
    text = replace(text, "        PLAN_PHYSICAL_ROWS,\n", "        PLAN_PHYSICAL_ROWS,\n        PLAN_TABLE_COLUMNS,\n")
    file.write_text(text)
    file = operator / "entry.py"
    text = file.read_text()
    text = replace(
        text,
        "from .attention_plan_stages import (\n",
        "from .attention_plan_stages import (\n    PHYSICAL_PLAN_COLUMNS,\n",
    )
    text = replace(
        text,
        "    PLAN_PHYSICAL_ROWS: pl.constexpr,\n",
        "    PLAN_PHYSICAL_ROWS: pl.constexpr,\n    PLAN_TABLE_COLUMNS: pl.constexpr,\n",
        2,
    )
    text = replace(
        text,
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n",
        "    PLAN_PHYSICAL_ROWS: pl.constexpr = False,\n"
        "    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,\n",
    )
    text, count = re.subn(r"(?m)^( +)PLAN_PHYSICAL_ROWS,$", r"\1PLAN_PHYSICAL_ROWS,\n\1PLAN_TABLE_COLUMNS,", text)
    assert count == 5, count
    file.write_text(text)
    file = operator / "policy.py"
    text = file.read_text()
    text = replace(
        text,
        "    bsh_projection_order: bool = False\n",
        "    bsh_projection_order: bool = False\n    plan_table_columns: int = 2048\n",
    )
    text = replace(
        text,
        '            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,\n',
        '            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,\n'
        '            "PLAN_TABLE_COLUMNS": self.plan_table_columns,\n',
    )
    text = replace(
        text,
        "def select_policy(*, tokens: int, requests: int, max_query_tokens: int, max_seq_len: int) -> CSAMPMDPolicy:",
        "def select_policy(\n    *, tokens: int, requests: int, max_query_tokens: int, max_seq_len: int,\n"
        "    compressed_table_columns: int | None = None,\n) -> CSAMPMDPolicy:",
    )
    text = replace(
        text,
        "    values = (tokens, requests, max_query_tokens, max_seq_len)\n",
        "    if compressed_table_columns is not None:\n"
        "        if type(compressed_table_columns) is not int or compressed_table_columns <= 0:\n"
        '            raise ValueError("compressed_table_columns须为已分配页表的正整数列数")\n'
        "    values = (tokens, requests, max_query_tokens, max_seq_len)\n",
    )
    text = replace(
        text,
        "        plan_physical_rows=tokens >= EARLY_GEOMETRY_MIN_TOKENS,\n",
        "        plan_physical_rows=tokens >= EARLY_GEOMETRY_MIN_TOKENS,\n"
        "        # 只依据已有Tensor形状；超过2080仍走原逻辑地址回退，不截断页表。\n"
        "        plan_table_columns=(\n            2080\n"
        "            if tokens >= EARLY_GEOMETRY_MIN_TOKENS\n"
        "            and compressed_table_columns is not None\n"
        "            and 2048 < compressed_table_columns <= 2080\n"
        "            else 2048\n        ),\n",
    )
    file.write_text(text)
    file = operator / "binding.py"
    text = replace(
        file.read_text(),
        "        self.policy = policy or select_policy(\n"
        "            tokens=t, requests=b, max_query_tokens=max_query_tokens, max_seq_len=max_seq_len\n        )",
        '        if native["cmp_block_table"].ndim != 2:\n'
        '            raise ValueError("cmp_block_table须为二维Tensor")\n'
        "        self.policy = policy or select_policy(\n"
        "            tokens=t, requests=b, max_query_tokens=max_query_tokens, max_seq_len=max_seq_len,\n"
        '            compressed_table_columns=native["cmp_block_table"].shape[1],\n        )',
    )
    file.write_text(text)
    file = operator / "service.py"
    text = replace(
        file.read_text(),
        "        # The shared service supplies this SPMD-only UB capacity descriptor.\n"
        "        # MPMD keeps its existing task resource policy and kernel arguments.\n"
        "        return select_policy(**descriptors)",
        "        # 复用服务已有页表形状描述符，不读取设备Q/L或页表内容。\n"
        "        return select_policy(compressed_table_columns=compressed_table_columns, **descriptors)",
    )
    file.write_text(text)
    # CPU单独编译2048/2080分支；该参数禁止用于NPU，实际运行必须从Native Tensor取形状。
    file = experiment / "run_case.py"
    text = replace(
        file.read_text(),
        '    parser.add_argument("--compile", action="store_true")\n',
        '    parser.add_argument("--compile", action="store_true")\n'
        '    parser.add_argument("--compile-table-columns", type=int, default=None)\n',
    )
    text = replace(
        text,
        "    args = parser.parse_args()\n",
        "    args = parser.parse_args()\n"
        "    if args.compile_table_columns is not None and not args.cpu:\n"
        '        parser.error("--compile-table-columns仅用于CPU编译；设备始终读取真实Tensor形状")\n',
    )
    text = replace(
        text,
        '                max_seq_len=max(case["seq_lens"]),\n            )\n'
        "            module.decode_csa_tnd._get_dep_graph()",
        '                max_seq_len=max(case["seq_lens"]),\n'
        "                compressed_table_columns=args.compile_table_columns,\n"
        '            )\n            report["cpu_table_columns"] = args.compile_table_columns\n'
        "            module.decode_csa_tnd._get_dep_graph()",
    )
    file.write_text(text)
    changes = [
        "attention_plan_stages.py",
        "attention_stream_stages.py",
        "entry.py",
        "policy.py",
        "binding.py",
        "service.py",
    ]
    patches = []
    for name in changes:
        relative = f"vllm_ascend/ops/pypto/deepseek_v4_flash_csa_tnd/{name}"
        patches.extend(
            difflib.unified_diff(
                (REPO / relative).read_text().splitlines(True),
                (operator / name).read_text().splitlines(True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    (HERE / "scheduling_v34_physical_capacity.patch").write_text("".join(patches))
    manifest = json.loads((experiment / "manifest.json").read_text())
    manifest.update(
        {
            "candidate": NAME,
            "reference_commit_for_capacity": "fad8ccb64",
            "changes": changes,
            "new_tasks": False,
            "no_device_readback": True,
            "frozen": False,
        }
    )
    (experiment / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(experiment)


if __name__ == "__main__":
    main()
