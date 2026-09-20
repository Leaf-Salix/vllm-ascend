# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in swimlane collection for live Qwen3 PyPTO execution."""

import json
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs


class _SwimlaneController:
    def __init__(self) -> None:
        self.level = envs.VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_LEVEL
        self.root = envs.VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_DIR.strip()
        self.limit = envs.VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_MAX_CAPTURES
        if not 0 <= self.level <= 4:
            raise ValueError("VLLM_ASCEND_PYPTO_QWEN3_SWIMLANE_LEVEL must be in 0..4")
        if self.level and (not self.root or self.limit < 1):
            raise ValueError("Qwen3 PyPTO swimlane requires a nonempty SWIMLANE_DIR and positive SWIMLANE_MAX_CAPTURES")
        self.output_dir: Path | None = None
        self.initialized = False
        self.captures = 0
        self.warmed_signatures: set[tuple[Any, ...]] = set()
        self.lock = threading.Lock()

    def initialize(self, pypto_init: Callable[..., None]) -> None:
        if self.initialized:
            return
        if not self.level:
            pypto_init()
        else:
            root = Path(self.root).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            self.output_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"worker_{os.getpid()}_device{torch.npu.current_device()}_",
                    dir=root,
                )
            )
            pypto_init(
                enable_chip_swimlane=self.level,
                enable_dep_gen=True,
                output_dir=self.output_dir,
            )
            logger.info("Qwen3 PyPTO swimlane enabled: %s", self.output_dir)
        self.initialized = True

    @contextmanager
    def eager(self, name: str, signature: tuple[Any, ...]) -> Iterator[None]:
        if not self.level or not self.initialized or torch.npu.is_current_stream_capturing():
            yield
            return
        with self.lock:
            if self.captures >= self.limit:
                yield
                return
            if signature not in self.warmed_signatures:
                yield
                self.warmed_signatures.add(signature)
                return
            with self._collect({"mode": "eager", "scope": name, "signature": signature}):
                yield

    @contextmanager
    def replay(self, graph_name: str) -> Iterator[None]:
        if not self.level or not self.initialized or torch.npu.is_current_stream_capturing():
            yield
            return
        with self.lock:
            if self.captures >= self.limit:
                yield
                return
            with self._collect({"mode": "graph_replay", "graph": graph_name}):
                yield

    @contextmanager
    def _collect(self, metadata: dict[str, Any]) -> Iterator[None]:
        from pypto.torch import begin_dfx, end_dfx

        begin_dfx()
        try:
            yield
        finally:
            end_dfx()
            window = self.captures
            self.captures += 1
        assert self.output_dir is not None
        output = self.output_dir if window == 0 else self.output_dir / f"window_{window}"
        self._export(output, metadata)

    def _export(self, output: Path, metadata: dict[str, Any]) -> None:
        records = output / "chip_swimlane_records.json"
        deps = output / "deps.json"
        if not records.is_file() or not records.stat().st_size:
            raise RuntimeError(f"Qwen3 PyPTO swimlane did not produce a nonempty artifact: {records}")
        record_payload = json.loads(records.read_text())
        metadata = {
            **metadata,
            "pid": os.getpid(),
            "level": self.level,
            "includes_dependency_collection_overhead": True,
        }
        if metadata["mode"] == "graph_replay" and not record_payload.get("aicore_tasks"):
            metadata["status"] = "no_pypto_tasks"
            (output / "capture.json").write_text(json.dumps(metadata, indent=2))
            logger.warning("Qwen3 replay contains no recorded PyPTO tasks: %s", output)
            return
        if not deps.is_file() or not deps.stat().st_size:
            raise RuntimeError(f"Qwen3 PyPTO swimlane did not produce a nonempty artifact: {deps}")
        (output / "capture.json").write_text(json.dumps(metadata, indent=2))
        merged = output / "merged_swimlane.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "simpler_setup.tools.swimlane_converter",
                str(records),
                "--deps-json",
                str(deps),
                "-o",
                str(merged),
            ],
            check=True,
            timeout=60,
        )
        events = json.loads(merged.read_text()).get("traceEvents", []) if merged.is_file() else []
        if not any(event.get("ph") == "X" and "taskId" in event.get("args", {}) for event in events):
            raise RuntimeError(f"Qwen3 PyPTO swimlane conversion did not produce a nonempty trace: {merged}")
        logger.info("Qwen3 PyPTO %s swimlane: %s", metadata["mode"], merged)


_CONTROLLER: _SwimlaneController | None = None


def _controller() -> _SwimlaneController:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = _SwimlaneController()
    return _CONTROLLER


def init_pypto(pypto_init: Callable[..., None]) -> None:
    """Initialize the process kernel context with immutable DFX settings."""
    _controller().initialize(pypto_init)


@contextmanager
def eager_model_swimlane(
    name: str,
    signature: tuple[Any, ...],
    *,
    graph_enabled: bool,
) -> Iterator[None]:
    """Collect a warmed live model forward without re-running it."""
    if graph_enabled:
        yield
        return
    with _controller().eager(name, signature):
        yield


@contextmanager
def graph_replay_swimlane(graph_name: str) -> Iterator[None]:
    """Collect a live replay without entering the Python model again."""
    with _controller().replay(graph_name):
        yield
