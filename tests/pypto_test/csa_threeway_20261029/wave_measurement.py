# SPDX-License-Identifier: Apache-2.0
"""仅新增内容hash观测副本，保留原冻结模型、调度、数学及金标。"""

import hashlib
import json
from pathlib import Path


def prepare_wave_runner(source, runner, audit):
    helper = source / "tests/pypto_test/csa_threeway_20261029"
    frozen = json.loads((source.parent / "manifest.json").read_text())["files"]
    receipt = {}

    def read_frozen(path):
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == frozen[str(path.relative_to(source))]
        return data.decode()

    def write_module(directory, prefix, text):
        digest = hashlib.sha256(text.encode()).hexdigest()
        name = prefix + "_" + digest[:12]
        path = directory / (name + ".py")
        if path.exists():
            assert path.read_text() == text
        else:
            with path.open("x") as stream:
                stream.write(text)
        receipt[prefix] = {"path": str(path), "sha256": digest}
        return name

    observer = read_frozen(helper / "observer.py")
    old = "metadata.decode.num_reqs_actual == 40"
    assert observer.count(old) == 1
    observer = observer.replace(old, "metadata.decode.num_reqs_actual == 8")
    observer = observer.replace(
        "physical FULL240 bucket, actual T recorded;", "actual B8, physical FULL240 bucket, actual T recorded;"
    )
    observer_module = write_module(helper, "wave_observer", observer)
    text = runner.read_text()
    if audit is not None:
        original = Path(audit["observer"])
        data = original.read_text()
        old = "from .observer import TimedObserver"
        assert data.count(old) == 1
        module = write_module(helper, "wave_audit", data.replace(old, f"from .{observer_module} import TimedObserver"))
        old = "csa_threeway_20261029." + original.stem + ".RuntimeAuditObserver"
        new = f"csa_threeway_20261029.{module}.RuntimeAuditObserver"
    else:
        old = "csa_threeway_20261029.observer.TimedObserver"
        new = f"csa_threeway_20261029.{observer_module}.TimedObserver"
    assert text.count(old) == 1
    text = text.replace(old, new)
    mixed = read_frozen(source / "tests/pypto_test/low_acceptance_20261001/mixed.py")
    replacements = (
        ("    records = []", "    records = []\n    generation_waves = []"),
        (
            "        outputs = generate_aligned_batch(llm, prompts, params)",
            "        begin = time.perf_counter()\n"
            "        outputs = generate_aligned_batch(llm, prompts, params)\n"
            "        generation_waves.append({'offset': offset, 'requests': len(chunk),\n"
            "            'elapsed_s': time.perf_counter() - begin,\n"
            "            'output_tokens': sum(len(o.outputs[0].token_ids) for o in outputs)})",
        ),
        (
            "'sampling': {'temperature':",
            "'generation_waves': generation_waves,\n            'sampling': {'temperature':",
        ),
    )
    for old, new in replacements:
        assert mixed.count(old) == 1
        mixed = mixed.replace(old, new)
    module = write_module(source / "tests/pypto_test/low_acceptance_20261001", "wave_mixed", mixed)
    old = "from low_acceptance_20261001.mixed import run_batches"
    assert text.count(old) == 1
    text = text.replace(old, f"from low_acceptance_20261001.{module} import run_batches")
    module = write_module(runner.parent, "run_wave", text)
    return runner.with_name(module + ".py"), receipt
