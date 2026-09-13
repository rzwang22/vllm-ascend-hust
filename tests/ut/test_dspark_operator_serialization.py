# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restricted capsule loading must not depend on Ascend Format pickle globals."""

import json
import subprocess
import sys
from enum import Enum, IntEnum
from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.test_dspark_operator_capture import MODULE, ROOT, fixture
from tools.dspark import operator_replay as replay


class Format(IntEnum):
    ND = 2
    FRACTAL_NZ = 29
    UNKNOWN = 999


class Label(Enum):
    SELECTED = "selected"


class NPUTensorDescriptor:
    """Exercise descriptor's backend branch using real CPU tensor attributes."""

    device = NS(type="npu")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


@pytest.mark.parametrize("code", [None, 2, -1, 999, Format.ND, Format.FRACTAL_NZ, Format.UNKNOWN])
def test_descriptor_normalizes_format_before_serialization(tmp_path, monkeypatch, code):
    monkeypatch.setitem(sys.modules, "torch_npu", NS(get_npu_format=lambda _: code))
    value = MODULE.descriptor(NPUTensorDescriptor(torch.ones(2)))
    assert value["npu_format"] is None if code is None else type(value["npu_format"]) is int
    assert value["npu_format"] == code
    path = tmp_path / "descriptor.pt"
    torch.save(value, path)
    assert torch.serialization.get_unsafe_globals_in_checkpoint(path) == []
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded == value
    assert loaded["npu_format"] is None if code is None else type(loaded["npu_format"]) is int


def test_legacy_enum_file_reproduces_restricted_load_failure(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"npu_format": Format.FRACTAL_NZ}, path)
    assert any(name.endswith(".Format") for name in torch.serialization.get_unsafe_globals_in_checkpoint(path))
    with pytest.raises(Exception, match="Unsupported global") as failure:
        torch.load(path, map_location="cpu", weights_only=True)
    assert type(failure.value).__name__ == "UnpicklingError"


@pytest.mark.parametrize("code", [None, 29, Format.FRACTAL_NZ])
def test_actual_capsule_restricted_load_in_fresh_cpu_replay_process(tmp_path, monkeypatch, code):
    original = MODULE.descriptor
    monkeypatch.setitem(sys.modules, "torch_npu", NS(get_npu_format=lambda _: code))
    monkeypatch.setattr(MODULE, "descriptor", lambda tensor: original(NPUTensorDescriptor(tensor)))
    f = fixture(tmp_path, monkeypatch)
    # Other metadata paths may contain enums too; do not persist their classes.
    f.capture.options["point"] = Label.SELECTED.value
    f.capture.options["test_label"] = Label.SELECTED
    capsule, _ = f.run(7)
    for layout in capsule["layouts"].values():
        assert layout["npu_format"] is None if code is None else type(layout["npu_format"]) is int
        assert layout["npu_format"] == code
    assert type(capsule["options"]["test_label"]) is str
    path = tmp_path / "rank-0-operator-7.pt"
    assert torch.serialization.get_unsafe_globals_in_checkpoint(path) == []
    # Exercise the formal CLI's map_location='cpu', weights_only=True load in a
    # fresh process. No fixture, Format class or safe-global registration there.
    script = """
import json, runpy, sys, torch
before = list(torch.serialization.get_safe_globals())
sys.argv = [sys.argv[1], sys.argv[2], '--mode', 'reference', '--output', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
assert torch.serialization.get_safe_globals() == before
print('CHILD_STATE=' + json.dumps({'torch_npu_imported': 'torch_npu' in sys.modules}))
"""
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(ROOT / "tools/dspark/operator_replay.py"),
            str(path),
            str(tmp_path / "reference"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads((tmp_path / "reference/result.json").read_text())
    assert result["identity"]["execution"] == 7 and result["reference_nonfinite_rows"] == []
    # A CPU-only local interpreter proves no torch_npu import is needed. On an
    # installed backend host Torch may auto-import it; record that distinction.
    print(proc.stdout)


@pytest.mark.parametrize("value", [object(), NS(format=2), torch.device("cpu")])
def test_unknown_backend_metadata_rejected_before_file_is_published(tmp_path, monkeypatch, value):
    f = fixture(tmp_path, monkeypatch)
    f.capture.options["unsupported"] = value
    with pytest.raises(TypeError, match="Unsupported operator capsule metadata type"):
        f.run(1)
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize(
    "expected,actual", [(None, None), (None, Format.ND), (29, 29), (29, Format.FRACTAL_NZ), (999, Format.UNKNOWN)]
)
def test_replay_compares_format_numbers(expected, actual):
    replay.check_npu_format(expected, actual)


@pytest.mark.parametrize("expected,actual", [(29, Format.ND), (999, 2), (29, None)])
def test_replay_does_not_replace_unknown_or_unsupported_format(expected, actual):
    with pytest.raises(ValueError, match="Original NPU format unavailable"):
        replay.check_npu_format(expected, actual)
