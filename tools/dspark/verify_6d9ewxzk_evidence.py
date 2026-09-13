# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify native bidirectional slot controls from actual archived tensor bytes."""

import argparse
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from tools.dspark import operator_replay as replay
from tools.dspark.operator_slot_controls import raw_bytes, replace_slot, slot_hash
from tools.dspark.verify_fppa5l9e_evidence import read_archive


def verify(path):
    """Recompute interventions, native output masks and watch histories from restricted archive loads."""
    files, links = read_archive(path, "83c47d3330329ba0a7fe83b5a36bbb513ad9e2a5f4357b313322788174d1a742", 67, 39002064)

    def load(name):
        assert not torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(files[name]))
        return torch.load(io.BytesIO(files[name]), map_location="cpu", weights_only=True)

    originals = {n: replay.validate(load(f"inputs/rank-0-operator-{n}.pt")) for n in (1802, 1803)}
    hashes = {
        1802: "ba628505667035fd9c146d43a12f0709a50863ee24af59bffa634b48716673fd",
        1803: "f065f77610d99505069c2077125a9bb8db1a2211af1200c05f8a16cd1d6001d8",
    }
    for n, digest in hashes.items():
        assert hashlib.sha256(files[f"inputs/rank-0-operator-{n}.pt"]).hexdigest() == digest
    assert files["status.txt"].decode() == "MAIN_RC=0\nPLUGIN=d90993191fb4df6c1f2761c575acaab77d85b226\n"
    status = {k: v.decode().strip() for k, v in files.items() if k.endswith(".pipestatus")}
    assert len(status) == 12 and set(status.values()) == {"0 0"}
    native = load("watch-test/test_npu_watch_observes_each_g0/capture-semantics.pt")
    assert native["stages"][0]["counter"].item() == 0 and not native["stages"][0]["snapshot_valid"]
    assert native["replays_completed"] == 3 and "error" not in native
    for s in native["stages"][1:]:
        assert s["values"].flatten().tolist() == s["expected"]
    preflight = load("watch-test/test_npu_watch_observes_each_g0/native-watch-evidence.pt")
    assert len(preflight["watch"]["snapshots"]) == 4
    for shot, expected in zip(preflight["watch"]["snapshots"], preflight["expectations"]["checks"]):
        assert shot["values"].eq(expected["expected_guard"]).all()
        assert shot["completed_call"] == expected["completed_call"]
    junit = ET.fromstring(files["watch.xml"]).findall(".//testcase")
    assert len(junit) == 1 and not any(junit[0].find(k) is not None for k in ("error", "failure", "skipped"))
    runtime = json.loads(files["capture-runtime.json"])
    artifacts = {a["path"]: a["sha256"] for a in runtime["artifacts"]}
    cases = []
    for case, n, donor, heads in [
        ("original-1802", 1802, None, []),
        ("original-1803", 1803, None, [1, 3, 7]),
        ("1803-from-1802", 1803, 1802, []),
        ("1802-from-1803", 1802, 1803, [1, 3, 4, 6]),
    ]:
        capsule = originals[n]
        intervention = None
        if donor:
            expected, intervention = replace_slot(capsule, originals[donor], 123, 31)
            derived = replay.validate(load(case + "/counterfactual.pt"))
            for key, value in expected["values"].items():
                assert raw_bytes(value) == raw_bytes(derived["values"][key])
            for key in capsule.keys() - {"values"}:
                assert capsule[key] == derived[key]
            a, b = raw_bytes(capsule["values"]["pages"]), raw_bytes(derived["values"]["pages"])
            changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            assert len(changed) == 977 and all(97280 <= i < 98304 for i in changed)
            stored = json.loads(files[case + "/intervention.json"])
            for key, value in intervention.items():
                assert stored[key] == value
            assert stored["derived_file"]["sha256"] == hashlib.sha256(files[case + "/counterfactual.pt"]).hexdigest()
            capsule = derived
        outputs = load(case + "/replay.pt")
        assert torch.equal(outputs["metadata"], capsule["values"]["metadata"])
        result = json.loads(files[case + "/result.json"])
        assert result["mode"] == "aclgraph" and result["metadata"] == "saved"
        identity = result["runtime"]
        assert identity["torch"] == "2.10.0+cpu" and identity["torch_npu"] == "2.10.0.post2"
        loaded = {a["path"]: a["sha256"] for a in identity["artifacts"]}
        common = sorted(artifacts.keys() & loaded.keys())
        assert common and all(artifacts[k] == loaded[k] for k in common)
        comparisons = []
        for out in outputs["outputs"]:
            c = replay.compare_output(out, replay.reference(capsule), 11)
            assert c["nan_row_heads"] == [[0, h] for h in heads] and not c["inf_row_heads"]
            if case == "1803-from-1802":
                assert c["output_finite_abs_max"] == 12.625
            comparisons.append(c)
        assert len(comparisons) == 3
        watch = load(case + "/slot-watch.pt")
        assert len(watch) == 4
        expected_slot = capsule["values"]["pages"][2, 31]
        for pair in watch:
            assert raw_bytes(pair["values"][0]) == raw_bytes(pair["values"][1]) == raw_bytes(expected_slot)
        state = json.loads(files[case + "/slot-watch.json"])
        assert state["replays_completed"] == 3 and state["completed_calls"] == 4
        assert not state["stages"][1]["snapshot_valid"]
        cases.append(
            {
                "case": case,
                "matching_runtime_artifacts": common,
                "capture_only_artifacts": sorted(artifacts.keys() - loaded.keys()),
                "intervention": intervention,
                "comparisons": comparisons,
                "watch_pairs_unchanged": 4,
                "slot_sha256": slot_hash(expected_slot),
            }
        )
    return {
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "links_not_followed": links,
        "input_hashes": hashes,
        "pipestatus": status,
        "cases": cases,
        "runtime_versions": {k: runtime[k] for k in ("python", "torch", "torch_npu")},
        "preflight": "1 passed; actual guard snapshots and capture/replay counts checked",
        "native_slot_causality": "VERIFIED",
        "actual_writer": "UNKNOWN",
        "worker_natural_exit": "NOT_TESTED",
        "performance_eligible": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(verify(a.archive), indent=2) + "\n")
