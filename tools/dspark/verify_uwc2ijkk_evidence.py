# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify native saved-operator results from archive bytes, never log summaries alone."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from tools.dspark import operator_replay as replay
from tools.dspark.operator_slot_controls import raw_bytes, replace_slot
from tools.dspark.verify_fppa5l9e_evidence import read_archive
from tools.dspark.verify_zvqidthd_evidence import float32_control


def verify(path, capture_archive=None):
    """Check native outputs first, then recompute CPU controls and optional byte matches."""
    files, links = read_archive(path, "48bd6fb4e1048c9cf3f637eaae6090668698d7a6c9fdd5147493b2b1dc78f580", 59, 22214066)

    def load(name):
        assert torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(files[name])) == []
        return torch.load(io.BytesIO(files[name]), map_location="cpu", weights_only=True)

    def digest(name):
        return hashlib.sha256(files[name]).hexdigest()

    capture = json.loads(files["capture-runtime.json"])
    expected = {a["path"]: a["sha256"] for a in capture["artifacts"]}
    inputs = {}
    hashes = {
        1802: "ba628505667035fd9c146d43a12f0709a50863ee24af59bffa634b48716673fd",
        1803: "f065f77610d99505069c2077125a9bb8db1a2211af1200c05f8a16cd1d6001d8",
    }
    assert files["status.txt"].decode() == "MAIN_RC=0\nPLUGIN=3b638d07062014c61a19af28f886d365f32ca4f4\n"
    for epoch, wanted in hashes.items():
        name = f"inputs/rank-0-operator-{epoch}.pt"
        assert digest(name) == wanted and wanted in files["inputs.sha256"].decode()
        inputs[epoch] = replay.validate(load(name))
    statuses = {k: v.decode().strip() for k, v in files.items() if k.endswith(".pipestatus")}
    assert len(statuses) == 14 and set(statuses.values()) == {"0 0"}
    cases = []
    for mode in ("aclgraph-saved", "eager-saved", "aclgraph-regenerated"):
        for epoch in (1802, 1803):
            prefix = f"{mode}-{epoch}/"
            result = json.loads(files[prefix + "result.json"])
            values = load(prefix + "replay.pt")
            ref = load(prefix + "reference.pt")
            capsule = inputs[epoch]
            torch.testing.assert_close(ref, replay.reference(capsule), atol=1e-12, rtol=1e-12, equal_nan=True)
            assert result["identity"] == capsule["identity"] and result["capsule"]["sha256"] == hashes[epoch]
            assert result["mode"] + "-" + result["metadata"] == mode
            assert not result["performance_eligible"]
            assert len(values["outputs"]) == 3
            valid = capsule["identity"]["query_start_loc_cpu"][-1]
            for i, output in enumerate(values["outputs"]):
                assert torch.equal(
                    output[:valid].contiguous().view(torch.uint8),
                    capsule["values"]["output"][:valid].contiguous().view(torch.uint8),
                )
                actual = replay.compare_output(output, ref, valid)
                wanted = [[0, 1], [0, 3], [0, 7]] if epoch == 1803 else []
                assert actual["nan_row_heads"] == result["replay_comparisons"][i]["nan_row_heads"] == wanted
                assert actual["inf_row_heads"] == []
            metadata_equal = torch.equal(values["metadata"], capsule["values"]["metadata"])
            assert metadata_equal == ("regenerated" not in mode)
            artifacts = {a["path"]: a["sha256"] for a in result["runtime"]["artifacts"]}
            assert all(artifacts[k] == expected[k] for k in expected.keys() & artifacts.keys())
            required = [k for k in expected if "vllm_ascend_C." in k or k.endswith("/libcust_opapi.so")]
            assert len(required) == 2 and all(artifacts.get(k) == expected[k] for k in required)
            cases.append(
                {
                    "case": prefix[:-1],
                    "pipestatus": statuses[prefix[:-1] + ".pipestatus"],
                    "replay_sha256": digest(prefix + "replay.pt"),
                    "result_sha256": digest(prefix + "result.json"),
                    "metadata_equal_to_saved": metadata_equal,
                    "metadata_different_elements": int((values["metadata"] != capsule["values"]["metadata"]).sum()),
                    "outputs_bitwise_equal_to_capture_valid_rows": [True] * 3,
                    "nan_row_heads": wanted,
                    "comparison": actual,
                    "matching_artifacts": len(artifacts.keys() & expected.keys()),
                    "capture_artifacts_not_loaded": sorted(expected.keys() - artifacts.keys()),
                }
            )
    controls = []
    for base, donor in ((1802, None), (1803, None), (1803, 1802), (1802, 1803)):
        capsule, intervention = (
            (inputs[base], None) if donor is None else replace_slot(inputs[base], inputs[donor], 123, 31)
        )
        output = float32_control(capsule)
        valid = capsule["identity"]["query_start_loc_cpu"][-1]
        controls.append(
            {
                "base": base,
                "donor": donor,
                "intervention": intervention,
                "CPU_float32_nan_heads": output[:valid].isnan().any(-1).nonzero().tolist(),
                "NPU_counterfactual": "PENDING",
            }
        )
    matches = None
    if capture_archive is not None:
        from tools.dspark.verify_zvqidthd_evidence import ARCHIVE_SHA

        old, _ = read_archive(capture_archive, ARCHIVE_SHA, 208, 448443829)
        pages = inputs[1803]["values"]["page_ids"].flatten().tolist()
        pattern = raw_bytes(inputs[1803]["values"]["pages"][pages.index(123), 31])
        names = sorted(k for k in old if "-operator-" in k and k.endswith(".pt"))
        assert len(names) == 24
        matches = []
        for name in names:
            capsule = replay.validate(torch.load(io.BytesIO(old[name]), map_location="cpu", weights_only=True))
            for key, value in capsule["values"].items():
                if not isinstance(value, torch.Tensor):
                    continue
                data, start, offsets = raw_bytes(value), 0, []
                while (index := data.find(pattern, start)) != -1:
                    offsets.append(index)
                    start = index + 1
                if offsets:
                    matches.append({"file": name, "tensor": key, "contiguous_payload_byte_offsets": offsets})
        assert {x["tensor"] for x in matches} == {"pages", "output"}
    return {
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "members": 59,
        "expanded_bytes": 22214066,
        "links": links,
        "input_hashes": hashes,
        "pipestatus": statuses,
        "cases": cases,
        "CPU_single_slot_controls": controls,
        "all_24_capsule_exact_1024_byte_matches": matches,
        "capture_runtime": capture,
        "native_replay": "completed and reproduced",
        "writer": "UNKNOWN",
        "slot_counterfactual_NPU": "PENDING",
        "model_worker_natural_exit": "NOT_TESTED",
        "performance_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-archive", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.archive, args.capture_archive), indent=2) + "\n")
