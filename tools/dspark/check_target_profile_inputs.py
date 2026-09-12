# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the small provenance fields actually available in the prior archive."""

import argparse
import hashlib
import json
from pathlib import Path

AUDIT_PATH = Path(__file__).with_name("PROFILE_f9kqffNA_AUDIT.json")
CHECKPOINT_FIELDS = ("config_sha256", "index_sha256", "checkpoint_weight_sha256")


def check(checkpoint_path, manifest_path):
    audit = json.loads(AUDIT_PATH.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    for name in CHECKPOINT_FIELDS:
        if checkpoint[name] != audit["checkpoint"][name]:
            raise ValueError(f"Target diagnostic checkpoint provenance changed: {name}")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != audit["input_manifest_sha256"]:
        raise ValueError("Target diagnostic input manifest changed")
    return {"status": "matched_archived_fields", "full_target_weight_hashes": "UNAVAILABLE_IN_PRIOR_ARCHIVE"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.checkpoint, args.manifest)))


if __name__ == "__main__":
    main()
