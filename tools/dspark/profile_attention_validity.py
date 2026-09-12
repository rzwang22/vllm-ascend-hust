# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read worker-owned first FULL receipts during generation, without an RPC."""

import json
from pathlib import Path

MAX_UNVALIDATED_TOKENS = 64
REQUIRED_ROUNDS = 3


class AttentionValidity:
    def __init__(self, directory, ranks):
        self.directory = Path(directory)
        self.ranks = ranks
        self.point = None
        self.passed = False

    def check(self, point, *, tokens=0, finished=False):
        if self.passed:
            return
        if self.point is None:
            self.point = point
        if point != self.point:
            raise RuntimeError("Attention validity did not pass in the first profile point")
        statuses = []
        for rank in range(self.ranks):
            path = self.directory / f"rank-{rank}-attention-validity.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())  # worker uses atomic replace
            if data.get("rank") != rank or data.get("point") != point or data.get("performance_eligible") is not False:
                raise RuntimeError("Attention validity receipt identity mismatch")
            if data.get("status") == "failed":
                raise RuntimeError(f"Attention diagnostic invalid on rank {rank}: {data.get('error')}")
            rounds = data.get("rounds", [])
            if data.get("status") == "passed":
                epochs = [r["execution"] for r in rounds]
                if len(epochs) != REQUIRED_ROUNDS or epochs != list(range(epochs[0], epochs[0] + REQUIRED_ROUNDS)):
                    raise RuntimeError("Attention validity lacks consecutive FULL execution receipts")
                if not all(r.get("valid") is True for r in rounds):
                    raise RuntimeError("Attention validity contains an invalid round")
                names = data.get("required_boundaries", [])
                if len([n for n in names if ".attention." in n]) != 12:
                    raise RuntimeError("Attention validity missing required boundary names")
                for row in rounds:
                    raw = row.get("raw_receipts", [])
                    if (
                        row.get("target_receipts") != [row["execution"]] * len(names)
                        or not raw
                        or raw != [row["execution"]] * len(raw)
                        or row.get("consume_receipts") != raw
                    ):
                        raise RuntimeError("Attention validity contains stale/missing device receipts")
                statuses.append(rank)
        self.passed = len(statuses) == self.ranks
        if not self.passed and (finished or tokens >= MAX_UNVALIDATED_TOKENS):
            raise RuntimeError(
                f"Attention validity unavailable before {MAX_UNVALIDATED_TOKENS} output tokens: ranks {statuses}"
            )
