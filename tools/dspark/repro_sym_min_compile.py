# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent CPU Dynamo compatibility probe; no plugin, fixtures or model weights."""

import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path

import torch

CAPACITY = 24


def legacy(value):
    return value[: torch.sym_min(value.shape[0], CAPACITY)] + 1


def bounded(value):
    return value[: min(value.shape[0], CAPACITY)] + 1


def probe():
    source = inspect.getsource(torch.sym_min)
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "sym_min_module": torch.sym_min.__module__,
        "sym_min_qualname": torch.sym_min.__qualname__,
        "sym_min_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "sym_min_source": source,
        "eager_shape_type": type(torch.ones(5, 8).shape[0]).__name__,
        "attempts": [],
    }
    for function in (legacy, bounded):
        for dynamic in (False, True):
            torch._dynamo.reset()
            graphs = []

            def backend(graph, examples, graphs=graphs):
                graphs.append({"code": graph.code, "argument_types": [type(x).__name__ for x in examples]})
                return graph.forward

            record = {"expression": function.__name__, "dynamic": dynamic, "graphs": graphs, "shapes": []}
            try:
                compiled = torch.compile(function, backend=backend, fullgraph=True, dynamic=dynamic)
                for size in (29, 5, 24, 30):
                    output = compiled(torch.ones(size, 8))
                    assert list(output.shape) == [min(size, CAPACITY), 8]
                    record["shapes"].append(list(output.shape))
                record["status"] = "passed"
            except Exception as error:
                # Preserve the intentional legacy failure as evidence. The
                # repaired expression must pass; no execution fallback occurs.
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
            result["attempts"].append(record)
    torch._dynamo.reset()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("python", "torch", "sym_min_module", "eager_shape_type")}))
    for row in result["attempts"]:
        print(row["expression"], "dynamic=" + str(row["dynamic"]), row["status"])
    if any(x["status"] != "passed" for x in result["attempts"] if x["expression"] == "bounded"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
