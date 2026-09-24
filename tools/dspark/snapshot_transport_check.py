# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-weight eight-worker Core MQ/RPC exercise with bounded shared-memory use."""

import argparse
import gc
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import resource
import shutil
import sys
import tarfile
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

# Load this stdlib-only shared implementation without importing the plugin's
# logging/platform initializer. Server workers use its normal installed path.
_spec = importlib.util.spec_from_file_location(
    "dspark_snapshot_files", Path(__file__).parents[2] / "vllm_ascend/diagnostics/dspark_snapshot_transport.py"
)
files = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(files)

RANKS = 8
CHUNK = 1024 * 1024
CHUNKS = 2
TIMEOUT = 60
RUN_SECONDS = 120
DEFAULT_CHUNK = 24 * 1024 * 1024


def core_api(core, adapter):
    if adapter:
        from tools.dspark.queue_source_adapter import load

        return load(core)
    from vllm.distributed.device_communicators import shm_broadcast
    from vllm.v1.executor import multiproc_executor

    for module, relative in (
        (shm_broadcast, "distributed/device_communicators/shm_broadcast.py"),
        (multiproc_executor, "v1/executor/multiproc_executor.py"),
    ):
        if Path(module.__file__).resolve() != (Path(core) / "vllm" / relative).resolve():
            raise ValueError("Wrong installed Core transport source")
    return shm_broadcast, multiproc_executor


def serialized(queue, status, payload):
    """Execute Core's actual enqueue serializer; intercept only the sink for sizing."""
    packets = []

    @contextmanager
    def acquire(_):
        yield bytearray(1)

    writer = NS(
        _is_writer=True,
        n_local_reader=1,
        n_remote_reader=0,
        buffer=NS(max_chunk_bytes=1),
        acquire_write=acquire,
        local_socket=NS(send_multipart=lambda values, **kwargs: packets.append([len(v) for v in values])),
        _spin_condition=NS(notify=lambda: None),
    )
    queue.MessageQueue.enqueue(writer, (status, payload))
    sizes = packets[0]
    threshold = 6 + sizes[0] + sum(n + 4 for n in sizes[1:])
    return dict(
        buffers=sizes,
        pickle_bytes=sizes[0],
        framed_bytes=3 + 4 * len(sizes) + sum(sizes),
        threshold_bytes=threshold,
        production_default_path="overflow_zmq" if threshold >= DEFAULT_CHUNK else "inline_shm",
    )


def utility_size(value):
    # Installed Core only: the second serialization leg is Msgpack, not SHM.
    from vllm.v1.engine import EngineCoreOutputs, UtilityOutput
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    output = EngineCoreOutputs(utility_output=UtilityOutput(call_id=1, result=UtilityResult(value)))
    buffers = MsgpackEncoder().encode_into(output, bytearray())
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(buffers)
    if decoded.utility_output.result.result != value:
        raise ValueError("Core/frontend utility serialization changed the snapshot")
    return dict(buffers=[len(b) for b in buffers], bytes=sum(len(b) for b in buffers), call_id=1)


def release(queue):
    """Close/unlink only this check's queues after all send/receive operations end."""
    names = []
    buffer = queue.buffer
    if buffer is not None:
        names.append(buffer.shared_memory.name)
        buffer.shared_memory.close()
        if buffer.is_creator:
            buffer.shared_memory.unlink()
            buffer.is_creator = False
    condition = queue._spin_condition
    sockets = [queue.local_socket, queue.remote_socket]
    if condition is not None:
        sockets += [
            getattr(condition, n, None) for n in ("local_notify_socket", "read_cancel_socket", "write_cancel_socket")
        ]
    contexts = {s.context for s in sockets if s is not None}
    for socket in sockets:
        if socket is not None:
            socket.close(linger=0)
    for context in contexts:
        context.term()
    return names


def worker(core, adapter, rank, rpc_handle, control, root):
    queue, executor = core_api(core, adapter)
    incoming = queue.MessageQueue.create_from_handle(queue.Handle(**rpc_handle), rank)
    outgoing = queue.MessageQueue(1, 1, max_chunk_bytes=CHUNK, max_chunks=CHUNKS)
    control.send(outgoing.export_handle())
    incoming.local_socket.setsockopt(queue.zmq.RCVTIMEO, TIMEOUT * 1000)
    outgoing.local_socket.setsockopt(queue.zmq.RCVTIMEO, TIMEOUT * 1000)
    incoming.wait_until_ready()
    outgoing.wait_until_ready()
    snapshot = json.loads((Path(root) / f"input-{rank}.json").read_text())
    metrics = []
    try:
        while True:
            method, args, kwargs, output_rank = incoming.dequeue(timeout=TIMEOUT)
            if method == "stop":
                break
            if method == "snapshot":
                value = snapshot
            elif method == "boundary":
                value = b"B" * kwargs["size"]
            elif method == "file":
                value = files.persist(root, kwargs["transfer"], kwargs["point"], rank, lambda: snapshot)
            else:
                raise ValueError(method)
            stats = serialized(queue, executor.WorkerProc.ResponseStatus.SUCCESS, value)
            if method == "file" and stats["framed_bytes"] > files.MAX_RECEIPT_BYTES:
                raise ValueError("File reply exceeded wire bound")
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metrics.append(
                dict(
                    method=method,
                    test_path="overflow_zmq" if stats["threshold_bytes"] >= CHUNK else "inline_shm",
                    max_rss_bytes=peak if sys.platform == "darwin" else peak * 1024,
                    **stats,
                )
            )
            executor.WorkerProc.enqueue_output(NS(worker_response_mq=outgoing), value)
    finally:
        release(incoming)
        release(outgoing)
        (Path(root) / f"worker-{rank}.json").write_text(json.dumps(metrics, indent=2))
        control.close()


def run(core, root, snapshots, adapter=False):
    """Actual spawn processes, SHM/ZMQ and Core collective_rpc, not mock responses."""
    deadline = time.monotonic() + RUN_SECONDS

    def remaining_seconds():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Transport check total deadline exceeded")
        return min(TIMEOUT, remaining)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if sorted(r["rank"] for r in snapshots) != list(range(RANKS)):
        raise ValueError("Need eight original rank snapshots")
    for r in snapshots:
        files.atomic_json(root / f"input-{r['rank']}.json", r)
    # Never try to reproduce ENOSPC in the machine's public tmpfs. Whole queues
    # reserve <17 MiB, and leave at least their size plus 16 MiB headroom.
    reservation = RANKS * CHUNKS * (CHUNK + 2) + 2 * (65536 + RANKS + 1)
    if Path("/dev/shm").exists():
        stat = os.statvfs("/dev/shm")
        free = stat.f_bavail * stat.f_frsize
        if free < 2 * reservation + 16 * 1024 * 1024:
            raise ValueError("Insufficient safe /dev/shm headroom; no queues allocated")
    else:
        free = None
    queue, executor = core_api(core, adapter)
    channel = queue.MessageQueue(RANKS, RANKS, max_chunk_bytes=65536, max_chunks=2)
    ctx = mp.get_context("spawn")
    processes, controls, replies = [], [], []
    names = [channel.buffer.shared_memory.name]
    report = dict(
        status="running",
        source_adapter=adapter,
        production_defaults_allocated=False,
        queue_chunk_bytes=CHUNK,
        queue_chunks=CHUNKS,
        reserved_bytes=reservation,
        shm_free_before=free,
        original_sigbus_cause="UNKNOWN",
        phases=[],
    )
    try:
        for rank in range(RANKS):
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=worker, args=(str(core), adapter, rank, vars(channel.export_handle()), child, str(root))
            )
            process.start()
            child.close()
            processes.append(process)
            controls.append(parent)
        for control in controls:
            if not control.poll(remaining_seconds()):
                raise TimeoutError("Queue handshake unavailable")
            reply = queue.MessageQueue.create_from_handle(control.recv(), 0)
            names.append(reply.buffer.shared_memory.name)
            replies.append(reply)
        channel.local_socket.setsockopt(queue.zmq.RCVTIMEO, int(remaining_seconds() * 1000))
        channel.wait_until_ready()
        for reply in replies:
            reply.local_socket.setsockopt(queue.zmq.RCVTIMEO, int(remaining_seconds() * 1000))
            reply.wait_until_ready()
        report["utility_serialization"] = (
            {"original": utility_size(snapshots)} if not adapter else {"status": "NOT_RUN_LOCAL_SOURCE_ADAPTER"}
        )
        parent = NS(is_failed=False, rpc_broadcast_mq=channel, response_mqs=replies, futures_queue=deque())
        point = snapshots[0]["cost_profile"]["measurements"][0]["point"]
        overhead = (
            serialized(queue, executor.WorkerProc.ResponseStatus.SUCCESS, b"B" * CHUNK)["threshold_bytes"] - CHUNK
        )
        edge = CHUNK - overhead
        report["boundary_payload_bytes"] = [edge - 1, edge, edge + 1]
        for repeat in range(3):
            actual = executor.MultiprocExecutor.collective_rpc(parent, "snapshot", timeout=remaining_seconds())
            if actual != snapshots:
                raise ValueError("Original snapshot RPC reconstruction mismatch")
            report["phases"].append(dict(kind="original_snapshot", repeat=repeat, ranks=RANKS))
            del actual
            for size in report["boundary_payload_bytes"]:
                values = executor.MultiprocExecutor.collective_rpc(
                    parent, "boundary", timeout=remaining_seconds(), kwargs=dict(size=size)
                )
                if any(value != b"B" * size for value in values):
                    raise ValueError("Boundary RPC corruption")
                report["phases"].append(dict(kind="boundary", bytes=size, repeat=repeat))
                del values
            transfer = files.prepare(root, point, RANKS)
            receipts = executor.MultiprocExecutor.collective_rpc(
                parent, "file", timeout=remaining_seconds(), kwargs=dict(transfer=transfer, point=point)
            )
            if not adapter:
                report["utility_serialization"]["file_receipts"] = utility_size(receipts)
            rebuilt = files.restore(root, transfer, point, receipts, RANKS)
            if rebuilt != snapshots or parent.futures_queue:
                raise ValueError("File transport reconstruction/future release failed")
            report["phases"].append(dict(kind="file_receipt", repeat=repeat, transfer=transfer, receipts=receipts))
            del rebuilt, receipts
        # Reading marks slots reusable but does not zero or decommit their bytes.
        # Capture this allocation property without filling public shared memory.
        report["consumed_slots_retain_data"] = any(any(q.buffer.shared_memory.buf[:CHUNK]) for q in replies)
        channel.enqueue(("stop", (), {}, None), timeout=remaining_seconds())
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        report["exitcodes"] = [p.exitcode for p in processes]
        if report["exitcodes"] != [0] * RANKS:
            raise ValueError("Transport worker did not exit normally")
        report["worker_metrics"] = [json.loads((root / f"worker-{r}.json").read_text()) for r in range(RANKS)]
        report["status"] = "PASSED_TRANSPORT_ONLY"
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        forced = [p for p in processes if p.is_alive()]
        report["forced_test_pids"] = [p.pid for p in forced]
        for process in forced:
            process.terminate()
        reap_deadline = time.monotonic() + 2
        for process in forced:
            process.join(max(0, reap_deadline - time.monotonic()))
        killed = [p for p in forced if p.is_alive()]
        for process in killed:
            process.kill()
        reap_deadline = time.monotonic() + 2
        for process in killed:
            process.join(max(0, reap_deadline - time.monotonic()))
        report["exitcodes"] = [p.exitcode for p in processes]
        for q in replies:
            release(q)
        release(channel)
        for control in controls:
            control.close()
        gc.collect()
        remaining = []
        for name in names:
            try:
                shm = queue.shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                continue
            shm.close()
            remaining.append(name)
        report["owned_shm_remaining"] = remaining
        if remaining:
            report.update(status="FAILED", error="Owned shared memory not released")
        report["queue_source_sha256"] = hashlib.sha256(
            (Path(core) / "vllm/distributed/device_communicators/shm_broadcast.py").read_bytes()
        ).hexdigest()
        (root / "transport-report.json").write_text(json.dumps(report, indent=2))
    if remaining:
        raise ValueError("Owned transport shared memory not released")
    return report


def preflight(core, archive, output):
    from tools.dspark.sigbus_system_evidence import capture

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    capture(output / "system-evidence.json")
    outer_sha = "ffe56dd4d67c6d1f45b1e3f6a7f2256bd37f078d9b42cdeb1f95e8043df9110f"
    inner_sha = "b0bdc8737e9a01d4d05a300e2e6ad47d2a0f34aaecf3da46fd1911018c8b6b8a"
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != outer_sha:
        raise ValueError("SIGBUS archive SHA mismatch")
    with tempfile.TemporaryDirectory(prefix="dspark-transport-input-") as tmp:
        embedded = Path(tmp) / "inner.tar.gz"
        with tarfile.open(archive) as bundle:
            member = bundle.getmember("dspark-batch-expansion.qsQcxCRh/model-evidence.tar.gz")
            with bundle.extractfile(member) as src, embedded.open("wb") as dest:
                shutil.copyfileobj(src, dest)
        if hashlib.sha256(embedded.read_bytes()).hexdigest() != inner_sha:
            raise ValueError("SIGBUS embedded archive SHA mismatch")
        key = "dspark-large-batch.3eW01ldi/b128/cost/runs/b128/ctx128-n96-t96-balanced.json"
        with tarfile.open(embedded) as bundle:
            data = bundle.extractfile(key).read()
        (output / "input-identity.json").write_text(
            json.dumps(
                dict(
                    archive=str(archive),
                    outer_sha256=outer_sha,
                    inner_sha256=inner_sha,
                    member=key,
                    snapshot_sha256=hashlib.sha256(data).hexdigest(),
                    scope=(
                        "JSON-reconstructed snapshot; original worker aliases and crash-time wire bytes unavailable"
                    ),
                ),
                indent=2,
            )
        )
        snapshots = json.loads(data)["ranks"]
    return run(core, output, snapshots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-adapter", action="store_true")
    args = parser.parse_args()
    run(args.core, args.output, json.loads(args.snapshot.read_text())["ranks"], args.source_adapter)


if __name__ == "__main__":
    main()
