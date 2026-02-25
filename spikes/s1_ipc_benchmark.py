"""
Spike S1: IPC Round-Trip Latency Benchmark
===========================================
Benchmarks Unix Domain Socket (UDS) + shared memory IPC between
a master and worker process.

Two modes:
  1. UDS-only baseline: payload sent entirely over the socket
  2. UDS + SHM: payload written to shared memory, descriptor sent via UDS

Usage:
    uv run python spikes/s1_ipc_benchmark.py
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.shared_memory as shm
import os
import socket
import struct
import sys
import tempfile
import time
import msgpack

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAYLOAD_SIZES = [
    (1024, "1 KB"),
    (16 * 1024, "16 KB"),
    (64 * 1024, "64 KB"),
    (256 * 1024, "256 KB"),
    (1024 * 1024, "1 MB"),
    (4 * 1024 * 1024, "4 MB"),
]

ITERATIONS = 1000
WARMUP = 50

# Length-prefix: 4-byte big-endian unsigned int
HDR_FMT = "!I"
HDR_SIZE = struct.calcsize(HDR_FMT)


# ---------------------------------------------------------------------------
# Helpers — length-prefixed send / recv over UDS
# ---------------------------------------------------------------------------

def send_msg(sock: socket.socket, data: bytes) -> None:
    """Send a length-prefixed message."""
    sock.sendall(struct.pack(HDR_FMT, len(data)))
    sock.sendall(data)


def recv_msg(sock: socket.socket) -> bytes:
    """Receive a length-prefixed message."""
    hdr = _recv_exact(sock, HDR_SIZE)
    length = struct.unpack(HDR_FMT, hdr)[0]
    return _recv_exact(sock, length)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed prematurely")
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Mode 1: UDS-only (baseline) — payload travels over the socket
# ---------------------------------------------------------------------------

def _worker_uds_only(sock_path: str, ready_event: multiprocessing.Event) -> None:  # type: ignore[type-arg]
    """Worker: connect, receive payload over UDS, send ACK."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Wait for the server to be ready
    ready_event.wait()
    client.connect(sock_path)

    try:
        while True:
            try:
                msg_bytes = recv_msg(client)
            except ConnectionError:
                break
            msg = msgpack.unpackb(msg_bytes, raw=False)
            if msg.get("cmd") == "stop":
                break

            # msg contains the full payload
            _payload = msg["payload"]

            # Send ACK
            ack = msgpack.packb({"ack": True})
            send_msg(client, ack)
    finally:
        client.close()


def bench_uds_only(payload_size: int, iterations: int) -> list[float]:
    """Master side: send payload over UDS, wait for ACK, measure RTT."""
    sock_path = os.path.join(tempfile.gettempdir(), f"nerva_bench_uds_{os.getpid()}.sock")
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)

    ready_event = multiprocessing.Event()
    worker = multiprocessing.Process(
        target=_worker_uds_only, args=(sock_path, ready_event)
    )
    worker.start()
    ready_event.set()

    conn, _ = server.accept()
    payload = os.urandom(payload_size)

    latencies: list[float] = []
    try:
        for i in range(WARMUP + iterations):
            msg = msgpack.packb({"cmd": "data", "payload": payload})
            t0 = time.perf_counter_ns()
            send_msg(conn, msg)
            _ack = recv_msg(conn)
            t1 = time.perf_counter_ns()
            if i >= WARMUP:
                latencies.append((t1 - t0) / 1000.0)  # ns -> us

        # Tell worker to stop
        stop_msg = msgpack.packb({"cmd": "stop"})
        send_msg(conn, stop_msg)
    finally:
        conn.close()
        server.close()
        worker.join(timeout=5)
        os.unlink(sock_path)

    return latencies


# ---------------------------------------------------------------------------
# Mode 2: UDS + Shared Memory — payload in SHM, descriptor over UDS
# ---------------------------------------------------------------------------

def _worker_shm(sock_path: str, ready_event: multiprocessing.Event) -> None:  # type: ignore[type-arg]
    """Worker: connect, read SHM descriptor from UDS, read payload from SHM, ACK."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ready_event.wait()
    client.connect(sock_path)

    attached: dict[str, shm.SharedMemory] = {}
    try:
        while True:
            try:
                msg_bytes = recv_msg(client)
            except ConnectionError:
                break
            msg = msgpack.unpackb(msg_bytes, raw=False)
            if msg.get("cmd") == "stop":
                break

            shm_name: str = msg["shm_name"]
            size: int = msg["size"]

            # Attach to SHM (cache handle to avoid repeated open/close overhead)
            if shm_name not in attached:
                attached[shm_name] = shm.SharedMemory(name=shm_name, create=False)
            sm = attached[shm_name]

            # Read the payload (force a copy so we actually touch the memory)
            _payload = bytes(sm.buf[:size])

            # Send ACK
            ack = msgpack.packb({"ack": True})
            send_msg(client, ack)
    finally:
        for sm in attached.values():
            sm.close()
        client.close()


def bench_shm(payload_size: int, iterations: int) -> list[float]:
    """Master side: write payload to SHM, send descriptor via UDS, wait for ACK."""
    sock_path = os.path.join(tempfile.gettempdir(), f"nerva_bench_shm_{os.getpid()}.sock")
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)

    ready_event = multiprocessing.Event()
    worker = multiprocessing.Process(
        target=_worker_shm, args=(sock_path, ready_event)
    )
    worker.start()
    ready_event.set()

    conn, _ = server.accept()
    payload = os.urandom(payload_size)

    # Allocate shared memory once, reuse across iterations
    sm = shm.SharedMemory(create=True, size=payload_size)

    latencies: list[float] = []
    try:
        for i in range(WARMUP + iterations):
            # Write payload into SHM
            sm.buf[:payload_size] = payload

            descriptor = msgpack.packb({
                "cmd": "data",
                "shm_name": sm.name,
                "size": payload_size,
            })
            t0 = time.perf_counter_ns()
            send_msg(conn, descriptor)
            _ack = recv_msg(conn)
            t1 = time.perf_counter_ns()
            if i >= WARMUP:
                latencies.append((t1 - t0) / 1000.0)  # ns -> us

        # Stop worker
        stop_msg = msgpack.packb({"cmd": "stop"})
        send_msg(conn, stop_msg)
    finally:
        conn.close()
        server.close()
        worker.join(timeout=5)
        sm.close()
        sm.unlink()
        os.unlink(sock_path)

    return latencies


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute percentile using linear interpolation (same as numpy default)."""
    n = len(sorted_data)
    k = (pct / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def report(label: str, size_label: str, latencies: list[float]) -> dict[str, float]:
    sorted_lats = sorted(latencies)
    p50 = _percentile(sorted_lats, 50)
    p95 = _percentile(sorted_lats, 95)
    p99 = _percentile(sorted_lats, 99)
    mean = sum(latencies) / len(latencies)
    print(f"  {label:20s} | {size_label:>8s} | "
          f"p50={p50:10.1f} us | p95={p95:10.1f} us | p99={p99:10.1f} us | "
          f"mean={mean:10.1f} us")
    return {"p50": p50, "p95": p95, "p99": p99, "mean": mean}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print(f"Nerva Spike S1: IPC Round-Trip Latency Benchmark")
    print(f"  iterations = {ITERATIONS}, warmup = {WARMUP}")
    print(f"  platform   = {sys.platform}, pid = {os.getpid()}")
    print("=" * 100)
    print()

    header = f"  {'Mode':20s} | {'Size':>8s} | {'p50':>13s} | {'p95':>13s} | {'p99':>13s} | {'mean':>13s}"
    sep = "-" * len(header)

    # --- UDS-only baseline ---
    print(">>> UDS-only (payload over socket) — baseline")
    print(header)
    print(sep)
    for size, label in PAYLOAD_SIZES:
        lats = bench_uds_only(size, ITERATIONS)
        report("UDS-only", label, lats)
    print()

    # --- UDS + SHM ---
    print(">>> UDS + Shared Memory (descriptor over socket, payload in SHM)")
    print(header)
    print(sep)
    for size, label in PAYLOAD_SIZES:
        lats = bench_shm(size, ITERATIONS)
        report("UDS+SHM", label, lats)
    print()

    print("Done.")


if __name__ == "__main__":
    main()
