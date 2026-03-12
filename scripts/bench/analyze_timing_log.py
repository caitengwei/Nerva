"""Analyze Nerva per-request timing logs.

Reads JSON timing logs produced when NERVA_TIMING_LOG_DIR is set,
computes per-component latency stats (p50/p95/p99/mean), and prints a table.

Usage:
    # Analyze all *.log files in a directory
    uv run python scripts/bench/analyze_timing_log.py --log-dir /tmp/nerva_timing

    # Analyze specific files
    uv run python scripts/bench/analyze_timing_log.py \\
        --log-files /tmp/nerva_timing/nerva_proxy_mm_preprocess.log \\
                    /tmp/nerva_timing/nerva_worker_12345.log

Events parsed:
    ipc_timing   — written by WorkerProxy; fields: request_id, model, ipc_round_trip_ms
    infer_timing — written by Worker;      fields: request_id, model, worker_deser_ms, backend_infer_ms
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from typing import Any


def _parse_events(
    files: list[str],
) -> tuple[dict[str, Any], dict[str, list[Any]], list[Any], list[Any]]:
    """Return (ipc_events, infer_events, executor_events, rpc_events)."""
    ipc_events: dict[str, Any] = {}
    infer_events: dict[str, list[Any]] = defaultdict(list)
    executor_events: list[Any] = []
    rpc_events: list[Any] = []

    for path in files:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event = data.get("event")
                    request_id = data.get("request_id", "")
                    if event == "ipc_timing":
                        ipc_events[request_id] = data
                    elif event == "infer_timing":
                        infer_events[request_id].append(data)
                    elif event == "executor_timing":
                        executor_events.append(data)
                    elif event == "rpc_timing":
                        rpc_events.append(data)
        except OSError as exc:
            print(f"  Warning: cannot read {path}: {exc}")

    return ipc_events, dict(infer_events), executor_events, rpc_events


def _pct(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return s[idx]


def _print_row(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label:<32s}  (no data)")
        return
    print(
        f"  {label:<32s}"
        f"  n={len(values):>5d}"
        f"  p50={_pct(values, 50):>8.3f} ms"
        f"  p95={_pct(values, 95):>8.3f} ms"
        f"  p99={_pct(values, 99):>8.3f} ms"
        f"  mean={statistics.mean(values):>8.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Nerva per-request timing logs")
    parser.add_argument("--log-dir", help="Directory containing *.log timing files")
    parser.add_argument("--log-files", nargs="+", metavar="FILE", help="Specific log files")
    args = parser.parse_args()

    files: list[str] = []
    if args.log_dir:
        files.extend(sorted(glob.glob(f"{args.log_dir}/*.log")))
    if args.log_files:
        files.extend(args.log_files)

    if not files:
        print("No log files found. Pass --log-dir or --log-files.")
        return

    print(f"Parsing {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    ipc_events, infer_events, executor_events, rpc_events = _parse_events(files)
    print(f"\nParsed {len(ipc_events)} ipc_timing, "
          f"{sum(len(v) for v in infer_events.values())} infer_timing, "
          f"{len(executor_events)} executor_timing, "
          f"{len(rpc_events)} rpc_timing events\n")

    # --- Aggregate metrics -------------------------------------------------
    ipc_rtt: list[float] = []
    proxy_ser: list[float] = []
    proxy_des: list[float] = []
    deser_all: list[float] = []
    infer_all: list[float] = []
    deser_by_model: dict[str, list[float]] = defaultdict(list)
    infer_by_model: dict[str, list[float]] = defaultdict(list)
    ipc_transport_by_model: dict[str, list[float]] = defaultdict(list)

    for ev in ipc_events.values():
        model = str(ev.get("model", "unknown"))
        rtt = ev.get("ipc_round_trip_ms")
        if isinstance(rtt, (int, float)):
            ipc_rtt.append(float(rtt))
        s = ev.get("proxy_serialize_ms")
        if isinstance(s, (int, float)):
            proxy_ser.append(float(s))
        d = ev.get("proxy_deserialize_ms")
        if isinstance(d, (int, float)):
            proxy_des.append(float(d))

    for events in infer_events.values():
        for ev in events:
            model = str(ev.get("model", "unknown"))
            d = ev.get("worker_deser_ms")
            i = ev.get("backend_infer_ms")
            if isinstance(d, (int, float)):
                deser_all.append(float(d))
                deser_by_model[model].append(float(d))
            if isinstance(i, (int, float)):
                infer_all.append(float(i))
                infer_by_model[model].append(float(i))

    # Per-model IPC transport = ipc_round_trip - worker_deser - backend_infer
    for rid, ipc_ev in ipc_events.items():
        model = str(ipc_ev.get("model", "unknown"))
        rtt = ipc_ev.get("ipc_round_trip_ms")
        infer_ev_list = infer_events.get(rid, [])
        matched = next((e for e in infer_ev_list if e.get("model") == model), None)
        if matched and isinstance(rtt, (int, float)):
            transport = float(rtt) - float(matched.get("worker_deser_ms", 0)) \
                        - float(matched.get("backend_infer_ms", 0))
            ipc_transport_by_model[model].append(transport)

    # Executor and RPC aggregates
    exec_overhead: list[float] = []
    exec_total: list[float] = []
    rpc_body: list[float] = []
    rpc_parse: list[float] = []
    rpc_execute: list[float] = []
    rpc_serialize: list[float] = []
    rpc_overhead: list[float] = []

    for ev in executor_events:
        v = ev.get("scheduler_overhead_ms")
        t = ev.get("total_execute_ms")
        if isinstance(v, (int, float)):
            exec_overhead.append(float(v))
        if isinstance(t, (int, float)):
            exec_total.append(float(t))

    for ev in rpc_events:
        for key, lst in [
            ("rpc_body_read_ms", rpc_body),
            ("rpc_parse_ms", rpc_parse),
            ("rpc_execute_ms", rpc_execute),
            ("rpc_serialize_ms", rpc_serialize),
            ("rpc_overhead_ms", rpc_overhead),
        ]:
            v = ev.get(key)
            if isinstance(v, (int, float)):
                lst.append(float(v))

    # --- Print summary table -----------------------------------------------
    print("=" * 80)
    print("Worker-side breakdown (per stage)")
    print("=" * 80)
    _print_row("worker_deser_ms   (all models)", deser_all)
    _print_row("backend_infer_ms  (all models)", infer_all)

    all_models = sorted(set(list(deser_by_model.keys()) + list(infer_by_model.keys())))
    if all_models:
        print()
        print("=" * 80)
        print("Per-model breakdown")
        print("=" * 80)
        for model in all_models:
            print(f"\n  [{model}]")
            _print_row("  worker_deser_ms", deser_by_model.get(model, []))
            _print_row("  backend_infer_ms", infer_by_model.get(model, []))
            _print_row("  ipc_transport_ms  (ZMQ round-trip overhead)", ipc_transport_by_model.get(model, []))

    print()
    print("=" * 80)
    print("Proxy-side serialization (per stage)")
    print("=" * 80)
    _print_row("proxy_serialize_ms   (msgpack encode inputs)", proxy_ser)
    _print_row("proxy_deserialize_ms (msgpack decode outputs)", proxy_des)

    if exec_overhead or exec_total:
        print()
        print("=" * 80)
        print("Executor DAG scheduling overhead (per request)")
        print("=" * 80)
        _print_row("executor_total_execute_ms", exec_total)
        _print_row("executor_scheduler_overhead_ms", exec_overhead)

    if rpc_overhead:
        print()
        print("=" * 80)
        print("RPC / HTTP layer breakdown (per request)")
        print("=" * 80)
        _print_row("rpc_body_read_ms    (await request.body())", rpc_body)
        _print_row("rpc_parse_ms        (frame parse + msgpack decode)", rpc_parse)
        _print_row("rpc_execute_ms      (executor.execute())", rpc_execute)
        _print_row("rpc_serialize_ms    (frame encode + msgpack encode)", rpc_serialize)
        _print_row("rpc_overhead_ms     (body+parse+serialize, excl. execute)", rpc_overhead)


if __name__ == "__main__":
    main()
