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


def _parse_events(files: list[str]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Return (ipc_events, infer_events) keyed by request_id."""
    ipc_events: dict[str, Any] = {}
    infer_events: dict[str, list[Any]] = defaultdict(list)

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
        except OSError as exc:
            print(f"  Warning: cannot read {path}: {exc}")

    return ipc_events, dict(infer_events)


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

    ipc_events, infer_events = _parse_events(files)
    print(f"\nParsed {len(ipc_events)} ipc_timing events, "
          f"{sum(len(v) for v in infer_events.values())} infer_timing events "
          f"({len(infer_events)} unique request_ids)\n")

    # --- Aggregate metrics -------------------------------------------------
    ipc_rtt: list[float] = []
    deser_all: list[float] = []
    infer_all: list[float] = []
    deser_by_model: dict[str, list[float]] = defaultdict(list)
    infer_by_model: dict[str, list[float]] = defaultdict(list)

    for ev in ipc_events.values():
        v = ev.get("ipc_round_trip_ms")
        if isinstance(v, (int, float)):
            ipc_rtt.append(float(v))

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

    # --- Print summary table -----------------------------------------------
    print("=" * 80)
    print("Overall latency breakdown")
    print("=" * 80)
    _print_row("ipc_round_trip_ms (WorkerProxy)", ipc_rtt)
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

    # --- Derived overhead estimate -----------------------------------------
    if ipc_rtt and infer_all:
        print()
        print("=" * 80)
        print("Derived estimates (p50)")
        print("=" * 80)
        ipc_p50 = _pct(ipc_rtt, 50)
        infer_p50 = _pct(infer_all, 50)
        # IPC network overhead = round-trip - backend_infer (approx, single stage)
        print(f"  ipc_round_trip_ms p50           = {ipc_p50:.3f} ms")
        print(f"  backend_infer_ms  p50           = {infer_p50:.3f} ms")
        print(f"  estimated IPC transport overhead = {ipc_p50 - infer_p50:.3f} ms"
              "  (round_trip - backend_infer, single-stage approximation)")


if __name__ == "__main__":
    main()
