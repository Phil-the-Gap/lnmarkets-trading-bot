# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
from typing import List, Dict, Any, Optional

from history import (
    fetch_ohlcs_v2,
    fetch_ohlcs_v2_window,
    now_ms,
    TF_SECONDS,
)

DATA_DIR = "data"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _csv_path(network: str, tf: str) -> str:
    _ensure_dir(DATA_DIR)
    return os.path.join(DATA_DIR, f"ohlcs_{network}_{tf}.csv")


def save_ohlcs_csv(path: str, ohlcs: List[Dict[str, Any]]) -> None:
    fieldnames = ["time", "open", "high", "low", "close"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in ohlcs:
            w.writerow({
                "time": int(c["time"]),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
            })


def load_ohlcs_csv(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                "time": int(row["time"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })
    out.sort(key=lambda x: x["time"])
    return out


def _tf_ms(tf: str) -> int:
    tf = tf.strip().lower()
    if tf not in TF_SECONDS:
        raise ValueError(f"Unknown timeframe: {tf}")
    return TF_SECONDS[tf] * 1000


def paginate_backward(network: str, tf: str, total: int, page_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Backfill: hole `total` Candles in Pages, rückwärts in die Vergangenheit.
    Nutzt fetch_ohlcs_v2(..., to_ms=...).
    """
    all_ohlcs: List[Dict[str, Any]] = []
    to_ms: Optional[int] = None

    while len(all_ohlcs) < total:
        remaining = total - len(all_ohlcs)
        limit = min(page_size, remaining)

        page = fetch_ohlcs_v2(network, tf=tf, limit=limit, to_ms=to_ms)
        if not page:
            break

        oldest = int(page[0]["time"])
        to_ms = oldest - 1

        seen = {c["time"] for c in all_ohlcs}
        for c in page:
            if c["time"] not in seen:
                all_ohlcs.append(c)

        all_ohlcs.sort(key=lambda x: x["time"])

        if len(page) < limit:
            break

    return all_ohlcs[-total:]


def paginate_forward(network: str, tf: str, from_ms: int, to_ms: int, page_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Incremental update: hole alle neuen Candles ab from_ms..to_ms in Pages.
    Wir schieben from_ms immer auf den letzten Candle + 1ms.
    """
    out: List[Dict[str, Any]] = []
    cur_from = int(from_ms)
    end = int(to_ms)

    while cur_from < end:
        page = fetch_ohlcs_v2_window(network, tf=tf, from_ms=cur_from, to_ms=end, limit=page_size)
        if not page:
            break

        # Dedup
        seen = {c["time"] for c in out}
        for c in page:
            if c["time"] not in seen:
                out.append(c)

        out.sort(key=lambda x: x["time"])

        # Wenn weniger als page_size, sind wir durch
        if len(page) < page_size:
            break

        # Sonst weiter vor: nächster from = letzter candle time + 1
        cur_from = int(out[-1]["time"]) + 1

    return out


def get_cached_or_fetch_incremental(
    network: str,
    tf: str,
    total: int,
    page_size: int = 1000,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    1) Wenn kein Cache oder force_refresh: backfill total und speichern.
    2) Wenn Cache vorhanden: incremental update (nur neue Candles), speichern.
    3) Wenn Cache danach immer noch < total: zusätzlich backfill und mergen.
    4) Return: letzte `total` Candles.
    """
    path = _csv_path(network, tf)
    tf_ms = _tf_ms(tf)

    if force_refresh or (not os.path.exists(path)):
        ohlcs = paginate_backward(network, tf=tf, total=total, page_size=page_size)
        save_ohlcs_csv(path, ohlcs)
        return ohlcs[-total:]

    # Cache laden
    cached = load_ohlcs_csv(path)
    if not cached:
        ohlcs = paginate_backward(network, tf=tf, total=total, page_size=page_size)
        save_ohlcs_csv(path, ohlcs)
        return ohlcs[-total:]

    last_time = int(cached[-1]["time"])
    # Start: nächster Candle-Slot nach dem letzten cached Candle (safe)
    from_ms = last_time + tf_ms
    to_ms = now_ms()

    # Neue Candles nachziehen
    newer = paginate_forward(network, tf=tf, from_ms=from_ms, to_ms=to_ms, page_size=page_size)

    if newer:
        existing_times = {c["time"] for c in cached}
        for c in newer:
            if c["time"] not in existing_times:
                cached.append(c)
        cached.sort(key=lambda x: x["time"])
        save_ohlcs_csv(path, cached)

    # Falls cache kleiner als total: backfill ältere
    if len(cached) < total:
        need = total - len(cached)
        oldest = int(cached[0]["time"])
        # backfill older candles ending just before oldest
        older = paginate_backward(network, tf=tf, total=need, page_size=page_size)
        merged = older + cached
        # dedup + sort
        seen = {}
        for c in merged:
            seen[int(c["time"])] = c
        merged = list(seen.values())
        merged.sort(key=lambda x: x["time"])
        save_ohlcs_csv(path, merged)
        cached = merged

    return cached[-total:]
