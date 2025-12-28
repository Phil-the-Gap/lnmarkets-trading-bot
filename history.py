# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import requests



# /v2/futures/ohlcs erlaubt range: 1,3,5,10,15,30,45,60,120,180,240,... :contentReference[oaicite:2]{index=2}
TF_TO_V2_RANGE = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "10m": "10",
    "15m": "15",
    "30m": "30",
    "45m": "45",
    "1h": "60",
    "2h": "120",
    "3h": "180",
    "4h": "240",
}

TF_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "10m": 600,
    "15m": 900,
    "30m": 1800,
    "45m": 2700,
    "1h": 3600,
    "2h": 7200,
    "3h": 10800,
    "4h": 14400,
}

def base_url_for(network: str) -> str:
    return "https://api.lnmarkets.com" if network.strip().lower() == "mainnet" else "https://api.testnet4.lnmarkets.com"

def now_ms() -> int:
    return int(time.time() * 1000)

def fetch_ohlcs_v2(network: str, *, tf: str, limit: int, to_ms: int | None = None) -> List[Dict[str, Any]]:
    """
    Fetch last `limit` OHLC candles for the given timeframe from /v2/futures/ohlcs. :contentReference[oaicite:3]{index=3}
    Public endpoints are rate-limited, so call this sparingly. :contentReference[oaicite:4]{index=4}
    """
    tf = tf.strip().lower()
    if tf not in TF_TO_V2_RANGE:
        raise ValueError(f"Unsupported timeframe for v2 ohlcs: {tf}")

    if to_ms is None:
        to_ms = now_ms()

    # request enough history window so that we surely get `limit` candles
    secs = TF_SECONDS[tf]
    from_ms = to_ms - (limit + 10) * secs * 1000  # +10 buffer

    params = {
        "from": from_ms,
        "to": to_ms,
        "limit": min(max(limit, 1), 1000),
        "range": TF_TO_V2_RANGE[tf],
    }


    url = base_url_for(network) + "/v2/futures/ohlcs"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # sort by time ascending, just in case
    data.sort(key=lambda x: x["time"])
    return data

def now_ms() -> int:
    return int(time.time() * 1000)

def fetch_ohlcs_v2_paginated(
    network: str,
    *,
    tf: str,
    total_limit: int,
    end_ms: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Fetch more than 1000 OHLC candles by paginating backwards in time.
    Uses fetch_ohlcs_v2 internally.
    """
    all_candles: List[Dict[str, Any]] = []

    if end_ms is None:
        end_ms = now_ms()

    remaining = int(total_limit)

    while remaining > 0:
        batch_limit = min(remaining, 1000)

        batch = fetch_ohlcs_v2(
            network,
            tf=tf,
            limit=batch_limit,
            to_ms=end_ms,
        )

        if not batch:
            break

        all_candles.extend(batch)

        # ⏪ nächster Request: vor die älteste Kerze springen
        end_ms = batch[0]["time"] - 1
        remaining -= len(batch)

        # Abbruch, wenn API weniger geliefert hat als angefragt
        if len(batch) < batch_limit:
            break

        # kleine Pause gegen Rate-Limits
        time.sleep(0.2)

    # sicherstellen: zeitlich korrekt sortiert
    all_candles.sort(key=lambda x: x["time"])

    return all_candles

def fetch_ohlcs_v2_window(
    network: str,
    *,
    tf: str,
    from_ms: int,
    to_ms: int,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Fetch OHLC candles for a specific time window [from_ms, to_ms].
    Uses /v2/futures/ohlcs with range, from, to, limit.
    """
    tf = tf.strip().lower()
    if tf not in TF_TO_V2_RANGE:
        raise ValueError(f"Unsupported timeframe for v2 ohlcs: {tf}")

    url = base_url_for(network) + "/v2/futures/ohlcs"
    params = {
        "from": int(from_ms),
        "to": int(to_ms),
        "limit": min(max(int(limit), 1), 1000),
        "range": TF_TO_V2_RANGE[tf],
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    data.sort(key=lambda x: x["time"])
    return data
