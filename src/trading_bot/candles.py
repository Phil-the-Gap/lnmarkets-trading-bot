# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Iterable, Tuple


@dataclass
class Candle:
    start: datetime   # start time of the candle (UTC)
    open: float
    high: float
    low: float
    close: float
    n: int = 1        # number of ticks aggregated (optional)

    def update(self, price: float) -> None:
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.n += 1


def parse_timeframe(tf: str) -> timedelta:
    """
    Parse timeframe strings like:
    "1m", "5m", "10m", "1h", "2h", "1d"
    """
    tf = tf.strip().lower()
    if len(tf) < 2:
        raise ValueError(f"Invalid timeframe: {tf}")

    unit = tf[-1]
    num_str = tf[:-1]
    if not num_str.isdigit():
        raise ValueError(f"Invalid timeframe: {tf}")

    n = int(num_str)
    if n <= 0:
        raise ValueError(f"Invalid timeframe: {tf}")

    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "d":
        return timedelta(days=n)

    raise ValueError(f"Unsupported timeframe unit: {unit} (use m/h/d)")


def floor_time(dt: datetime, step: timedelta) -> datetime:
    """
    Floors dt (UTC) down to the start of its timeframe bucket.
    Works for steps that are whole seconds (true for m/h/d).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    step_seconds = int(step.total_seconds())
    if step_seconds <= 0:
        raise ValueError("step must be > 0 seconds")

    epoch = int(dt.timestamp())
    floored = (epoch // step_seconds) * step_seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)


class CandleBuilder:
    """
    Incremental candle builder:
    feed (timestamp, price) ticks → emits finished candles when a new bucket starts.
    """

    def __init__(self, timeframe: str) -> None:
        self.step = parse_timeframe(timeframe)
        self.current: Optional[Candle] = None

    def add_tick(self, ts: datetime, price: float) -> List[Candle]:
        """
        Add a tick. Returns a list of finished candles (0 or 1 normally).
        """
        start = floor_time(ts, self.step)
        finished: List[Candle] = []

        if self.current is None:
            self.current = Candle(start=start, open=price, high=price, low=price, close=price, n=1)
            return finished

        if start == self.current.start:
            self.current.update(price)
            return finished

        # New candle bucket started → close previous candle
        finished.append(self.current)
        self.current = Candle(start=start, open=price, high=price, low=price, close=price, n=1)
        return finished

    def flush(self) -> Optional[Candle]:
        """
        Call at the end to get the last (still-open) candle.
        """
        return self.current
