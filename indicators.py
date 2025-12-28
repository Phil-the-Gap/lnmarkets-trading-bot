# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math


@dataclass(frozen=True)
class BollingerBands:
    middle: float  # SMA
    upper: float
    lower: float
    stdev: float


def sma(values: List[float]) -> float:
    """Simple moving average."""
    if not values:
        raise ValueError("sma() needs at least 1 value")
    return sum(values) / float(len(values))


def stddev_population(values: List[float]) -> float:
    """
    Population standard deviation.
    stdev = sqrt( mean( (x - mean)^2 ) )
    """
    if not values:
        raise ValueError("stddev_population() needs at least 1 value")

    m = sma(values)
    var = sum((x - m) ** 2 for x in values) / float(len(values))
    return math.sqrt(var)


def bollinger_bands(
    closes: List[float],
    period: int = 20,
    k: float = 2.0,
) -> Optional[BollingerBands]:
    """
    Compute Bollinger Bands for the latest close, using the last `period` closes.
    Returns None if there is not enough data yet.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")

    if len(closes) < period:
        return None

    window = closes[-period:]
    middle = sma(window)
    stdev = stddev_population(window)
    upper = middle + k * stdev
    lower = middle - k * stdev

    return BollingerBands(
        middle=middle,
        upper=upper,
        lower=lower,
        stdev=stdev,
    )

#RSI Berechnung

def rsi_wilder(closes: List[float], period: int = 14) -> Optional[float]:
    """
    Wilder's RSI (standard RSI).
    Returns the latest RSI value (0..100) or None if not enough data.

    Needs at least period+1 closes to compute the first average gain/loss.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(closes) < period + 1:
        return None

    # price changes
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # first averages (simple average of first `period` deltas)
    gains = [max(d, 0.0) for d in deltas[:period]]
    losses = [max(-d, 0.0) for d in deltas[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder smoothing for the rest
    for d in deltas[period:]:
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    # edge cases
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def ema_last(values: list[float], period: int) -> float | None:
    """Return last EMA value; None if not enough data."""
    if period <= 0 or len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    ema = sum(values[:period]) / period  # SMA seed
    for x in values[period:]:
        ema = alpha * x + (1.0 - alpha) * ema
    return ema

def bollinger_width(bb) -> float:
    """
    Normalized Bollinger Band width: (upper-lower)/middle
    Returns 0 if middle is 0 (shouldn't happen in practice).
    """
    if bb is None or bb.middle == 0:
        return 0.0
    return (bb.upper - bb.lower) / bb.middle

def ema_update(prev_ema: float, x: float, period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    return alpha * x + (1.0 - alpha) * prev_ema


