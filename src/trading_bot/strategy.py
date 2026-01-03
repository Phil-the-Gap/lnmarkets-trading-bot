# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Literal

from .indicators import BollingerBands, bollinger_width


class Signal(Enum):
    HOLD = auto()
    BUY = auto()      # enter long
    SELL = auto()     # exit long
    SHORT = auto()    # enter short
    COVER = auto()    # exit short


@dataclass
class StrategyParams:
    bb_period: int = 20
    bb_k: float = 2.0
    rsi_period: int = 14

    ema_fast_period: Optional[int] = None
    ema_slow_period: Optional[int] = 200
    ema_filter_mode: Optional[Literal["trend", "price", "crossover"]] = None

    @property
    def ema_period(self) -> int:
        # Legacy-Fallback für Code, der noch p.ema_period erwartet
        return int(self.ema_slow_period or 200)

    rsi_buy_level: float = 25.0
    rsi_sell_level: float = 75.0

    reentry_required: bool = True
    exit_at_middle: bool = False
    time_stop_bars: int = 60        #wird von params.py gesetzt

    bb_width_min: float = 0.0
    max_ema_dist_pct: Optional[float] = None

    trail_stop_pct: float = 0.0     #wird von params.py gesetzt


def rsi_cross_up(prev: float, curr: float, level: float) -> bool:
    return prev < level and curr >= level


def rsi_cross_down(prev: float, curr: float, level: float) -> bool:
    return prev > level and curr <= level


def _ema_filter_ok(close: float, ema: Optional[float], max_dist_pct: Optional[float]) -> bool:
    if max_dist_pct is None or max_dist_pct <= 0.0:
        return True
    if ema is None or ema == 0:
        return False
    dist = abs(close - ema) / ema
    return dist <= max_dist_pct


def bb_rsi_strategy(
    *,
    close: float,
    prev_close: float,
    bb: BollingerBands,
    rsi_prev: float,
    rsi_curr: float,
    pos_side,
    bars_in_position: int,
    ema_fast: Optional[float],
    ema_slow: Optional[float],
    p: StrategyParams,
) -> Signal:
    w = bollinger_width(bb)
    if p.bb_width_min and w < p.bb_width_min:
        return Signal.HOLD

    # --- EMA distance filter (optional) ---
    # nutzt ema_slow als Referenz
    if not _ema_filter_ok(close, ema_slow, p.max_ema_dist_pct):
        return Signal.HOLD

    # --- EMA trend/price filter (optional) ---
    mode = p.ema_filter_mode

    if mode == "trend":
        # nur long wenn fast > slow, nur short wenn fast < slow
        if ema_fast is None or ema_slow is None:
            return Signal.HOLD
        allow_long = ema_fast > ema_slow
        allow_short = ema_fast < ema_slow

    elif mode == "price":
        # nur long wenn close > slow, nur short wenn close < slow
        if ema_slow is None:
            return Signal.HOLD
        allow_long = close > ema_slow
        allow_short = close < ema_slow

    else:
        # None oder unbekannt => kein EMA Regime Filter
        allow_long = True
        allow_short = True

    # --- Exits first ---
    if pos_side.name == "LONG":
        if p.exit_at_middle and close >= bb.middle:
            return Signal.SELL
        if p.time_stop_bars and bars_in_position >= p.time_stop_bars:
            return Signal.SELL
        return Signal.HOLD

    if pos_side.name == "SHORT":
        if p.exit_at_middle and close <= bb.middle:
            return Signal.COVER
        if p.time_stop_bars and bars_in_position >= p.time_stop_bars:
            return Signal.COVER
        return Signal.HOLD

    # --- Entries (FLAT) ---
    lower_breach_prev = prev_close < bb.lower
    upper_breach_prev = prev_close > bb.upper

    long_reentry_ok = (not p.reentry_required) or (lower_breach_prev and close >= bb.lower)
    short_reentry_ok = (not p.reentry_required) or (upper_breach_prev and close <= bb.upper)

    long_entry = long_reentry_ok and rsi_cross_up(rsi_prev, rsi_curr, p.rsi_buy_level) and allow_long
    short_entry = short_reentry_ok and rsi_cross_down(rsi_prev, rsi_curr, p.rsi_sell_level) and allow_short

    if long_entry:
        return Signal.BUY
    if short_entry:
        return Signal.SHORT

    return Signal.HOLD

