# paper.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime
from typing import Optional, Literal

from capital import (
    CapitalConfig,
    PositionSizing,
    position_size_from_risk,
    net_pnl_sats,
)

SignalName = Literal["BUY", "SELL", "SHORT", "COVER", "HOLD"]


class PositionSide(Enum):
    FLAT = auto()
    LONG = auto()
    SHORT = auto()


@dataclass
class Position:
    side: PositionSide = PositionSide.FLAT
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    peak_price: float = 0.0
    trade_id: Optional[str] = None

        # trailing state
    highest: Optional[float] = None  # LONG
    lowest: Optional[float] = None   # SHORT

    # capital model
    notional_sats: int = 0
    margin_sats: int = 0
    leverage: float = 1.0

    def reset(self) -> None:
        self.side = PositionSide.FLAT
        self.entry_price = None
        self.entry_time = None
        self.highest = None
        self.lowest = None
        self.notional_sats = 0
        self.margin_sats = 0
        self.leverage = 1.0


@dataclass
class Trade:
    side: str  # "BUY"/"SELL"/"SHORT"/"COVER"/"SELL_TRAIL"/"COVER_STOP"/...
    time: datetime
    price: float

    notional_sats: int
    margin_sats: int
    leverage: float

    entry_price: Optional[float] = None
    pnl_sats: Optional[float] = None

    fees_sats: float = 0.0
    funding_sats: float = 0.0
    slippage_sats: float = 0.0

    reason: str = ""  # e.g. "TRAIL_STOP", "HARD_STOP", "CATA_STOP", "SIGNAL"


def _default_stop_price_long(entry_price: float, stop_distance_pct: float) -> float:
    return entry_price * (1.0 - stop_distance_pct)


def _default_stop_price_short(entry_price: float, stop_distance_pct: float) -> float:
    return entry_price * (1.0 + stop_distance_pct)


def _clamp_sizing_to_equity(sizing: PositionSizing, equity_sats: float, cfg: CapitalConfig) -> PositionSizing:
    """
    Safety clamp: if margin > equity, cap notional so margin <= equity.
    """
    if sizing.margin_sats <= equity_sats:
        return sizing

    max_notional = int(equity_sats * cfg.leverage)
    return PositionSizing(
        notional_sats=max_notional,
        margin_sats=int(equity_sats),
        risk_sats=sizing.risk_sats,
        stop_distance_pct=sizing.stop_distance_pct,
        leverage=cfg.leverage,
    )


def _exit_trade(
    *,
    exit_side: str,
    reason: str,
    ts: datetime,
    price: float,
    pos: Position,
    cfg: CapitalConfig,
    funding_rate: float,
) -> Trade:
    """
    Computes net pnl via capital.net_pnl_sats (fees + funding + slippage)
    and resets the position.
    """
    entry_price = pos.entry_price if pos.entry_price is not None else price
    entry_time = pos.entry_time if pos.entry_time is not None else ts
    notional = int(pos.notional_sats)

    side = "LONG" if pos.side == PositionSide.LONG else "SHORT"

    pnl, costs = net_pnl_sats(
        notional_sats=notional,
        entry_price=float(entry_price),
        exit_price=float(price),
        entry_time=entry_time,
        exit_time=ts,
        side=side,
        cfg=cfg,
        funding_rate=float(funding_rate),
    )

    t = Trade(
        side=exit_side,
        time=ts,
        price=float(price),
        notional_sats=notional,
        margin_sats=int(pos.margin_sats),
        leverage=float(pos.leverage),
        entry_price=float(entry_price),
        pnl_sats=float(pnl),
        fees_sats=float(costs.entry_fee_sats + costs.exit_fee_sats),
        funding_sats=float(costs.funding_sats),
        slippage_sats=float(costs.slippage_sats),
        reason=reason,
    )

    pos.reset()
    return t


def paper_execute_long_short(
    signal_name: SignalName,
    ts: datetime,
    price: float,
    pos: Position,
    equity_sats: float,
    cfg: CapitalConfig,
    *,
    funding_rate: float = 0.0,
    stop_distance_pct: float = 0.01,
    trail_stop_pct: float = 0.0,
    use_hard_stop: bool = True,
    catastrophe_mult: float = 3.0,
) -> Optional[Trade]:
    """
    Paper execution:
    - Risk-based sizing (risk_per_trade_pct) using stop_distance_pct
    - Fees + Funding + Slippage via capital.net_pnl_sats()
    - Exits:
        - signal exits: SELL/COVER
        - hard stop (optional)
        - trailing stop (optional)
        - catastrophe stop (failsafe, always on)
    """

    # ----------------------------
    # 1) ENTRY (nur wenn FLAT)
    # ----------------------------
    if pos.side == PositionSide.FLAT:
        if signal_name == "BUY":
            entry_price = float(price)
            stop_price = _default_stop_price_long(entry_price, float(stop_distance_pct))

            sizing = position_size_from_risk(
                equity_sats=float(equity_sats),
                entry_price=entry_price,
                stop_price=stop_price,
                cfg=cfg,
            )
            sizing = _clamp_sizing_to_equity(sizing, float(equity_sats), cfg)

            pos.side = PositionSide.LONG
            pos.entry_price = entry_price
            pos.entry_time = ts
            pos.notional_sats = int(sizing.notional_sats)
            pos.margin_sats = int(sizing.margin_sats)
            pos.leverage = float(cfg.leverage)

            pos.highest = entry_price
            pos.lowest = None

            return Trade(
                side="BUY",
                time=ts,
                price=entry_price,
                notional_sats=pos.notional_sats,
                margin_sats=pos.margin_sats,
                leverage=pos.leverage,
                reason="SIGNAL",
            )

        if signal_name == "SHORT":
            entry_price = float(price)
            stop_price = _default_stop_price_short(entry_price, float(stop_distance_pct))

            sizing = position_size_from_risk(
                equity_sats=float(equity_sats),
                entry_price=entry_price,
                stop_price=stop_price,
                cfg=cfg,
            )
            sizing = _clamp_sizing_to_equity(sizing, float(equity_sats), cfg)

            pos.side = PositionSide.SHORT
            pos.entry_price = entry_price
            pos.entry_time = ts
            pos.notional_sats = int(sizing.notional_sats)
            pos.margin_sats = int(sizing.margin_sats)
            pos.leverage = float(cfg.leverage)

            pos.lowest = entry_price
            pos.highest = None

            return Trade(
                side="SHORT",
                time=ts,
                price=entry_price,
                notional_sats=pos.notional_sats,
                margin_sats=pos.margin_sats,
                leverage=pos.leverage,
                reason="SIGNAL",
            )

        return None

    # ----------------------------
    # 2) Update trailing state
    # ----------------------------
    if pos.side == PositionSide.LONG:
        pos.highest = float(price) if pos.highest is None else max(pos.highest, float(price))
    elif pos.side == PositionSide.SHORT:
        pos.lowest = float(price) if pos.lowest is None else min(pos.lowest, float(price))

    # Guard
    if pos.entry_price is None or pos.entry_time is None or pos.notional_sats <= 0:
        pos.reset()
        return None

    entry = float(pos.entry_price)

    # ----------------------------
    # 3) Risk exits (Stops)
    # ----------------------------
    if pos.side == PositionSide.LONG:
        hard_stop = entry * (1.0 - float(stop_distance_pct))
        cat_stop = entry * (1.0 - float(catastrophe_mult) * float(stop_distance_pct))

        if use_hard_stop and float(price) <= hard_stop:
            return _exit_trade(
                exit_side="SELL_STOP",
                reason="HARD_STOP",
                ts=ts,
                price=float(price),
                pos=pos,
                cfg=cfg,
                funding_rate=float(funding_rate),
            )

        if float(trail_stop_pct) > 0.0 and pos.highest is not None:
            trail_price = pos.highest * (1.0 - float(trail_stop_pct))
            if float(price) <= trail_price:
                return _exit_trade(
                    exit_side="SELL_TRAIL",
                    reason="TRAIL_STOP",
                    ts=ts,
                    price=float(price),
                    pos=pos,
                    cfg=cfg,
                    funding_rate=float(funding_rate),
                )

        # failsafe
        if float(price) <= cat_stop:
            return _exit_trade(
                exit_side="SELL_CATA",
                reason="CATA_STOP",
                ts=ts,
                price=float(price),
                pos=pos,
                cfg=cfg,
                funding_rate=float(funding_rate),
            )

    elif pos.side == PositionSide.SHORT:
        hard_stop = entry * (1.0 + float(stop_distance_pct))
        cat_stop = entry * (1.0 + float(catastrophe_mult) * float(stop_distance_pct))

        if use_hard_stop and float(price) >= hard_stop:
            return _exit_trade(
                exit_side="COVER_STOP",
                reason="HARD_STOP",
                ts=ts,
                price=float(price),
                pos=pos,
                cfg=cfg,
                funding_rate=float(funding_rate),
            )

        if float(trail_stop_pct) > 0.0 and pos.lowest is not None:
            trail_price = pos.lowest * (1.0 + float(trail_stop_pct))
            if float(price) >= trail_price:
                return _exit_trade(
                    exit_side="COVER_TRAIL",
                    reason="TRAIL_STOP",
                    ts=ts,
                    price=float(price),
                    pos=pos,
                    cfg=cfg,
                    funding_rate=float(funding_rate),
                )

        # failsafe
        if float(price) >= cat_stop:
            return _exit_trade(
                exit_side="COVER_CATA",
                reason="CATA_STOP",
                ts=ts,
                price=float(price),
                pos=pos,
                cfg=cfg,
                funding_rate=float(funding_rate),
            )

    # ----------------------------
    # 4) Signal exits
    # ----------------------------
    if signal_name == "SELL" and pos.side == PositionSide.LONG:
        return _exit_trade(
            exit_side="SELL",
            reason="SIGNAL",
            ts=ts,
            price=float(price),
            pos=pos,
            cfg=cfg,
            funding_rate=float(funding_rate),
        )

    if signal_name == "COVER" and pos.side == PositionSide.SHORT:
        return _exit_trade(
            exit_side="COVER",
            reason="SIGNAL",
            ts=ts,
            price=float(price),
            pos=pos,
            cfg=cfg,
            funding_rate=float(funding_rate),
        )

    return None
