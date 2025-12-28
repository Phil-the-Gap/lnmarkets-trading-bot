# -*- coding: utf-8 -*-
"""
capital.py

Kapitalmodell für Backtest/Papertrading in *sats*:
- Startkapital: 1_000_000 sats (standard)
- Risk pro Trade: 1.0% (standard)
- Leverage: 10x (standard)
- Fees (pro Seite): 0.03% (standard)
- Funding: modelliert über fundingRate (z.B. aus /v3/futures/ticker) und 8h-Intervalle

Wichtig:
- Dieses Modell ist bewusst "Backtest-tauglich": einfach, deterministisch, nachvollziehbar.
- Strategy bleibt: BUY/SELL/HOLD
- Execution/Backtest ruft nur: sizing + fee/funding + pnl
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Literal
import math


Side = Literal["LONG", "SHORT"]


@dataclass
class CapitalConfig:
    start_equity_sats: int = 1_000_000      # wird überschrieben in bot.py
    risk_per_trade_pct: float = 0.00       #  wird überschrieben in bot.py
    leverage: int = 20                   # wird überschrieben in bot.py

    # Fees
    fee_rate_per_side: float = 0.0002         # 0.02% pro Seite (Entry/Exit)

    # Funding
    funding_interval_hours: int = 8           # typisch Perps
    # Optional: Slippage später
    slippage_bps: float = 1.0                 # 0.0 = aus; z.B. 1.0 = 1 bp = 0.01%


@dataclass
class PositionSizing:
    notional_sats: int          # Positions-Notional in sats (vereinfacht)
    margin_sats: int            # gebundene Margin (notional / leverage)
    risk_sats: int              # maximaler Verlust (z.B. 10_000 sats)
    stop_distance_pct: float    # z.B. 0.01 = 1%
    leverage: float


@dataclass
class CostBreakdown:
    entry_fee_sats: float = 0.0
    exit_fee_sats: float = 0.0
    funding_sats: float = 0.0
    slippage_sats: float = 0.0

    @property
    def total_sats(self) -> float:
        return self.entry_fee_sats + self.exit_fee_sats + self.funding_sats + self.slippage_sats


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _funding_events_between(cfg: CapitalConfig, entry_time: datetime, exit_time: datetime) -> int:
    """
    Zählt, wie viele Funding-Intervalle (8h) zwischen entry und exit liegen.
    Einfaches Modell: floor(delta_hours / interval)
    """
    e = _to_utc(entry_time)
    x = _to_utc(exit_time)
    if x <= e:
        return 0
    delta_hours = (x - e).total_seconds() / 3600.0
    return int(math.floor(delta_hours / float(cfg.funding_interval_hours)))


def calc_risk_sats(equity_sats: float, cfg: CapitalConfig) -> int:
    return int(math.floor(equity_sats * cfg.risk_per_trade_pct))


def position_size_from_risk(
    equity_sats: float,
    entry_price: float,
    stop_price: float,
    cfg: CapitalConfig,
) -> PositionSizing:
    """
    Kernidee:
    - Du definierst Stop-Distanz (aus Strategy oder rules).
    - Du willst max risk_sats verlieren, wenn Stop getroffen wird.
    - Daraus folgt Positions-NOTIONAL in sats.

    Vereinfachtes PnL-Modell (Backtest):
    - PnL_sats ≈ notional_sats * (exit - entry) / entry
      -> Bei Stop: exit=stop -> Verlust ≈ notional_sats * stop_distance_pct
      -> notional = risk / stop_distance_pct

    Leverage wirkt als Margin-Effizienz:
    - margin = notional / leverage
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")

    stop_distance_pct = abs(entry_price - stop_price) / entry_price
    if stop_distance_pct <= 0:
        raise ValueError("stop_distance_pct must be > 0 (stop != entry)")

    risk_sats = calc_risk_sats(equity_sats, cfg)

    # Notional so wählen, dass ein Move von stop_distance_pct genau risk_sats kostet
    notional = risk_sats / stop_distance_pct

    # Runden/Clamp
    notional_sats = int(math.floor(notional))
    margin_sats = int(math.ceil(notional_sats / cfg.leverage))

    return PositionSizing(
        notional_sats=notional_sats,
        margin_sats=margin_sats,
        risk_sats=risk_sats,
        stop_distance_pct=stop_distance_pct,
        leverage=cfg.leverage,
    )


def fees_for_notional(notional_sats: float, cfg: CapitalConfig) -> float:
    """Entry oder Exit Fee für eine Seite."""
    return float(notional_sats) * cfg.fee_rate_per_side


def slippage_cost_sats(notional_sats: float, cfg: CapitalConfig) -> float:
    """
    Optionales Slippage-Modell: bps auf Notional (grob).
    1 bp = 0.01% = 0.0001
    """
    if cfg.slippage_bps <= 0:
        return 0.0
    return float(notional_sats) * (cfg.slippage_bps * 1e-4)


def funding_cost_sats(
    notional_sats: float,
    funding_rate: float,
    entry_time: datetime,
    exit_time: datetime,
    side: Side,
    cfg: CapitalConfig,
) -> float:
    """
    Sehr einfaches Funding-Modell:
    - funding_rate gilt pro Funding-Event
    - funding_cost = notional * funding_rate * n_events
    - Vorzeichen:
      - Häufig: Long zahlt, Short bekommt (wenn funding_rate > 0)
      - Wir modellieren:
        funding_rate > 0 => LONG zahlt (cost positiv), SHORT bekommt (cost negativ)
    """
    n = _funding_events_between(cfg, entry_time, exit_time)
    if n <= 0:
        return 0.0

    gross = float(notional_sats) * float(funding_rate) * float(n)

    if funding_rate >= 0:
        return gross if side == "LONG" else -gross
    else:
        # funding_rate < 0: SHORT zahlt, LONG bekommt
        return -gross if side == "LONG" else gross


def pnl_sats_from_move(
    notional_sats: float,
    entry_price: float,
    exit_price: float,
    side: Side,
) -> float:
    """
    Vereinfachtes PnL-Modell in sats:
    - LONG:  notional * (exit-entry)/entry
    - SHORT: notional * (entry-exit)/entry
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    move = (exit_price - entry_price) / entry_price
    if side == "LONG":
        return float(notional_sats) * move
    else:
        return float(notional_sats) * (-move)


def net_pnl_sats(
    notional_sats: float,
    entry_price: float,
    exit_price: float,
    entry_time: datetime,
    exit_time: datetime,
    side: Side,
    cfg: CapitalConfig,
    funding_rate: float = 0.0,
) -> tuple[float, CostBreakdown]:
    """
    Liefert:
    - net_pnl_sats (nach Fees/Funding/Slippage)
    - CostBreakdown
    """
    gross = pnl_sats_from_move(notional_sats, entry_price, exit_price, side)

    entry_fee = fees_for_notional(notional_sats, cfg)
    exit_fee = fees_for_notional(notional_sats, cfg)

    funding = funding_cost_sats(
        notional_sats=notional_sats,
        funding_rate=funding_rate,
        entry_time=entry_time,
        exit_time=exit_time,
        side=side,
        cfg=cfg,
    )

    slip = slippage_cost_sats(notional_sats, cfg)

    costs = CostBreakdown(
        entry_fee_sats=entry_fee,
        exit_fee_sats=exit_fee,
        funding_sats=funding,
        slippage_sats=slip,
    )

    net = gross - costs.total_sats
    return net, costs
