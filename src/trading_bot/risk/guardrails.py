from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class GuardrailConfig:
    max_trades_per_day: int = 10
    daily_loss_limit_pct: float = 0.02   # 2% of starting equity (or ref equity)
    cooldown_after_exit_sec: int = 60


@dataclass
class GuardrailState:
    trades_today: int = 0
    day_key: str = ""
    last_exit_ts: Optional[float] = None  # epoch seconds
    daily_pnl_sats: float = 0.0


def _utc_day_key(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d")


def reset_if_new_day(state: GuardrailState, now: datetime) -> None:
    key = _utc_day_key(now)
    if state.day_key != key:
        state.day_key = key
        state.trades_today = 0
        state.daily_pnl_sats = 0.0
        state.last_exit_ts = None


def in_cooldown(state: GuardrailState, now_epoch: float, cfg: GuardrailConfig) -> bool:
    if state.last_exit_ts is None:
        return False
    return (now_epoch - state.last_exit_ts) < float(cfg.cooldown_after_exit_sec)


def can_open_new_trade(
    *,
    state: GuardrailState,
    cfg: GuardrailConfig,
    starting_equity_sats: float,
) -> bool:
    # Max trades/day
    if cfg.max_trades_per_day > 0 and state.trades_today >= cfg.max_trades_per_day:
        return False

    # Daily loss limit (simple, based on starting equity reference)
    if cfg.daily_loss_limit_pct > 0:
        limit = -abs(float(starting_equity_sats)) * float(cfg.daily_loss_limit_pct)
        if state.daily_pnl_sats <= limit:
            return False

    return True


def record_trade_open(state: GuardrailState) -> None:
    state.trades_today += 1


def record_trade_exit(state: GuardrailState, now_epoch: float, pnl_sats: float) -> None:
    state.last_exit_ts = float(now_epoch)
    state.daily_pnl_sats += float(pnl_sats)
