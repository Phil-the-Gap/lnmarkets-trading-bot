from datetime import datetime, timezone

from trading_bot.risk.guardrails import (
    GuardrailConfig,
    GuardrailState,
    reset_if_new_day,
    can_open_new_trade,
    record_trade_open,
    record_trade_exit,
)

def test_reset_if_new_day_resets_counters():
    cfg = GuardrailConfig(max_trades_per_day=2, daily_loss_limit_pct=0.1, cooldown_after_exit_sec=60)
    st = GuardrailState(trades_today=5, daily_pnl_sats=-123, day_key="2020-01-01")
    reset_if_new_day(st, datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert st.trades_today == 0
    assert st.daily_pnl_sats == 0.0
    assert st.day_key == "2025-01-01"

def test_can_open_trade_blocks_on_max_trades():
    cfg = GuardrailConfig(max_trades_per_day=2, daily_loss_limit_pct=0.0, cooldown_after_exit_sec=0)
    st = GuardrailState(trades_today=2, day_key="2025-01-01")
    assert can_open_new_trade(state=st, cfg=cfg, starting_equity_sats=1_000_000) is False

def test_can_open_trade_blocks_on_daily_loss():
    cfg = GuardrailConfig(max_trades_per_day=999, daily_loss_limit_pct=0.02, cooldown_after_exit_sec=0)
    st = GuardrailState(trades_today=0, day_key="2025-01-01", daily_pnl_sats=-25_000)
    # 2% of 1,000,000 = 20,000
    assert can_open_new_trade(state=st, cfg=cfg, starting_equity_sats=1_000_000) is False

def test_record_trade_open_and_exit_updates_state():
    cfg = GuardrailConfig(max_trades_per_day=10, daily_loss_limit_pct=0.0, cooldown_after_exit_sec=0)
    st = GuardrailState(day_key="2025-01-01")
    record_trade_open(st)
    assert st.trades_today == 1
    record_trade_exit(st, now_epoch=1000.0, pnl_sats=-123.0)
    assert st.daily_pnl_sats == -123.0
    assert st.last_exit_ts == 1000.0
