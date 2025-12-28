# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict
from strategy import StrategyParams

SCENARIOS: Dict[str, StrategyParams] = {
    "baseline_30_70": StrategyParams(rsi_buy_level=30, rsi_sell_level=70),

    "tighter_25_75": StrategyParams(rsi_buy_level=25, rsi_sell_level=75),

    # the good one you had (volatility filter + trailing handled in paper.py)
    "tighter_25_75_bw": StrategyParams(
        rsi_buy_level=25,
        rsi_sell_level=75,
        bb_width_min=0.00,
        ema_fast_period=75,
        ema_slow_period=200,
        ema_filter_mode="trend",
        trail_stop_pct=0.03,
        time_stop_bars=600, 
    ),

}
