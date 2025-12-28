# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from cache_ohlcs import get_cached_or_fetch_incremental
from indicators import bollinger_bands, rsi_wilder, ema_update
from strategy import bb_rsi_strategy
from paper import Position, paper_execute_long_short, PositionSide
from capital import CapitalConfig
from params import SCENARIOS

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def max_drawdown(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = x - peak
        if dd < max_dd:
            max_dd = dd
    return max_dd



def run_backtest(network: str, tf: str, limit: int, scenario_name: str) -> Dict[str, Any]:
    exit_reasons = {
    "TRAIL": 0,
    "TIME": 0,
}

    
    
    print("Starting backtest...")
    p = SCENARIOS[scenario_name]
    ema_period = getattr(p, "ema_slow_period", None) or getattr(p, "ema_period", 200)
    print("ema_period =", ema_period)
    print("Scenario:", scenario_name)
    print("trail_stop_pct =", p.trail_stop_pct)


    ohlcs = get_cached_or_fetch_incremental(network, tf=tf, total=limit, page_size=1000)
    print("OHLCs loaded:", len(ohlcs),
          "first_ts=", ohlcs[0]["time"] if ohlcs else None,
          "last_ts=", ohlcs[-1]["time"] if ohlcs else None)

    closes: List[float] = []
    rsi_hist: List[float] = []

    cfg_cap = CapitalConfig(start_equity_sats=1_000_000, risk_per_trade_pct=0.04, leverage=10.0)
    equity_sats = float(cfg_cap.start_equity_sats)
    pos = Position()

    bars_in_position = 0

    trades = 0
    long_entries = short_entries = 0
    long_exits = short_exits = 0

    trade_pnls: List[float] = []
    trade_rois: List[float] = []
    trade_notionals: List[float] = []
    trade_leverages: List[float] = []

    equity_curve_realized: List[float] = []
    equity_curve_candles: List[float] = []
    times: List[datetime] = []

    # set trailing here (so we don't bake it into StrategyParams)
    TRAIL_STOP_PCT = 0.03  # change here per run if you want

    # ✅ VOR der Schleife (einmal!)
    ema_fast_period = p.ema_fast_period
    ema_slow_period = p.ema_slow_period or 200

    ema_fast = None
    ema_slow = None

    print("EMA fast/slow:", ema_fast_period, ema_slow_period, "mode:", getattr(p, "ema_filter_mode", None))




    for i, c in enumerate(ohlcs, 1):
        close = float(c["close"])
        ts = datetime.fromtimestamp(int(c["time"]) / 1000, tz=timezone.utc)

        # 1) close zuerst speichern (damit windows korrekt sind)
        closes.append(close)

        # 2) EMA updaten (inkrementell)
        # EMA slow
        if ema_slow is None:
            if len(closes) >= ema_slow_period:
                ema_slow = sum(closes[-ema_slow_period:]) / ema_slow_period
        else:
            ema_slow = ema_update(ema_slow, close, ema_slow_period)

        # EMA fast (nur wenn konfiguriert)
        if ema_fast_period is not None:
            if ema_fast is None:
                if len(closes) >= ema_fast_period:
                    ema_fast = sum(closes[-ema_fast_period:]) / ema_fast_period
            else:
                ema_fast = ema_update(ema_fast, close, ema_fast_period)


        # 3) equity per candle tracken (immer)
        times.append(ts)
        equity_curve_candles.append(equity_sats)

        # 4) Warmup check
        need = max(
            p.bb_period,
            p.rsi_period + 1,
            (ema_slow_period or 2),
            (ema_fast_period or 2),
            2
        )
        if len(closes) < need:
            continue


        prev_close = closes[-2]

        window_bb = closes[-p.bb_period:]
        window_rsi = closes[-(p.rsi_period + 1):]

        bb = bollinger_bands(window_bb, period=p.bb_period, k=p.bb_k)
        rsi = rsi_wilder(window_rsi, period=p.rsi_period)
        if bb is None or rsi is None:
            continue

        # RSI history (für cross)
        rsi_hist.append(rsi)
        if len(rsi_hist) < 2:
            continue
        rsi_prev, rsi_curr = rsi_hist[-2], rsi_hist[-1]

        # bars in position
        if pos.side != PositionSide.FLAT:
            bars_in_position += 1
        else:
            bars_in_position = 0

        signal = bb_rsi_strategy(
            close=close,
            prev_close=prev_close,
            bb=bb,
            rsi_prev=rsi_prev,
            rsi_curr=rsi_curr,
            pos_side=pos.side,
            bars_in_position=bars_in_position,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            p=p,
        )


        trade = paper_execute_long_short(
            signal.name,
            ts,
            close,
            pos,
            equity_sats,
            cfg_cap,
            funding_rate=0.0,
            stop_distance_pct=0.0125,
            trail_stop_pct=p.trail_stop_pct,   # ✅ EINMAL
            use_hard_stop=True,
            catastrophe_mult=3.0,
        )



        if trade:
            # ENTRY
            if trade.side == "BUY":
                long_entries += 1
                bars_in_position = 0

            elif trade.side == "SHORT":
                short_entries += 1
                bars_in_position = 0

            # EXIT LONG
            elif trade.side.startswith("SELL"):
                long_exits += 1

                # Reason nur bei echtem Exit zählen
                if p.time_stop_bars and bars_in_position >= p.time_stop_bars:
                    exit_reasons["TIME"] += 1
                else:
                    exit_reasons["TRAIL"] += 1

                bars_in_position = 0

            # EXIT SHORT
            elif trade.side.startswith("COVER"):
                short_exits += 1

                if p.time_stop_bars and bars_in_position >= p.time_stop_bars:
                    exit_reasons["TIME"] += 1
                else:
                    exit_reasons["TRAIL"] += 1

                bars_in_position = 0



        

        if trade and trade.pnl_sats is not None:
            equity_sats += float(trade.pnl_sats)
            trades += 1

            notional = float(trade.notional_sats)
            roi = (float(trade.pnl_sats) / notional) * 100.0 if notional else 0.0

            trade_pnls.append(float(trade.pnl_sats))
            trade_rois.append(roi)
            trade_notionals.append(notional)
            trade_leverages.append(float(trade.leverage))
            equity_curve_realized.append(equity_sats)

        if i % 10000 == 0:
            print(f"Processed {i} candles | trades={trades} | equity_sats={equity_sats:,.0f}")


    print("Finished processing candles. Now computing stats + plotting...")

    curve_for_dd = equity_curve_realized if equity_curve_realized else equity_curve_candles
    mdd = max_drawdown(curve_for_dd)

    wins = [x for x in trade_pnls if x > 0]
    winrate = (len(wins) / trades * 100.0) if trades else 0.0
    avg_trade = (sum(trade_pnls) / trades) if trades else 0.0
    avg_roi = (sum(trade_rois) / trades) if trades else 0.0
    avg_notional = (sum(trade_notionals) / trades) if trades else 0.0
    avg_lev = (sum(trade_leverages) / trades) if trades else 0.0
    avg_margin = (avg_notional / avg_lev) if avg_lev else 0.0

    pnl_sats = equity_sats - float(cfg_cap.start_equity_sats)
    pnl_pct = (pnl_sats / float(cfg_cap.start_equity_sats)) * 100.0

    print("\nBacktest summary")
    print("-------------------------")
    print(f"Candles:              {len(ohlcs)}")
    print(f"Trades:               {trades}")
    print(f"Winrate:              {winrate:.1f} %")
    print(f"Avg trade PnL:        {avg_trade:.2f} sats")
    print(f"Avg ROI per trade:    {avg_roi:.3f} %")
    print(f"Avg pos. notional:    {avg_notional:,.0f} sats")
    print(f"Avg margin used:      {avg_margin:,.0f} sats")
    print(f"Avg leverage used:    {avg_lev:.2f} x")
    print(f"Total PnL:            {pnl_sats:,.0f} sats ({pnl_pct:.2f} %)")
    print(f"Max Drawdown:         {mdd:,.0f} sats")
    print(f"Long entries:          {long_entries}")
    print(f"Short entries:         {short_entries}")
    print(f"Long exits:            {long_exits}")
    print(f"Short exits:           {short_exits}")

    print("\nExit reasons")
    print("-----------")
    for k, v in exit_reasons.items():
        print(f"{k}: {v}")


    # plot (Equity + BTC-Preis auf rechter Achse)
    outfile = f"equity_{network}_{tf}_{scenario_name}_{len(ohlcs)}.png"

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # --- Left axis: Equity (sats)
    l1, = ax1.plot(times, equity_curve_candles, linewidth=1.6, label="Equity (sats)", zorder=3)
    ax1.set_title(f"Equity Curve | {network} | tf={tf} | {scenario_name}")
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel("Equity (sats)")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,} sats"))

    # --- Right axis: BTC price (USD)
    ax2 = ax1.twinx()
    l2, = ax2.plot(times, closes, linewidth=1.1, alpha=0.35, label="BTC Price (USD)", zorder=1)
    ax2.set_ylabel("BTC Price (USD)")
    ax2.ticklabel_format(style="plain", axis="y")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Optional: Legende kombiniert
    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print("Saved plot to:", outfile)


    return {
        "scenario": scenario_name,
        "pnl_sats": pnl_sats,
        "pnl_pct": pnl_pct,
        "mdd_sats": mdd,
        "trades": trades,
        "winrate": winrate,
    }


if __name__ == "__main__":
    scenarios_to_run = ["tighter_25_75_bw"]
    results = []
    for name in scenarios_to_run:
        results.append(run_backtest(network="mainnet", tf="10m", limit=100000, scenario_name=name))

    print("\n=== Scenario comparison ===")
    for r in results:
        print(
            f"{r['scenario']}: "
            f"PnL={r['pnl_sats']:,.0f} sats ({r['pnl_pct']:.2f}%), "
            f"MDD={r['mdd_sats']:,.0f} sats, "
            f"Trades={r['trades']}, "
            f"Winrate={r['winrate']:.1f}%"
        )
