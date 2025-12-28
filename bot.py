# -*- coding: utf-8 -*-
from datetime import datetime, timezone
import time
import os
import socket
import requests
import csv

from dotenv import load_dotenv
from lnm_rest_client import LNMarketsRestClient, LNMarketsConfig
from candles import CandleBuilder
from indicators import bollinger_bands, rsi_wilder, ema_last, bollinger_width
from strategy import bb_rsi_strategy, Signal
from params import SCENARIOS
from capital import CapitalConfig
from paper import Position, paper_execute_long_short, PositionSide
from history import fetch_ohlcs_v2
from typing import Optional, Any


POLL_INTERVAL = 6  # Sekunden


def must(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise SystemExit(f"Missing env var: {name}")
    return v


def iso_to_dt(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)

def fetch_live_equity_sats(client) -> int:
        account = client.request("get", "account")
        return int(account["balance"])

def _safe_get(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def fetch_isolated_running_trades(client: LNMarketsRestClient) -> list[dict]:
    # Running trades = echte offene Positionen (filled, not closed)
    res = client.request("get", "/futures/isolated/trades/running", auth=True)
    return res if isinstance(res, list) else []


def fetch_isolated_open_orders(client: LNMarketsRestClient) -> list[dict]:
    # Open trades = offene Limit-Orders (waiting to be filled)
    res = client.request("get", "/futures/isolated/trades/open", auth=True)
    return res if isinstance(res, list) else []


def fetch_isolated_active(client: LNMarketsRestClient) -> list[dict]:
    # Für Recovery: running hat Priorität. Falls API running nicht liefert, fallback open.
    active = []
    try:
        active.extend(fetch_isolated_running_trades(client))
    except Exception:
        pass

    try:
        active.extend(fetch_isolated_open_orders(client))
    except Exception:
        pass

    return active


def v3_isolated_new_trade(client: LNMarketsRestClient, payload: dict) -> Optional[dict]:
    running = fetch_isolated_running_trades(client)
    if running:
        print("SKIP ENTRY: running trade exists")
        return None

    print("payload:", payload)
    return client.request(
        "post",
        "/futures/isolated/trade",
        payload=payload,
        auth=True,
    )


def v3_isolated_close_trade(client: LNMarketsRestClient, trade_id: str) -> dict:
    return client.request(
        "post",
        "/futures/isolated/trade/close",   # ✅ exakt wie Doku
        payload={"id": trade_id},
        auth=True,
    )


def recover_single_position_on_start(*, client, pos, current_price):
    active = fetch_isolated_active(client)

    # nur RUNNING als echte Position betrachten:
    running = [t for t in active if isinstance(t, dict) and t.get("running") is True and t.get("closed") is False]

    if not running:
        pos.side = PositionSide.FLAT
        pos.trade_id = None
        pos.entry_price = 0.0
        pos.entry_ts = None
        pos.peak_price = 0.0
        return 0, "RECOVERY: no running position"

    if len(running) > 1:
        ids = [t.get("id") for t in running]
        raise SystemExit(f"RECOVERY ABORT: more than 1 running trade: {ids}")

    t = running[0]
    pos.trade_id = t["id"]
    pos.entry_price = float(t.get("entryPrice") or current_price)

    side = (t.get("side") or "").lower()
    if side == "buy":
        pos.side = PositionSide.LONG
    elif side == "sell":
        pos.side = PositionSide.SHORT
    else:
        raise SystemExit(f"RECOVERY ABORT: unknown side={t.get('side')}")

    pos.peak_price = current_price
    return 0, f"RECOVERY: restored {pos.side.name} id={pos.trade_id} entry={pos.entry_price}"



def live_execute_market_only(
    *,
    client: LNMarketsRestClient,
    signal: str,               # "BUY"/"SELL"/"SHORT"/"COVER"/"HOLD"
    ts: datetime,
    price: float,
    pos: Position,
    equity_sats: float,
    cfg_cap: CapitalConfig,
    stop_distance_pct: float = 0.01,
    trail_stop_pct: float = 0.0,
    time_stop_bars: int = 0,
    bars_in_position: int = 0,
    use_server_stoploss: bool = False,
) -> Optional[dict]:
    """
    Live execution (LN Markets Futures) - Market-only:
    - ENTRY:  POST /v2/futures  type='m'
    - EXIT:   DELETE /v2/futures?id=<trade_id>

    Trailing stop + time stop are handled locally (bot logic),
    while stoploss can optionally be set server-side at entry.
    """
    lev = cfg_cap.leverage
    quantity = 0
    quantity = int(quantity)

    # -------------------------
    # ENTRY (market)
    # -------------------------
    if pos.side == PositionSide.FLAT and signal in ("BUY", "SHORT"):

        live_equity = fetch_live_equity_sats(client)

        risk_sats = max(1.0, float(live_equity) * float(cfg_cap.risk_per_trade_pct))
        margin_sats = risk_sats / (float(cfg_cap.leverage) * float(stop_distance_pct))
        margin_sats = int(max(1_000, margin_sats))

        # quantity ableiten (konservativ): Notional in USD
        margin_btc = margin_sats / 100_000_000.0
        notional_usd = margin_btc * float(price) * float(cfg_cap.leverage)
        quantity = int(max(1, round(notional_usd)))

        payload = {
            "type": "market",  # ✅ market => price muss fehlen
            "side": "buy" if signal == "BUY" else "sell",
            "leverage": int(lev),
            "quantity": int(quantity),  # ✅ quantity ODER margin
            "clientId": "lnm-mini-bot",  # optional, max 64 chars
        }
        
        

        if use_server_stoploss:
            stoploss = price * (1.0 - stop_distance_pct) if signal == "BUY" else price * (1.0 + stop_distance_pct)
            stoploss = round(stoploss * 2) / 2.0  # 0.5 tick
            if float(stoploss).is_integer():
                stoploss = int(stoploss)
            payload["stoploss"] = stoploss

        # payload ist fertig
        print("ENTRY PAYLOAD PREVIEW:", payload)

        t = v3_isolated_new_trade(client, payload)
        trade_id = str(t["id"])


        pos.side = PositionSide.LONG if signal == "BUY" else PositionSide.SHORT
        pos.entry_price = float(t.get("entry_price") or price)
        pos.entry_ts = ts
        pos.trade_id = trade_id
        pos.peak_price = price  # Start peak at entry

        return {
            "kind": "ENTRY",
            "reason": "SIGNAL",
            "side": signal,
            "id": trade_id,
            "price": price,
            "ts": ts,
            "margin": margin_sats,
            "equity_sats": live_equity,
        }

    # -------------------------
    # MANAGE OPEN POSITION
    # -------------------------
    if pos.side != PositionSide.FLAT and pos.trade_id:
        # Update peak for trailing
        if pos.side == PositionSide.LONG:
            pos.peak_price = max(pos.peak_price, price)
        else:
            pos.peak_price = min(pos.peak_price, price)

        exit_reason = None

        # 1) Strategy exit signal
        if (signal == "SELL" and pos.side == PositionSide.LONG) or (signal == "COVER" and pos.side == PositionSide.SHORT):
            exit_reason = "SIGNAL"

        # 2) Time stop
        if exit_reason is None and time_stop_bars and bars_in_position >= int(time_stop_bars):
            exit_reason = "TIME_STOP"

        # 3) Trailing stop (local)
        if exit_reason is None and trail_stop_pct and trail_stop_pct > 0:
            if pos.side == PositionSide.LONG:
                trail_price = pos.peak_price * (1.0 - float(trail_stop_pct))
                if price <= trail_price:
                    exit_reason = "TRAIL_STOP"
            else:
                trail_price = pos.peak_price * (1.0 + float(trail_stop_pct))
                if price >= trail_price:
                    exit_reason = "TRAIL_STOP"

        # If exit triggered -> market close via DELETE
        if exit_reason is not None:
            v3_isolated_close_trade(client, str(pos.trade_id))
            live_equity = fetch_live_equity_sats(client)

            out = {
                "kind": "EXIT",
                "reason": exit_reason,
                "side": "SELL" if pos.side == PositionSide.LONG else "COVER",
                "id": pos.trade_id,
                "price": price,
                "ts": ts,
                "entry_price": float(pos.entry_price),
                "peak_price": float(pos.peak_price),
                "equity_sats": live_equity,
            }

            # reset local pos
            pos.side = PositionSide.FLAT
            pos.entry_price = 0.0
            pos.entry_ts = None
            pos.qty_sats = 0.0
            pos.trade_id = None
            pos.peak_price = 0.0

            return out

    return None


def utc_day_key(ts: datetime) -> str:
    # YYYY-MM-DD in UTC
    return ts.astimezone(timezone.utc).date().isoformat()


def append_trade_csv(path: str, row: dict) -> None:
    # schreibt header automatisch wenn Datei neu
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def main():
    load_dotenv()

    cfg = LNMarketsConfig(
        key=must("LNM_KEY"),
        secret=must("LNM_SECRET"),
        passphrase=must("LNM_PASSPHRASE"),
        network=os.getenv("LNM_NETWORK", "mainnet").strip() or "mainnet",
    )

    client = LNMarketsRestClient(cfg)

    # --- MODE ---
    DRY_RUN = False  # ✅ erstmal True lassen (live orders später)

    # --- SAFETY / GUARDRAILS (für Live-Test mit kleinem Geld) ---
    MAX_NOTIONAL_SATS = 1_000_000          # ✅ harte Obergrenze pro Trade (Notional)
    DAILY_LOSS_LIMIT_PCT = 1         # ✅ -8% am Tag => Bot stoppt
    MAX_TRADES_PER_DAY = 100             # ✅ Spam-Schutz
    COOLDOWN_AFTER_EXIT_SEC = 180       # ✅ 3 Minuten Pause nach Exit
    TRADE_LOG_CSV = "live_trade_log.csv"

    # --- Timeframe / Warmup ---
    TIMEFRAME = "10m"
    SEED_CANDLES = 300  # ✅ wegen EMA200

    # --- Capital model (Paper) ---
    cfg_cap = CapitalConfig(
        start_equity_sats=1_000_000,
        risk_per_trade_pct=0.025,
        leverage=10,
    )

    # --- Strategy params (nur 1 Scenario) ---
    p = SCENARIOS["tighter_25_75_bw"]
    scenario_name = "tighter_25_75_bw"
    print("Scenario:", scenario_name, p,"Risk_per_trade: ",cfg_cap.risk_per_trade_pct)

    cb = CandleBuilder(TIMEFRAME)
    closes: list[float] = []
    rsi_hist: list[float] = []
    bars_in_position = 0

    # Seed history
    ohlcs = fetch_ohlcs_v2("mainnet", tf=TIMEFRAME, limit=SEED_CANDLES)
    print("seed ohlcs len =", len(ohlcs), "tf=", TIMEFRAME, "net=", cfg.network)
    if ohlcs:
        print("seed first time =", ohlcs[0]["time"], "last time =", ohlcs[-1]["time"])

    closes.extend([float(c["close"]) for c in ohlcs])
    print("Seeded last close:", closes[-1])
    print(f"Seeded {len(closes)} closes.")

    equity_sats = fetch_live_equity_sats(client)
    pos = Position()

        # --- get current price once for recovery ---
    try:
        ticker0 = client.request("get", "/futures/ticker")
        ts0 = iso_to_dt(ticker0["time"])
        best0 = ticker0["prices"][0]
        mid0 = (best0["bidPrice"] + best0["askPrice"]) / 2.0
    except Exception as e:
        raise SystemExit(f"Startup failed: cannot fetch ticker for recovery. {repr(e)}")

    # --- startup recovery (single position) ---
    bars_in_position, info = recover_single_position_on_start(
        client=client,
        pos=pos,
        current_price=float(mid0),
    )
    print(info)


    account = client.request("get", "account")
    print("ACCOUNT:", account)

    # --- Guardrail runtime state ---
    current_day = None
    day_start_equity = equity_sats
    trades_today = 0
    cooldown_until = 0.0

    last_poll = 0.0

    while True:
        now = time.time()
        if now - last_poll < POLL_INTERVAL:
            time.sleep(1)
            continue
        last_poll = now

        # ticker holen
        try:
            ticker = client.request("get", "/futures/ticker")

        except RuntimeError as e:
            msg = str(e)
            print("API error:", msg)
            time.sleep(60 if "429" in msg else 10)
            continue

        except (requests.exceptions.RequestException, socket.gaierror) as e:
            print("Network/DNS error:", repr(e))
            time.sleep(10)
            continue

        ts = iso_to_dt(ticker["time"])
        best = ticker["prices"][0]
        mid = (best["bidPrice"] + best["askPrice"]) / 2.0

        finished = cb.add_tick(ts, mid)

        for candle in finished:
            if pos.side != PositionSide.FLAT:
                running = fetch_isolated_running_trades(client)  # /futures/isolated/trades/running
                if not running:
                    print("SYNC: position was closed manually on exchange -> resetting local position to FLAT")
                    pos.side = PositionSide.FLAT
                    pos.trade_id = None
                    pos.entry_price = 0.0
                    pos.entry_ts = None
                    pos.peak_price = 0.0
                    bars_in_position = 0
                
                else:
                    rid = str(running[0]["id"])
                    if pos.trade_id and rid != str(pos.trade_id):
                        print(f"SYNC: exchange trade id changed {pos.trade_id} -> {rid}, recovering")
                        bars_in_position, info = recover_single_position_on_start(
                            client=client, pos=pos, current_price=float(close)
                        )
                        print(info)

            candle_ts = candle.start  # ✅ Candle-Zeit für Signale/Trades

            # --- Day rollover reset ---
            day_key = utc_day_key(candle_ts)
            if current_day is None:
                current_day = day_key
                day_start_equity = equity_sats
                trades_today = 0
            elif day_key != current_day:
                current_day = day_key
                day_start_equity = equity_sats
                trades_today = 0
                cooldown_until = 0.0
                print(f"--- NEW UTC DAY {current_day} | day_start_equity={day_start_equity:,.0f} sats ---")

            # --- Daily loss limit kill-switch ---
            if equity_sats < day_start_equity * (1.0 - DAILY_LOSS_LIMIT_PCT):
                raise SystemExit(
                    f"DAILY LOSS LIMIT HIT: equity={equity_sats:,.0f} < "
                    f"{day_start_equity*(1.0-DAILY_LOSS_LIMIT_PCT):,.0f} "
                    f"(-{DAILY_LOSS_LIMIT_PCT*100:.2f}%). Stopping."
                )

            # --- Max trades per day ---
            if trades_today >= MAX_TRADES_PER_DAY:
                raise SystemExit(f"MAX TRADES PER DAY HIT: {trades_today} >= {MAX_TRADES_PER_DAY}. Stopping.")

            close = float(candle.close)
            closes.append(close)

            need = max(p.bb_period, p.rsi_period + 1, p.ema_slow_period, 2)
            if len(closes) < need:
                print(f"[{candle.start.isoformat()}] warmup {len(closes)}/{need}")
                continue

            prev_close = closes[-2]

            bb = bollinger_bands(closes[-p.bb_period:], period=p.bb_period, k=p.bb_k)
            rsi = rsi_wilder(closes[-(p.rsi_period + 1):], period=p.rsi_period)

            ema_fast = ema_last(closes, p.ema_fast_period)
            ema_slow = ema_last(closes, p.ema_slow_period)

            if bb is None or rsi is None:
                continue

            rsi_hist.append(rsi)
            if len(rsi_hist) < 2:
                continue
            rsi_prev, rsi_curr = rsi_hist[-2], rsi_hist[-1]

            # bars held
            if pos.side != PositionSide.FLAT:
                bars_in_position += 1
            else:
                bars_in_position = 0

            # --- Cooldown after exit (only blocks new entries) ---
            in_cooldown = (time.time() < cooldown_until)

            # Signal
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

            #signal = Signal.SELL #Hier kann ein hartes Signal gesetzt werden für Tests

            # block entries during cooldown
            if in_cooldown and pos.side == PositionSide.FLAT and signal.name in ("BUY", "SHORT"):
                signal = Signal.HOLD

            # --- Compute intended notional & cap it ---
            intended_notional = equity_sats * cfg_cap.leverage
            #capped_notional = min(intended_notional, float(MAX_NOTIONAL_SATS))

            # ⚠️ Wir "simulieren" die kleinere Size, indem wir equity_sats für paper_execute
            # so skalieren, dass qty_sats = equity*lev = capped_notional.
            equity_for_exec = intended_notional / cfg_cap.leverage

          
            if DRY_RUN:
                trade = paper_execute_long_short(
                    signal.name,
                    candle_ts,              # ✅ Candle time (nicht ticker time)
                    close,
                    pos,
                    equity_for_exec,        # ✅ size cap greift hier
                    cfg_cap,
                    funding_rate=0.0,
                    stop_distance_pct=0.01,
                    trail_stop_pct=p.trail_stop_pct,
                    use_hard_stop=False,
                    catastrophe_mult=3.0,
                )

            else:
                trade = live_execute_market_only(
                    client=client,
                    signal=signal.name,
                    ts=candle_ts,
                    price=close,
                    pos=pos,
                    equity_sats=equity_sats,  # wird beim entry sowieso live neu geholt
                    cfg_cap=cfg_cap,
                    stop_distance_pct=0.01,
                    trail_stop_pct=p.trail_stop_pct,       # ✅ aus params.py
                    time_stop_bars=p.time_stop_bars,       # ✅ aus params.py
                    bars_in_position=bars_in_position,     # ✅ wichtig für time stop
                    use_server_stoploss=True, 
                )

                                    
                    
                

            if trade:
                trades_today += 1

                if isinstance(trade, dict):
                    print(
                        f"=== LIVE {trade['kind']} {trade['side']} ({trade.get('reason','')}) === "
                        f"{trade['ts'].isoformat()} price={trade['price']:,.2f} id={trade['id']}"
                    )

                    equity_sats = fetch_live_equity_sats(client)

                    if trade.get("kind") == "EXIT":
                        cooldown_until = time.time() + COOLDOWN_AFTER_EXIT_SEC

                    row = {
                        "time": trade["ts"].isoformat(),
                        "side": trade["side"],
                        "price": float(trade["price"]),
                        "entry_price": float(trade.get("entry_price") or 0) if trade.get("entry_price") is not None else "",
                        "pnl_sats": trade.get("pnl_sats", ""),
                        "fees_sats": trade.get("fees_sats", ""),
                        "slippage_sats": trade.get("slippage_sats", ""),
                        "funding_sats": trade.get("funding_sats", ""),
                        "equity_sats": float(equity_sats),
                        "day": current_day,
                        "trades_today": trades_today,
                        "max_notional_sats": MAX_NOTIONAL_SATS,
                        "trail_pct": p.trail_stop_pct,
                        "signal": signal.name,
                    }
                    append_trade_csv(TRADE_LOG_CSV, row)

                else:
                    # Paper-trade Objekt
                    print(
                        f"=== TRADE {trade.side} === {trade.time.isoformat()} "
                        f"price={trade.price:,.2f} notional={trade.notional_sats:,} sats lev={trade.leverage:.1f}x"
                    )

                    if trade.pnl_sats is not None:
                        equity_sats += float(trade.pnl_sats)

                    row = {
                        "time": trade.time.isoformat(),
                        "side": trade.side,
                        "price": float(trade.price),
                        "entry_price": float(getattr(trade, "entry_price", 0) or 0) if getattr(trade, "entry_price", None) is not None else "",
                        "pnl_sats": "" if trade.pnl_sats is None else float(trade.pnl_sats),
                        "fees_sats": "" if getattr(trade, "fees_sats", None) is None else float(trade.fees_sats),
                        "slippage_sats": "" if getattr(trade, "slippage_sats", None) is None else float(trade.slippage_sats),
                        "funding_sats": "" if getattr(trade, "funding_sats", None) is None else float(trade.funding_sats),
                        "equity_sats": float(equity_sats),
                        "day": current_day,
                        "trades_today": trades_today,
                        "max_notional_sats": MAX_NOTIONAL_SATS,
                        "trail_pct": p.trail_stop_pct,
                        "signal": signal.name,
                    }
                    append_trade_csv(TRADE_LOG_CSV, row)


            # Equity update: im Live-Modus holen wir Equity neu von der API
            if trade:
                if isinstance(trade, dict):
                    # Live: Equity nach Entry/Exit neu holen (du machst das oben schon)
                    pass
                else:
                    # Paper: pnl_sats ist Attribut
                    if trade.pnl_sats is not None:
                        equity_sats += float(trade.pnl_sats)


            w = bollinger_width(bb)
            print(
                f"[{candle.start.isoformat()}] close={close:,.2f} "
                f"BW={w*100:,.2f}% RSI={rsi_curr:5.2f} "
                f"SIG={signal.name:<6} POS={pos.side.name:<5} "
                f"EQ={equity_sats:,.0f} sats "
                f"trail={p.trail_stop_pct*100:.2f}% "
                f"notional_cap={MAX_NOTIONAL_SATS:,} "
                f"trades_today={trades_today}"
            )


if __name__ == "__main__":
    main()
