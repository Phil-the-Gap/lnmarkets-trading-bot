from __future__ import annotations

import os
from ..backtest import run_backtest


def main() -> None:
    network = os.getenv("LNM_NETWORK", "mainnet")
    tf = os.getenv("TIMEFRAME", "10m")
    scenario = os.getenv("SCENARIO", "tighter_25_75_bw")
    limit = int(os.getenv("BACKTEST_LIMIT", "100000"))

    run_backtest(network=network, tf=tf, limit=limit, scenario_name=scenario)


if __name__ == "__main__":
    main()
