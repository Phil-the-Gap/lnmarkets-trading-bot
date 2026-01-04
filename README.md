This repository uses GitHub Actions to run tests on every push and pull request.


## Exchange / Platform Context

This trading system was originally designed and tested against the API of **LNMarkets.com**.

The project architecture is intentionally modular:
- Strategy logic, indicators, and risk management are **exchange-agnostic**
- Execution and account interaction are isolated behind adapter-style components

This allows the core trading logic to be reused or extended for other exchanges with minimal changes.

```markdown
![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)

\# Algorithmic Trading Bot (Python)



A production-oriented algorithmic trading system written in Python with a strong focus on \*\*engineering quality\*\*, \*\*risk controls\*\*, and \*\*reproducibility\*\*.



This project was built as a learning and research system and is intended as a \*\*software engineering portfolio project\*\*, not as a commercial trading product.



> ⚠️ \*\*Live trading is disabled by default.\*\*

> The repository is safe to run locally in backtest or paper mode.

---

\## Key Characteristics



\* Clear separation between \*\*strategy\*\*, \*\*risk\*\*, and \*\*execution\*\*

\* Deterministic \*\*backtesting\*\*

\* \*\*Paper trading\*\* with realistic fee, funding, and slippage modeling

\* Explicit \*\*risk guardrails\*\* (daily loss limit, max trades per day, cooldowns)

\* \*\*Position recovery\*\* after process restarts

\* Incremental OHLC data caching to reduce API load

\* No secrets or credentials committed to the repository



---



\## Trading Logic (High-Level)



\* Strategy:



&nbsp; \* Bollinger Bands + RSI based mean-reversion logic

&nbsp; \* Optional EMA trend filtering

\* Execution:



&nbsp; \* Paper execution engine

&nbsp; \* Live execution adapter (explicit opt-in only)

\* Risk:



&nbsp; \* Fixed fractional risk per trade

&nbsp; \* Stop-loss, trailing stop, time stop

&nbsp; \* Hard safety limits at runtime



The \*\*strategy layer is fully decoupled\*\* from execution and capital management.



---



\## Project Structure (Simplified)



```

.

├── bot.py              # Runtime loop (paper or live)

├── backtest.py         # Deterministic backtesting runner

├── strategy.py         # Signal generation (no execution)

├── indicators.py       # Pure technical indicators

├── capital.py          # Risk \& capital model

├── paper.py            # Paper execution + PnL accounting

├── candles.py          # Candle aggregation

├── history.py          # Market data fetching

├── cache\_ohlcs.py      # Local OHLC caching

├── params.py           # Strategy parameter scenarios

├── README.md

├── .env.example

└── .gitignore

```



---



\## Installation



\### Requirements



\* Python 3.10+

\* pip



\### Install dependencies



```bash

pip install -r requirements.txt

```



---



\## Configuration



Copy the example environment file:



```bash

cp .env.example .env

```



Edit `.env` as needed.

\*\*Do not commit `.env` to version control.\*\*



Important variables:



```env

BOT\_MODE=paper        # paper (default) or live

LNM\_NETWORK=mainnet   # or testnet

TIMEFRAME=10m

SCENARIO=tighter\_25\_75\_bw

```



---



\## Running a Backtest



```bash

python backtest.py

```



This will:



\* Load historical OHLC data (cached locally)

\* Run the selected strategy scenario

\* Print performance statistics

\* Save an equity curve plot locally



---



\## Running Paper Trading (Safe)



```bash

BOT\_MODE=paper python bot.py

```



\* No real orders are sent

\* All trades are simulated

\* PnL is computed using the capital model

---

“Run tests: pytest”

“Install: pip install -e .”

---


\## Live Trading (Explicit Opt-In)



⚠️ \*\*Live trading is disabled by default.\*\*



To enable live trading, you must:



1\. Provide valid API credentials

2\. Explicitly enable live mode



```bash

BOT\_MODE=live python bot.py

```



This is intentionally \*\*not enabled by default\*\* to prevent accidental execution.



---



\## Risk \& Safety Notes



\* This project is provided for \*\*educational and research purposes only\*\*

\* No profitability claims are made

\* Trading involves risk and can result in loss of capital

\* Use at your own risk



---



\## Engineering Focus



This project emphasizes:



\* deterministic behavior

\* recoverability after failures

\* explicit runtime safety checks

\* clean separation of concerns

\* reproducible experiments



It is intended to demonstrate \*\*backend, systems, and applied Python engineering skills\*\* rather than trading performance.



---



\## License



MIT License



