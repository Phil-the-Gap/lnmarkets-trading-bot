from __future__ import annotations

import os
from ..bot import main as bot_main


def main() -> None:
    mode = os.getenv("BOT_MODE", "paper").strip().lower()
    confirm = os.getenv("LIVE_CONFIRM", "").strip().upper()

    # Hard safety: Live requires two explicit opt-ins
    if mode == "live" and confirm != "YES":
        raise SystemExit(
            "Refusing to start in LIVE mode without LIVE_CONFIRM=YES. "
            "Set LIVE_CONFIRM=YES only if you really want to trade live."
        )

    bot_main()


if __name__ == "__main__":
    main()
