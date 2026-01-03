from __future__ import annotations

import os
from ..bot import main as bot_main


def main() -> None:
    # Force-safe defaults
    os.environ.setdefault("BOT_MODE", "paper")

    print("[runner] Starting PAPER runtime (no live trading possible)")
    bot_main()


if __name__ == "__main__":
    main()
