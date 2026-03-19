"""KabuAI Daytrade - AI-Powered Day Trading System for Japanese Stocks."""

import asyncio
import argparse
import subprocess
import sys

from loguru import logger

from core.config import settings
from core.safety import SafetyGuard


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="KabuAI Daytrade - AI-powered Japanese stock day-trading system",
    )
    parser.add_argument(
        "--mode",
        choices=["trade", "analyze", "backtest", "ui"],
        default="ui",
        help="Operating mode (default: ui)",
    )
    parser.add_argument(
        "--date",
        help="Date for analysis mode (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    # ---- Safety first ----
    if settings.ALLOW_LIVE_TRADING:
        logger.error("LIVE TRADING IS ENABLED - ABORTING FOR SAFETY")
        raise RuntimeError(
            "ALLOW_LIVE_TRADING must be False. "
            "This system is in development and must not place real orders."
        )

    logger.info("KabuAI Daytrade starting in PAPER TRADING mode")

    # ---- Initialise database ----
    from db.database import DatabaseManager

    db = DatabaseManager()
    await db.init_db()

    # ---- Register strategies ----
    from strategies.registry import StrategyRegistry

    StrategyRegistry.register_all_defaults()

    # ---- Dispatch by mode ----
    if args.mode == "trade":
        logger.info("Starting trading loop...")
        from execution.engine import ExecutionEngine

        engine = ExecutionEngine(db=db)
        await engine.run_trading_loop()

    elif args.mode == "analyze":
        logger.info("Starting analysis for date={}", args.date)
        from analytics.learning_loop import LearningLoop

        loop = LearningLoop(db=db)
        await loop.run_daily_loop(args.date)

    elif args.mode == "backtest":
        logger.info("Starting interactive backtest session...")
        from tools.backtest import Backtester

        backtester = Backtester(db=db)
        await backtester.run_interactive()

    elif args.mode == "ui":
        logger.info("Launching Streamlit UI...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "ui/app.py"],
            check=False,
        )


if __name__ == "__main__":
    asyncio.run(main())
