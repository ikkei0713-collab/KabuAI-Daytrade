"""KabuAI-Daytrade central configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Safety
    ALLOW_LIVE_TRADING: bool = False
    PAPER_TRADING: bool = True

    # Paths
    BASE_DIR: Path = Path.home() / "dev" / "KabuAI-Daytrade"
    DB_PATH: Path = Path.home() / "dev" / "KabuAI-Daytrade" / "db" / "kabuai.db"
    LOG_DIR: Path = Path.home() / "dev" / "KabuAI-Daytrade" / "logs"

    # J-Quants
    JQUANTS_API_KEY: str = ""
    JQUANTS_MAIL: str = ""
    JQUANTS_PASSWORD: str = ""
    JQUANTS_REFRESH_TOKEN: str = ""

    # Trading params – 保守的チューニング (2026-03-19)
    MAX_POSITIONS: int = 2             # 5→2: 同時保有を絞りリスク集中を回避
    MAX_POSITION_SIZE: float = 250000  # 50万→25万: 1件あたり損失上限を半減
    TOTAL_CAPITAL: float = 3000000     # 300万円
    MAX_LOSS_PER_DAY: float = -15000   # -5万→-1.5万: 日次損失0.5%で停止
    MAX_HOLDING_MINUTES: int = 360     # 6時間=当日完結

    # Market hours (JST)
    MARKET_OPEN: str = "09:00"
    MARKET_CLOSE: str = "15:30"
    PRE_MARKET_SCAN: str = "08:30"
    FORCE_CLOSE_TIME: str = "14:50"  # 強制決済 15:20→14:50 余裕を持って決済

    # Strategy
    MIN_CONFIDENCE: float = 0.65   # 0.6→0.65: 高確信シグナルのみ通過
    STRATEGY_SCORE_THRESHOLD: float = 0.5

    # Tachibana (future)
    TACHIBANA_API_URL: str = ""
    TACHIBANA_USER: str = ""
    TACHIBANA_PASSWORD: str = ""

    model_config = {"env_file": ".env", "env_prefix": "KABUAI_"}


settings = Settings()
