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

    # Convergence filter (v3.3)
    # MA 収束フィルタ: 拡散飛び乗りを抑制し、収束後の再拡大を狙う
    # 最適化結果 (2026-03-23, 6ヶ月データ, 219回試行, OOS PF=1.92)
    MA_SHORT_WINDOW: int = 5
    MA_MID_WINDOW: int = 10
    MA_LONG_WINDOW: int = 20
    CONVERGENCE_LOOKBACK: int = 5
    COMPRESSION_LOOKBACK: int = 5
    MAX_MA_SPREAD_PCT_FOR_ENTRY: float = 0.03       # 最適化: 0.03 (緩め = 件数確保)
    MIN_MA_CONVERGENCE_SCORE: float = 0.55          # 最適化: 0.55 (維持)
    MIN_RANGE_COMPRESSION_SCORE: float = 0.30       # 最適化: 0.30 (緩和 = 件数確保)
    MIN_VOLATILITY_COMPRESSION_SCORE: float = 0.50  # 最適化: 0.50 (やや厳格)
    CONVERGENCE_CONFIDENCE_BOOST: float = 0.04      # 最適化: 0.04 (控えめ)
    EXPANSION_PENALTY_AFTER_CROSS: float = 0.10     # 最適化: 0.10 (控えめ)
    SCANNER_CONVERGENCE_WEIGHT: float = 0.05

    # Tachibana (future)
    TACHIBANA_API_URL: str = ""
    TACHIBANA_USER: str = ""
    TACHIBANA_PASSWORD: str = ""

    model_config = {"env_file": ".env", "env_prefix": "KABUAI_", "extra": "ignore"}


settings = Settings()
