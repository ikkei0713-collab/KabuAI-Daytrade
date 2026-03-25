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

    # Trading params – 攻撃的チューニング (2026-03-25)
    # 目標: 3万円→6万円/月 (月利100%)
    MAX_POSITIONS: int = 3             # 同時3ポジション: 機会最大化
    MAX_POSITION_SIZE: float = 30000   # 3万円: 全資金1ポジションもあり
    TOTAL_CAPITAL: float = 30000       # 3万円
    MAX_LOSS_PER_DAY: float = -3000    # -3,000円: 資金の10%で日次停止
    MAX_HOLDING_MINUTES: int = 360     # 6時間=当日完結

    # Market hours (JST)
    MARKET_OPEN: str = "09:00"
    MARKET_CLOSE: str = "15:30"
    PRE_MARKET_SCAN: str = "08:30"
    FORCE_CLOSE_TIME: str = "14:50"  # 強制決済 15:20→14:50 余裕を持って決済

    # 後場 PM-VWAP reclaim (vwap_reclaim 強化用, JST)
    PM_SESSION_START: str = "12:30"
    PM_ENTRY_START: str = "13:00"
    PM_FORCE_EXIT: str = "14:45"
    PM_RECLAIM_MIN_HOLD_COUNT: int = 2
    PM_RELATIVE_VOLUME_THRESHOLD: float = 1.8
    PM_TURNOVER_THRESHOLD: float = 1_000_000_000.0
    PM_INTRADAY_QUALITY_MIN: float = 0.60
    PM_CONFIDENCE_BOOST: float = 0.08
    PM_EVENT_BOOST: float = 0.05
    PM_LOW_PRICE_BONUS_MAX: float = 0.03
    PM_EXPECTED_PM_VOLUME_FRACTION: float = 0.45
    PM_VWAP_SLOPE_MAX_NEG: float = 0.0008
    PM_SETUP_WEIGHT: float = 0.08
    LOW_PRICE_BONUS_ENABLED: bool = True
    LOW_PRICE_BONUS_MIN: int = 100
    LOW_PRICE_BONUS_MAX: int = 500
    LOW_PRICE_BONUS_WEIGHT: float = 0.03

    # Strategy
    MIN_CONFIDENCE: float = 0.30   # 0.30: 機会最大化
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
