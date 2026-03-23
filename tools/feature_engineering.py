"""Feature engineering for Japanese stock day-trading.

Calculates technical indicators, volume metrics, VWAP statistics,
price features, candlestick patterns, and momentum signals from
OHLCV DataFrames.  All computations use pandas / numpy directly
so there is no hard dependency on external TA libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FeatureSet:
    """Container for all computed features on a single ticker."""

    technical: dict[str, Any] = field(default_factory=dict)
    volume: dict[str, Any] = field(default_factory=dict)
    vwap: dict[str, Any] = field(default_factory=dict)
    price: dict[str, Any] = field(default_factory=dict)
    pattern: dict[str, Any] = field(default_factory=dict)
    momentum: dict[str, Any] = field(default_factory=dict)
    convergence: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        """Merge every category into a single flat dict."""
        out: dict[str, Any] = {}
        for section in (
            self.technical,
            self.volume,
            self.vwap,
            self.price,
            self.pattern,
            self.momentum,
            self.convergence,
        ):
            out.update(section)
        return out


class FeatureEngineer:
    """Stateless feature calculator.

    All public methods accept a pandas DataFrame with *at least* the
    columns ``open``, ``high``, ``low``, ``close``, ``volume`` (case-
    insensitive).  The caller is responsible for providing enough rows
    for the longest look-back window (typically 50 bars for SMA-50).
    """

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def calculate_all_features(self, ohlcv: pd.DataFrame) -> dict[str, Any]:
        """Compute every feature category and return a flat dict.

        Args:
            ohlcv: DataFrame with columns open/high/low/close/volume.

        Returns:
            A flat ``{feature_name: value}`` dictionary suitable for
            storage and downstream model consumption.
        """
        df = self._normalise_columns(ohlcv)
        if df.empty:
            logger.warning("Empty DataFrame passed to calculate_all_features")
            return {}

        fs = FeatureSet(
            technical=self._technical(df),
            volume=self._volume(df),
            vwap=self._vwap(df),
            price=self._price(df),
            pattern=self._pattern(df),
            momentum=self._momentum(df),
            convergence=self._convergence(df),
        )
        out = fs.to_flat_dict()

        # Aliases: strategies expect short names
        if "atr_14" in out:
            out["atr"] = out["atr_14"]
        if "distance_from_vwap" in out:
            out["vwap_distance"] = out["distance_from_vwap"]

        # Simulated features for paper trading (日足ベース推定)
        # NOTE: これらは intraday proxy (擬似特徴量) であり、
        #       日足データから推定した値で実際のイントラデーデータではない。
        #       proxy 依存が高い戦略の評価信頼度は限定的。
        close = df["close"].iloc[-1] if not df.empty else 0
        high = df["high"].iloc[-1] if not df.empty else 0
        low = df["low"].iloc[-1] if not df.empty else 0
        vol = df["volume"].iloc[-1] if not df.empty else 0
        vol_avg = df["volume"].tail(20).mean() if len(df) >= 20 else vol

        out.setdefault("pre_market_volume", vol * 0.1)
        out.setdefault("sector_momentum", 0.0)
        out.setdefault("opening_range_size", high - low)
        out.setdefault("opening_range_high", high)
        out.setdefault("opening_range_low", low)
        out.setdefault("volume_first_5min", vol * 0.05)
        out.setdefault("tick_direction", 1 if close > df["open"].iloc[-1] else -1)
        out.setdefault("time_below_vwap", 30)
        out.setdefault("volume_at_reclaim", vol * 1.2)
        out.setdefault("vwap_touches_today", 2)
        out.setdefault("trend_direction", 1 if out.get("ema_9", 0) and out.get("ema_21", 0) and out["ema_9"] > out["ema_21"] else -1)
        out.setdefault("trend_strength", abs(out.get("rsi_14", 50) - 50) / 50)
        out.setdefault("volume_trend", vol / vol_avg if vol_avg > 0 else 1.0)
        out.setdefault("atr_distance_from_vwap", abs(out.get("vwap_distance", 0)) / out.get("atr", 1) if out.get("atr") else 0)
        out.setdefault("volume_climax", 1 if vol > vol_avg * 2.5 else 0)
        out.setdefault("volume_spike", 1 if vol > vol_avg * 2 else 0)
        out.setdefault("price_vs_bollinger", (close - out.get("bb_lower", close)) / max(out.get("bb_upper", close) - out.get("bb_lower", close), 0.01))
        out.setdefault("intraday_drop_pct", (low - high) / high * 100 if high > 0 else 0)
        out.setdefault("volume_surge", vol / vol_avg if vol_avg > 0 else 1.0)
        out.setdefault("selling_exhaustion", 0)
        out.setdefault("support_level", low)
        out.setdefault("bid_ask_ratio", 1.0 + (close - df["open"].iloc[-1]) / max(out.get("atr", 1), 0.01))
        out.setdefault("spread", out.get("atr", 1) * 0.05)
        out.setdefault("depth_imbalance", out.get("bid_ask_ratio", 1.0) - 1.0)
        out.setdefault("volume_price_divergence", 0.0)
        out.setdefault("large_trade_detection", 1 if vol > vol_avg * 3 else 0)
        out.setdefault("spread_percentile", 0.5)
        out.setdefault("volume_building", vol / vol_avg if vol_avg > 0 else 1.0)
        out.setdefault("price_compression", out.get("bb_pct", 0.5))
        out.setdefault("event_type", "")
        out.setdefault("event_magnitude", 0.0)
        out.setdefault("market_cap", 0.0)
        out.setdefault("historical_event_response", 0.0)
        out.setdefault("earnings_surprise_pct", 0.0)
        out.setdefault("revenue_growth", 0.0)
        out.setdefault("guidance_change", 0.0)
        out.setdefault("news_sentiment", 0.0)
        out.setdefault("price_acceleration", 0.0)
        out.setdefault("historical_catalyst_response", 0.0)

        # Proxy feature metadata: 各特徴量が proxy (擬似) か real かを記録
        out["_proxy_features"] = self.get_proxy_features()
        out["_proxy_usage_rate"] = self._calc_proxy_usage_rate(out)

        return out

    # ------------------------------------------------------------------
    # Proxy feature tracking
    # ------------------------------------------------------------------

    # 擬似 intraday 特徴量リスト
    # これらは日足データから推定しており、実際の intraday データではない
    PROXY_FEATURES: set[str] = {
        # Orderbook 系 (完全ダミー)
        "spread_percentile",    # 固定値 0.5
        "bid_ask_ratio",        # close-open から推定
        "spread",               # ATR * 0.05 で推定
        "depth_imbalance",      # bid_ask_ratio - 1.0
        "large_trade_detection",  # volume から推定
        # Intraday timing 系 (日足では不可能)
        "time_below_vwap",      # 固定値 30
        "volume_at_reclaim",    # volume * 1.2 で推定
        "vwap_touches_today",   # 固定値 2
        "volume_first_5min",    # volume * 0.05 で推定
        "pre_market_volume",    # volume * 0.1 で推定
        "tick_direction",       # close vs open で推定
        # Opening range 系 (日足 high/low で代替)
        "opening_range_size",   # high - low (日足レンジ = intraday OR ではない)
        "opening_range_high",   # high (日足高値 ≠ 始値5分高値)
        "opening_range_low",    # low (日足安値 ≠ 始値5分安値)
        # Intraday dynamics 系
        "intraday_drop_pct",    # (low-high)/high で推定
        "selling_exhaustion",   # 固定値 0
        "volume_price_divergence",  # 固定値 0
        "price_acceleration",   # 固定値 0
        # Event 系 (外部データなしの場合)
        "earnings_surprise_pct",  # 固定値 0
        "revenue_growth",         # 固定値 0
        "guidance_change",        # 固定値 0
        "news_sentiment",         # 固定値 0
        "historical_catalyst_response",  # 固定値 0
        "historical_event_response",     # 固定値 0 (TDnet注入時は real)
        "market_cap",            # 固定値 0
    }

    # 収束系特徴量 (日足 OHLCV から直接計算 → proxy ではない)
    CONVERGENCE_FEATURES: set[str] = {
        "ma_short", "ma_mid", "ma_long",
        "ma_spread_pct", "ma_spread_zscore",
        "price_to_ma_short_pct", "price_to_ma_mid_pct", "price_to_ma_long_pct",
        "price_ma_cluster_score", "ma_convergence_score", "ma_convergence_trend",
        "range_compression_score", "volatility_compression_score",
        "post_cross_expansion_flag", "post_cross_consolidation_flag",
        "squeeze_breakout_ready", "extension_from_ma_score", "pullback_to_ma_score",
    }

    # 戦略が依存する特徴量マッピング
    STRATEGY_PROXY_DEPS: dict[str, list[str]] = {
        "vwap_reclaim": ["time_below_vwap", "volume_at_reclaim"],
        "orb": ["opening_range_high", "opening_range_low"],
        "trend_follow": [],  # EMA/VWAP/convergence は real
        "spread_entry": ["spread_percentile", "volume_building", "price_compression"],
        "gap_go": [],
        "gap_fade": [],
        "open_drive": ["opening_range_high", "opening_range_low"],
        "orderbook_imbalance": ["bid_ask_ratio", "depth_imbalance", "spread"],
        "large_absorption": ["large_trade_detection", "bid_ask_ratio"],
        "overextension": ["intraday_drop_pct"],
        "rsi_reversal": ["selling_exhaustion"],
        "crash_rebound": ["intraday_drop_pct", "selling_exhaustion"],
        "tdnet_event": [],  # TDnet は real データ
        "earnings_momentum": ["earnings_surprise_pct", "revenue_growth"],
        "catalyst_initial": ["news_sentiment", "historical_catalyst_response"],
        "vwap_bounce": ["time_below_vwap", "volume_at_reclaim"],
    }

    @classmethod
    def get_proxy_features(cls) -> list[str]:
        """Return the list of proxy (simulated) feature names."""
        return sorted(cls.PROXY_FEATURES)

    @classmethod
    def get_proxy_usage_rate(cls, strategy_name: str) -> float:
        """Return the proxy dependency rate (0-1) for a strategy.

        Higher values indicate greater reliance on simulated features,
        meaning lower evaluation reliability.
        """
        deps = cls.STRATEGY_PROXY_DEPS.get(strategy_name, [])
        if not deps:
            return 0.0
        proxy_count = sum(1 for d in deps if d in cls.PROXY_FEATURES)
        return proxy_count / max(len(deps), 1)

    @classmethod
    def get_all_proxy_usage_rates(cls) -> dict[str, float]:
        """Return proxy_usage_rate for all known strategies."""
        return {
            name: cls.get_proxy_usage_rate(name)
            for name in cls.STRATEGY_PROXY_DEPS
        }

    @classmethod
    def get_proxy_penalty(cls, strategy_name: str) -> float:
        """Return a confidence penalty (0 to 0.15) based on proxy dependency.

        Strategies that rely heavily on proxy features get a larger penalty
        to prevent overconfidence in their signals.

        Active 戦略 (vwap_reclaim) は主力なのでペナルティ上限を 0.05 に抑える。
        他戦略は rate * 0.15 (最大 0.15)。
        """
        from strategies.registry import StrategyRegistry
        rate = cls.get_proxy_usage_rate(strategy_name)
        status = StrategyRegistry.STRATEGY_STATUS.get(strategy_name, "off")
        if status == "active":
            # 主戦略は proxy ペナルティを軽減 (最大 0.05)
            return round(rate * 0.05, 3)
        # 0.0 → no penalty, 1.0 → -0.15
        return round(rate * 0.15, 3)

    def _calc_proxy_usage_rate(self, features: dict) -> float:
        """Calculate what fraction of non-None features are proxy."""
        if not features:
            return 0.0
        total = 0
        proxy_count = 0
        for k, v in features.items():
            if k.startswith("_"):
                continue
            if v is None:
                continue
            total += 1
            if k in self.PROXY_FEATURES:
                proxy_count += 1
        return round(proxy_count / max(total, 1), 3)

    # ------------------------------------------------------------------
    # Convergence / compression features (v3.3)
    # ------------------------------------------------------------------

    def _convergence(self, df: pd.DataFrame) -> dict[str, Any]:
        """MA収束・ボラティリティ圧縮・GC/DC後の状態を計算する.

        日足ベースなので intraday proxy ではなく、比較的安定して計算可能。
        """
        from core.config import settings

        close = df["close"]
        high = df["high"]
        low = df["low"]
        features: dict[str, Any] = {}
        n = len(df)

        sw = settings.MA_SHORT_WINDOW   # 5
        mw = settings.MA_MID_WINDOW     # 10
        lw = settings.MA_LONG_WINDOW    # 20
        conv_lb = settings.CONVERGENCE_LOOKBACK  # 5
        comp_lb = settings.COMPRESSION_LOOKBACK  # 5

        # --- Moving averages ---
        ma_short = close.rolling(window=sw, min_periods=sw).mean()
        ma_mid = close.rolling(window=mw, min_periods=mw).mean()
        ma_long = close.rolling(window=lw, min_periods=lw).mean()

        features["ma_short"] = _last(ma_short)
        features["ma_mid"] = _last(ma_mid)
        features["ma_long"] = _last(ma_long)

        last_close = close.iloc[-1] if n > 0 else None
        last_ma_s = features["ma_short"]
        last_ma_m = features["ma_mid"]
        last_ma_l = features["ma_long"]

        # --- MA spread ---
        if all(v is not None for v in [last_ma_s, last_ma_m, last_ma_l]) and last_close > 0:
            ma_max = max(last_ma_s, last_ma_m, last_ma_l)
            ma_min = min(last_ma_s, last_ma_m, last_ma_l)
            ma_spread = ma_max - ma_min
            features["ma_spread_pct"] = ma_spread / last_close

            # MA spread z-score (rolling over convergence lookback)
            spread_series = pd.Series(dtype=float)
            if n >= lw + conv_lb:
                for i in range(conv_lb):
                    idx = -(conv_lb - i)
                    s_val = ma_short.iloc[idx] if not pd.isna(ma_short.iloc[idx]) else None
                    m_val = ma_mid.iloc[idx] if not pd.isna(ma_mid.iloc[idx]) else None
                    l_val = ma_long.iloc[idx] if not pd.isna(ma_long.iloc[idx]) else None
                    if all(v is not None for v in [s_val, m_val, l_val]):
                        c_val = close.iloc[idx]
                        if c_val > 0:
                            spread_series = pd.concat([spread_series,
                                pd.Series([(max(s_val, m_val, l_val) - min(s_val, m_val, l_val)) / c_val])])
                if len(spread_series) >= 2 and spread_series.std() > 1e-10:
                    features["ma_spread_zscore"] = float(
                        (features["ma_spread_pct"] - spread_series.mean()) / spread_series.std()
                    )
                else:
                    features["ma_spread_zscore"] = 0.0
            else:
                features["ma_spread_zscore"] = 0.0

            # --- Price to MA distances ---
            features["price_to_ma_short_pct"] = (last_close - last_ma_s) / last_close
            features["price_to_ma_mid_pct"] = (last_close - last_ma_m) / last_close
            features["price_to_ma_long_pct"] = (last_close - last_ma_l) / last_close

            # --- Price-MA cluster score ---
            # 価格とMA群がどれだけ近いか (0-1, 高いほど密集)
            avg_dist = (abs(last_close - last_ma_s) + abs(last_close - last_ma_m) + abs(last_close - last_ma_l)) / 3.0
            avg_dist_pct = avg_dist / last_close if last_close > 0 else 1.0
            # 2% 以内なら 1.0, 5% 以上なら 0.0 に近づく
            features["price_ma_cluster_score"] = max(0.0, min(1.0, 1.0 - avg_dist_pct / 0.05))

            # --- MA convergence score ---
            # MA 群の距離が小さいほど高い (0-1)
            ma_spread_pct = features["ma_spread_pct"]
            # 1% 以内なら 1.0, 4% 以上なら 0.0
            features["ma_convergence_score"] = max(0.0, min(1.0, 1.0 - ma_spread_pct / 0.04))

            # --- MA convergence trend ---
            # 直近N本で ma_spread_pct が低下しているか (-1 ~ +1)
            if n >= lw + conv_lb:
                spread_vals = []
                for i in range(conv_lb):
                    idx = -(conv_lb - i)
                    s_v = ma_short.iloc[idx]
                    m_v = ma_mid.iloc[idx]
                    l_v = ma_long.iloc[idx]
                    c_v = close.iloc[idx]
                    if not any(pd.isna(x) for x in [s_v, m_v, l_v]) and c_v > 0:
                        spread_vals.append((max(s_v, m_v, l_v) - min(s_v, m_v, l_v)) / c_v)
                if len(spread_vals) >= 2:
                    # 負 = spread 縮小中 (収束中), 正 = 拡散中
                    diffs = [spread_vals[i] - spread_vals[i - 1] for i in range(1, len(spread_vals))]
                    avg_diff = sum(diffs) / len(diffs)
                    # normalize to -1 ~ +1
                    features["ma_convergence_trend"] = max(-1.0, min(1.0, -avg_diff / 0.005))
                else:
                    features["ma_convergence_trend"] = 0.0
            else:
                features["ma_convergence_trend"] = 0.0

            # --- Extension from MA score ---
            # MA群から離れすぎていないか (0-1, 低いほど乖離大)
            max_dist_pct = max(abs(features["price_to_ma_short_pct"]),
                               abs(features["price_to_ma_mid_pct"]),
                               abs(features["price_to_ma_long_pct"]))
            # 1% 以内なら 1.0, 5% 以上なら 0.0
            features["extension_from_ma_score"] = max(0.0, min(1.0, 1.0 - max_dist_pct / 0.05))

            # --- Pullback to MA score ---
            # 強いトレンド中に価格がMA群へ軽く押してきたか (0-1)
            if last_ma_s > last_ma_m > last_ma_l:  # uptrend order
                # 価格がshort MA付近 (0.5% 以内) なら高スコア
                pull_dist = abs(last_close - last_ma_s) / last_close
                features["pullback_to_ma_score"] = max(0.0, min(1.0, 1.0 - pull_dist / 0.01))
            elif last_ma_s < last_ma_m < last_ma_l:  # downtrend order
                pull_dist = abs(last_close - last_ma_s) / last_close
                features["pullback_to_ma_score"] = max(0.0, min(1.0, 1.0 - pull_dist / 0.01))
            else:
                features["pullback_to_ma_score"] = 0.0

        else:
            # Not enough data for MA convergence
            features["ma_spread_pct"] = None
            features["ma_spread_zscore"] = None
            features["price_to_ma_short_pct"] = None
            features["price_to_ma_mid_pct"] = None
            features["price_to_ma_long_pct"] = None
            features["price_ma_cluster_score"] = None
            features["ma_convergence_score"] = None
            features["ma_convergence_trend"] = None
            features["extension_from_ma_score"] = None
            features["pullback_to_ma_score"] = None

        # --- Range compression score ---
        # 直近N本の高安幅が縮小しているか (0-1)
        if n >= comp_lb + 5:
            ranges_recent = ((high.iloc[-comp_lb:] - low.iloc[-comp_lb:]) / close.iloc[-comp_lb:]).values
            ranges_prior = ((high.iloc[-comp_lb * 2:-comp_lb] - low.iloc[-comp_lb * 2:-comp_lb]) /
                            close.iloc[-comp_lb * 2:-comp_lb]).values if n >= comp_lb * 2 + 5 else ranges_recent
            avg_recent = float(np.mean(ranges_recent)) if len(ranges_recent) > 0 else 0
            avg_prior = float(np.mean(ranges_prior)) if len(ranges_prior) > 0 else avg_recent
            if avg_prior > 1e-10:
                ratio = avg_recent / avg_prior
                # ratio < 1.0 = 圧縮中, ratio > 1.0 = 拡大中
                features["range_compression_score"] = max(0.0, min(1.0, 1.0 - (ratio - 0.5) / 1.0))
            else:
                features["range_compression_score"] = 0.5
        else:
            features["range_compression_score"] = None

        # --- Volatility compression score ---
        # ATR の縮小度合い (0-1)
        if n >= 14 + comp_lb:
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(window=14, min_periods=14).mean()
            atr_recent = atr_series.iloc[-1]
            atr_prior = atr_series.iloc[-(comp_lb + 1)] if n >= 14 + comp_lb + 1 else atr_recent
            if atr_prior is not None and not pd.isna(atr_prior) and atr_prior > 1e-10:
                ratio = float(atr_recent / atr_prior) if not pd.isna(atr_recent) else 1.0
                features["volatility_compression_score"] = max(0.0, min(1.0, 1.0 - (ratio - 0.5) / 1.0))
            else:
                features["volatility_compression_score"] = 0.5
        else:
            features["volatility_compression_score"] = None

        # --- GC/DC and post-cross flags ---
        if all(v is not None for v in [last_ma_s, last_ma_m]) and n >= sw + 2:
            prev_ma_s = ma_short.iloc[-2] if not pd.isna(ma_short.iloc[-2]) else None
            prev_ma_m = ma_mid.iloc[-2] if not pd.isna(ma_mid.iloc[-2]) else None

            if prev_ma_s is not None and prev_ma_m is not None:
                # Golden Cross: short MA crossed above mid MA recently
                gc_just_happened = prev_ma_s <= prev_ma_m and last_ma_s > last_ma_m
                # Death Cross: short MA crossed below mid MA recently
                dc_just_happened = prev_ma_s >= prev_ma_m and last_ma_s < last_ma_m

                # Check last N bars for any cross
                recent_cross = gc_just_happened or dc_just_happened
                if not recent_cross and n >= sw + conv_lb:
                    for i in range(2, min(conv_lb + 1, n - sw)):
                        s1 = ma_short.iloc[-(i + 1)]
                        m1 = ma_mid.iloc[-(i + 1)]
                        s2 = ma_short.iloc[-i]
                        m2 = ma_mid.iloc[-i]
                        if not any(pd.isna(x) for x in [s1, m1, s2, m2]):
                            if (s1 <= m1 and s2 > m2) or (s1 >= m1 and s2 < m2):
                                recent_cross = True
                                break

                # post_cross_expansion_flag: GC/DC 直後で MA がまだ大きく開いている
                ma_spread_pct = features.get("ma_spread_pct", 0) or 0
                features["post_cross_expansion_flag"] = (
                    recent_cross and ma_spread_pct > 0.01
                )

                # post_cross_consolidation_flag: GC/DC 後にいったん収束した
                conv_score = features.get("ma_convergence_score", 0) or 0
                features["post_cross_consolidation_flag"] = (
                    recent_cross and conv_score > 0.6
                )
            else:
                features["post_cross_expansion_flag"] = False
                features["post_cross_consolidation_flag"] = False
        else:
            features["post_cross_expansion_flag"] = False
            features["post_cross_consolidation_flag"] = False

        # --- Squeeze breakout ready ---
        # MA 収束 + 価格圧縮 + ボラ縮小 → すべて閾値以上なら True
        conv_score = features.get("ma_convergence_score")
        rng_comp = features.get("range_compression_score")
        vol_comp = features.get("volatility_compression_score")
        if all(v is not None for v in [conv_score, rng_comp, vol_comp]):
            features["squeeze_breakout_ready"] = (
                conv_score >= 0.55 and rng_comp >= 0.50 and vol_comp >= 0.50
            )
        else:
            features["squeeze_breakout_ready"] = False

        return features

    # ------------------------------------------------------------------
    # Column normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case column names and validate required columns."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        # Drop duplicate columns (keep first)
        df = df.loc[:, ~df.columns.duplicated()]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            logger.error("Missing required columns: {}", missing)
            return pd.DataFrame()
        # Ensure numeric
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=list(required), inplace=True)
        return df

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    def _technical(self, df: pd.DataFrame) -> dict[str, Any]:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        features: dict[str, Any] = {}

        # Simple Moving Averages
        for period in (5, 20, 50):
            sma = close.rolling(window=period, min_periods=period).mean()
            features[f"sma_{period}"] = _last(sma)

        # Exponential Moving Averages
        for period in (9, 21):
            ema = close.ewm(span=period, adjust=False).mean()
            features[f"ema_{period}"] = _last(ema)

        # RSI
        for period in (5, 14):
            features[f"rsi_{period}"] = self._rsi(close, period)

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        features["macd"] = _last(macd_line)
        features["macd_signal"] = _last(signal_line)
        features["macd_histogram"] = _last(histogram)

        # Bollinger Bands (20, 2)
        sma20 = close.rolling(window=20, min_periods=20).mean()
        std20 = close.rolling(window=20, min_periods=20).std()
        features["bb_upper"] = _last(sma20 + 2 * std20)
        features["bb_middle"] = _last(sma20)
        features["bb_lower"] = _last(sma20 - 2 * std20)
        bb_upper_val = features["bb_upper"]
        bb_lower_val = features["bb_lower"]
        if bb_upper_val is not None and bb_lower_val is not None and bb_upper_val != bb_lower_val:
            features["bb_pct"] = (close.iloc[-1] - bb_lower_val) / (bb_upper_val - bb_lower_val)
        else:
            features["bb_pct"] = None

        # ATR (14)
        features["atr_14"] = self._atr(high, low, close, 14)

        return features

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------

    def _volume(self, df: pd.DataFrame) -> dict[str, Any]:
        vol = df["volume"]
        close = df["close"]
        features: dict[str, Any] = {}

        # Volume ratio vs 20-day average
        vol_ma20 = vol.rolling(window=20, min_periods=1).mean()
        last_vol = vol.iloc[-1]
        last_vol_ma20 = vol_ma20.iloc[-1]
        features["volume_ratio"] = (
            last_vol / last_vol_ma20 if last_vol_ma20 > 0 else None
        )

        # Volume MA (5)
        vol_ma5 = vol.rolling(window=5, min_periods=1).mean()
        features["volume_ma_5"] = _last(vol_ma5)

        # On-Balance Volume
        obv = self._obv(close, vol)
        features["obv"] = _last(obv)

        return features

    # ------------------------------------------------------------------
    # VWAP features
    # ------------------------------------------------------------------

    def _vwap(self, df: pd.DataFrame) -> dict[str, Any]:
        features: dict[str, Any] = {}
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

        features["vwap"] = _last(vwap_series)

        last_close = df["close"].iloc[-1]
        last_vwap = features["vwap"]
        if last_vwap is not None and last_vwap > 0:
            features["distance_from_vwap"] = (last_close - last_vwap) / last_vwap
        else:
            features["distance_from_vwap"] = None

        # Fraction of bars where close was above VWAP
        above = (df["close"] > vwap_series).sum()
        total = len(df)
        features["time_above_vwap"] = above / total if total > 0 else None

        return features

    # ------------------------------------------------------------------
    # Price features
    # ------------------------------------------------------------------

    def _price(self, df: pd.DataFrame) -> dict[str, Any]:
        features: dict[str, Any] = {}

        # Gap percentage (today open vs previous close)
        if len(df) >= 2:
            prev_close = df["close"].iloc[-2]
            today_open = df["open"].iloc[-1]
            features["gap_pct"] = (
                (today_open - prev_close) / prev_close if prev_close > 0 else 0.0
            )
        else:
            features["gap_pct"] = 0.0

        # Intraday range (current bar)
        last_high = df["high"].iloc[-1]
        last_low = df["low"].iloc[-1]
        last_close = df["close"].iloc[-1]
        features["intraday_range"] = (
            (last_high - last_low) / last_close if last_close > 0 else 0.0
        )

        # Distance from recent high / low (20-day)
        recent_high = df["high"].tail(20).max()
        recent_low = df["low"].tail(20).min()
        features["price_vs_high"] = (
            (last_close - recent_high) / recent_high if recent_high > 0 else 0.0
        )
        features["price_vs_low"] = (
            (last_close - recent_low) / recent_low if recent_low > 0 else 0.0
        )

        return features

    # ------------------------------------------------------------------
    # Candlestick patterns
    # ------------------------------------------------------------------

    def _pattern(self, df: pd.DataFrame) -> dict[str, Any]:
        features: dict[str, Any] = {}

        o = df["open"].iloc[-1]
        h = df["high"].iloc[-1]
        l = df["low"].iloc[-1]  # noqa: E741
        c = df["close"].iloc[-1]

        body = abs(c - o)
        full_range = h - l if h != l else 1e-9
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        # Doji: very small body relative to range
        features["is_doji"] = body / full_range < 0.1

        # Hammer: small body at top, long lower shadow
        features["is_hammer"] = (
            lower_shadow > 2 * body and upper_shadow < body * 0.5 and body > 0
        )

        # Inverted hammer
        features["is_inverted_hammer"] = (
            upper_shadow > 2 * body and lower_shadow < body * 0.5 and body > 0
        )

        # Bullish / bearish engulfing (need at least 2 bars)
        if len(df) >= 2:
            prev_o = df["open"].iloc[-2]
            prev_c = df["close"].iloc[-2]
            features["is_bullish_engulfing"] = (
                prev_c < prev_o  # previous bearish
                and c > o  # current bullish
                and o <= prev_c  # opens below prev close
                and c >= prev_o  # closes above prev open
            )
            features["is_bearish_engulfing"] = (
                prev_c > prev_o  # previous bullish
                and c < o  # current bearish
                and o >= prev_c  # opens above prev close
                and c <= prev_o  # closes below prev open
            )
        else:
            features["is_bullish_engulfing"] = False
            features["is_bearish_engulfing"] = False

        # Aggregate pattern label
        if features["is_bullish_engulfing"]:
            features["candle_pattern"] = "bullish_engulfing"
        elif features["is_bearish_engulfing"]:
            features["candle_pattern"] = "bearish_engulfing"
        elif features["is_hammer"]:
            features["candle_pattern"] = "hammer"
        elif features["is_inverted_hammer"]:
            features["candle_pattern"] = "inverted_hammer"
        elif features["is_doji"]:
            features["candle_pattern"] = "doji"
        else:
            features["candle_pattern"] = "none"

        return features

    # ------------------------------------------------------------------
    # Momentum features
    # ------------------------------------------------------------------

    def _momentum(self, df: pd.DataFrame) -> dict[str, Any]:
        close = df["close"]
        features: dict[str, Any] = {}

        # Rate of change
        for period in (5, 10):
            if len(close) > period:
                prev = close.iloc[-period - 1]
                features[f"roc_{period}"] = (
                    (close.iloc[-1] - prev) / prev if prev > 0 else 0.0
                )
            else:
                features[f"roc_{period}"] = None

        # Momentum (raw price difference over 10 bars)
        if len(close) > 10:
            features["momentum_10"] = close.iloc[-1] - close.iloc[-11]
        else:
            features["momentum_10"] = None

        return features

    # ------------------------------------------------------------------
    # Internal calculation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> float | None:
        """Compute RSI using the Wilder smoothing method."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return _last(rsi)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float | None:
        """Average True Range."""
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return _last(atr)

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _last(series: pd.Series) -> float | None:
    """Return the last non-NaN value of a series, or None."""
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)
