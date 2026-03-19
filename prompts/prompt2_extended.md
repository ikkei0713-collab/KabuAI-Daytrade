# Prompt2: 基本+追加テキストフィードバック

Prompt1 の全情報に加え、以下の追加情報を含む。

## 追加情報1: 特徴量統計 (Feature Statistics)
各特徴量の count, mean, std, min, p1, p5, p50, p95, p99, max, skew, kurtosis, missing_ratio, zero_ratio。
proxy特徴量には is_proxy=true フラグが付く。

{feature_statistics_table}

## 追加情報2: IC/ICIR (近似)
- IC (Spearman): selector_score と realized return の順位相関
- ICIR: rolling IC の mean/std
- 注: 厳密な因子ICではなく近似値

| 指標 | 値 |
|------|-----|
| IC | {ic} |
| ICIR | {icir} |
| p-value | {ic_p_value} |
| サンプル数 | {n_samples} |

## 追加情報3: Net Exposure
- Long/Short 件数: {long_count} / {short_count}
- 方向バイアス: {direction_bias} ({direction_bias_label})
- 戦略集中度: {strategy_concentration}

## 使い方
1. `cat knowledge/feedback_packet.json | python -m json.tool` で全データ取得
2. feature_statistics, ic_icir, net_exposure セクションの値を確認
3. 特に is_proxy=true の特徴量に注意
4. IC が低い場合、selector_score の有用性を疑うこと
