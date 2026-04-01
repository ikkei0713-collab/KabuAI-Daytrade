# KabuAI フィードバックサマリー
生成日時: 2026-04-01T09:55:37

## 全体成績
| 指標 | 全体 | IS | OOS |
|------|------|----|----|
| トレード数 | 250 | 139 | 111 |
| 勝率 | 44.8% | 43.2% | 46.8% |
| PF | 1.03 | 0.70 | 1.47 |
| 損益 | +1,376 | -8,093 | +9,469 |
| MaxDD | 14,894 (250.0%) | - | - |

## 戦略別 OOS 成績
- **open_drive**: OOS 34件 WR=26% PF=0.68 -1,536円
- **spread_entry**: OOS 34件 WR=53% PF=1.92 +7,597円
- **trend_follow**: OOS 20件 WR=60% PF=1.93 +2,118円
- **vwap_bounce**: OOS 2件 WR=100% PF=0.00 +514円
- **vwap_reclaim**: OOS 21件 WR=52% PF=1.16 +776円

## 問題点トップ5
1. 弱い戦略: open_drive OOS PF=0.68
2. 最大連敗: 8連敗

## 強い戦略
- spread_entry: OOS PF=1.92 (34件)
- trend_follow: OOS PF=1.93 (20件)

## データ品質
- 擬似特徴量使用率: 10.3%
- 使用中の擬似特徴量: bid_ask_ratio, depth_imbalance, opening_range_high, opening_range_low, price_compression

## 注意事項
- 本システムはintraday proxy (日足+擬似特徴量) で動作
- ORB/spread/orderbook系の評価信頼度は限定的
- OOS PF > IS PF の場合はサンプルサイズ不足の可能性
- 収束系特徴量は日足から直接計算 (proxy ではない) だが intraday 精度は限定的

## Net Exposure
- 方向: Long 190 / Short 60 (バイアス: long偏重)
- 戦略集中度: 37%

## 収束フィルタ分析 (v3.3)
- 収束後エントリー: 99件 WR=46% PF=1.04 avg=+7円
- 拡散エントリー: 151件 WR=44% PF=1.02 avg=+5円
- GC/DC直後拡散: 191件 WR=44%
- GC/DC後収束: 177件 WR=43%
- **収束フィルタは有効** (収束後 PF > 拡散 PF)
- 拡散飛び乗り損失率: 56%

## 後場 PM-VWAP reclaim
- PM reclaim トレード数: 0
- PM reclaim PF: 0.00
- OOS PM PF: 0.00
- 低位株 bonus 適用: 7件

## 次に改善すべき観点
1. OOS PF が低い戦略のパラメータ見直し or 停止
2. 過学習が疑われる戦略の条件厳格化
3. 擬似特徴量依存の低減 (分足データ導入検討)
4. ストップロス幅の再検討
5. 銘柄選定フィルタの見直し
6. 収束フィルタ閾値の調整 (MAX_MA_SPREAD_PCT, MIN_CONVERGENCE_SCORE 等)
