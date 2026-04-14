# KabuAI フィードバックサマリー
生成日時: 2026-04-14T10:10:50

## 全体成績
| 指標 | 全体 | IS | OOS |
|------|------|----|----|
| トレード数 | 363 | 192 | 171 |
| 勝率 | 45.2% | 43.2% | 47.4% |
| PF | 0.92 | 0.67 | 1.19 |
| 損益 | -6,052 | -12,846 | +6,794 |
| MaxDD | 17,326 (441.2%) | - | - |

## 戦略別 OOS 成績
- **open_drive**: OOS 34件 WR=32% PF=0.69 -1,416円
- **spread_entry**: OOS 76件 WR=46% PF=1.23 +4,695円
- **trend_follow**: OOS 20件 WR=60% PF=2.00 +2,274円
- **vwap_bounce**: OOS 2件 WR=100% PF=0.00 +514円
- **vwap_reclaim**: OOS 39件 WR=54% PF=1.08 +727円

## 問題点トップ5
1. 弱い戦略: open_drive OOS PF=0.69
2. 最大連敗: 10連敗

## 強い戦略
- spread_entry: OOS PF=1.23 (76件)
- trend_follow: OOS PF=2.00 (20件)

## データ品質
- 擬似特徴量使用率: 10.3%
- 使用中の擬似特徴量: bid_ask_ratio, depth_imbalance, opening_range_high, opening_range_low, price_compression

## 注意事項
- 本システムはintraday proxy (日足+擬似特徴量) で動作
- ORB/spread/orderbook系の評価信頼度は限定的
- OOS PF > IS PF の場合はサンプルサイズ不足の可能性
- 収束系特徴量は日足から直接計算 (proxy ではない) だが intraday 精度は限定的

## Net Exposure
- 方向: Long 298 / Short 65 (バイアス: long偏重)
- 戦略集中度: 34%

## 収束フィルタ分析 (v3.3)
- 収束後エントリー: 147件 WR=46% PF=0.87 avg=-26円
- 拡散エントリー: 216件 WR=44% PF=0.95 avg=-10円
- GC/DC直後拡散: 267件 WR=43%
- GC/DC後収束: 243件 WR=43%
- 拡散飛び乗り損失率: 56%

## 後場 PM-VWAP reclaim
- PM reclaim トレード数: 0
- PM reclaim PF: 0.00
- OOS PM PF: 0.00
- 低位株 bonus 適用: 12件

## 次に改善すべき観点
1. OOS PF が低い戦略のパラメータ見直し or 停止
2. 過学習が疑われる戦略の条件厳格化
3. 擬似特徴量依存の低減 (分足データ導入検討)
4. ストップロス幅の再検討
5. 銘柄選定フィルタの見直し
6. 収束フィルタ閾値の調整 (MAX_MA_SPREAD_PCT, MIN_CONVERGENCE_SCORE 等)
