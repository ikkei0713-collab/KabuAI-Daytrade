# KabuAI フィードバックサマリー
生成日時: 2026-03-30T10:43:02

## 全体成績
| 指標 | 全体 | IS | OOS |
|------|------|----|----|
| トレード数 | 806 | 378 | 428 |
| 勝率 | 44.3% | 41.0% | 47.2% |
| PF | 0.91 | 0.68 | 1.14 |
| 損益 | -14,719 | -26,415 | +11,696 |
| MaxDD | 36,125 (421.1%) | - | - |

## 戦略別 OOS 成績
- **open_drive**: OOS 34件 WR=26% PF=0.68 -1,536円
- **orderbook_imbalance**: OOS 317件 WR=47% PF=1.02 +1,557円
- **spread_entry**: OOS 34件 WR=53% PF=1.92 +7,597円
- **trend_follow**: OOS 20件 WR=60% PF=2.22 +2,788円
- **vwap_bounce**: OOS 2件 WR=100% PF=0.00 +514円
- **vwap_reclaim**: OOS 21件 WR=52% PF=1.16 +776円

## 問題点トップ5
1. 弱い戦略: open_drive OOS PF=0.68
2. 最大連敗: 10連敗

## 強い戦略
- spread_entry: OOS PF=1.92 (34件)
- trend_follow: OOS PF=2.22 (20件)

## データ品質
- 擬似特徴量使用率: 10.3%
- 使用中の擬似特徴量: bid_ask_ratio, depth_imbalance, opening_range_high, opening_range_low, price_compression

## 注意事項
- 本システムはintraday proxy (日足+擬似特徴量) で動作
- ORB/spread/orderbook系の評価信頼度は限定的
- OOS PF > IS PF の場合はサンプルサイズ不足の可能性
- 収束系特徴量は日足から直接計算 (proxy ではない) だが intraday 精度は限定的

## Net Exposure
- 方向: Long 376 / Short 430 (バイアス: 中立)
- 戦略集中度: 69%

## 収束フィルタ分析 (v3.3)
- 収束後エントリー: 282件 WR=44% PF=0.83 avg=-35円
- 拡散エントリー: 524件 WR=45% PF=0.96 avg=-9円
- GC/DC直後拡散: 569件 WR=45%
- GC/DC後収束: 521件 WR=44%
- 拡散飛び乗り損失率: 55%

## 後場 PM-VWAP reclaim
- PM reclaim トレード数: 0
- PM reclaim PF: 0.00
- OOS PM PF: 0.00
- 低位株 bonus 適用: 16件

## 次に改善すべき観点
1. OOS PF が低い戦略のパラメータ見直し or 停止
2. 過学習が疑われる戦略の条件厳格化
3. 擬似特徴量依存の低減 (分足データ導入検討)
4. ストップロス幅の再検討
5. 銘柄選定フィルタの見直し
6. 収束フィルタ閾値の調整 (MAX_MA_SPREAD_PCT, MIN_CONVERGENCE_SCORE 等)
