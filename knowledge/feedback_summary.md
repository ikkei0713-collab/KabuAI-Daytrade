# KabuAI フィードバックサマリー
生成日時: 2026-03-20T02:09:46

## 全体成績
| 指標 | 全体 | IS | OOS |
|------|------|----|----|
| トレード数 | 18 | 13 | 5 |
| 勝率 | 66.7% | 76.9% | 40.0% |
| PF | 2.08 | 3.54 | 1.57 |
| 損益 | +88,567 | +54,126 | +34,441 |
| MaxDD | 60,587 (2549.3%) | - | - |

## 戦略別 OOS 成績
- **vwap_reclaim**: OOS 5件 WR=40% PF=1.57 +34,441円 **[過学習疑い]**

## 問題点トップ5
1. 過学習: vwap_reclaim IS PF=3.54 → OOS PF=1.57 (gap=1.97)

## 強い戦略
- vwap_reclaim: OOS PF=1.57 (5件)

## データ品質
- 擬似特徴量使用率: 14.6%
- 使用中の擬似特徴量: bid_ask_ratio, depth_imbalance, opening_range_high, opening_range_low, price_compression

## 注意事項
- 本システムはintraday proxy (日足+擬似特徴量) で動作
- ORB/spread/orderbook系の評価信頼度は限定的
- OOS PF > IS PF の場合はサンプルサイズ不足の可能性

## Net Exposure
- 方向: Long 18 / Short 0 (バイアス: long偏重)
- 戦略集中度: 100%

## 次に改善すべき観点
1. OOS PF が低い戦略のパラメータ見直し or 停止
2. 過学習が疑われる戦略の条件厳格化
3. 擬似特徴量依存の低減 (分足データ導入検討)
4. ストップロス幅の再検討
5. 銘柄選定フィルタの見直し
