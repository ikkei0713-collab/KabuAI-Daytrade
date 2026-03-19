# Prompt1: 基本テキストフィードバック

以下はKabuAIデイトレードシステムのバックテスト結果です。
改善提案をしてください。

## 全体成績
- トレード数: {total_trades}
- 勝率: {win_rate}
- Profit Factor: {pf}
- Sharpe Ratio: {sharpe}
- 累計損益: {total_pnl}円
- Max Drawdown: {max_drawdown}円 ({max_drawdown_pct})

## In-Sample / Out-of-Sample
| 指標 | IS | OOS |
|------|----|----|
| トレード数 | {is_trades} | {oos_trades} |
| 勝率 | {is_wr} | {oos_wr} |
| PF | {is_pf} | {oos_pf} |
| 損益 | {is_pnl} | {oos_pnl} |

## 戦略別成績
{strategy_table}

## データ品質
- proxy特徴量使用率: {proxy_rate}
- 注意: 本システムはintraday proxy (日足+擬似特徴量) で動作

## 使い方
1. `cat knowledge/feedback_summary.md` の内容をここに貼り付け
2. 上記テンプレートの {変数} を実際の値に置換
3. Claude Code に「このフィードバックに基づいて改善して」と依頼
