# Prompt3: テキスト+プロットフィードバック

Prompt2 の全情報に加え、以下のプロットを参照すること。

## プロット一覧 (knowledge/plots/)
| ファイル | 内容 | 論文対応 |
|----------|------|----------|
| equity_curve.png | 累計損益曲線 | Equity curve |
| drawdown_curve.png | ドローダウン曲線 | Drawdown |
| strategy_oos.png | 戦略別OOS PF | - |
| regime_heatmap.png | 戦略×レジーム PF ヒートマップ | - |
| confidence_vs_pnl.png | selector_score vs PnL | - |
| feature_win_loss_compare.png | 勝ち/負け特徴量比較 | - |
| cumulative_ic.png | 累積IC曲線 | Cumulative IC |
| net_exposure.png | Net Exposure (方向バイアス) | Net exposure |

## 使い方
1. feedback_summary.md を読む
2. 上記8枚のプロットを確認
3. 特に以下に注意:
   - equity_curve: 右肩上がりか？大きなドローダウンはないか？
   - cumulative_ic: IC が累積的に正か？selector_score は有効か？
   - net_exposure: 方向バイアスが偏りすぎていないか？
   - regime_heatmap: どのレジームで弱いか？
4. Claude Code に画像を見せながら改善を依頼

## 生成コマンド
```bash
python run_backtest_learn.py
# 自動的に knowledge/feedback_*.json/.md と knowledge/plots/*.png が生成される
```
