# KabuAI デイトレード

AI駆動型の日本株デイトレード学習システム。
ペーパートレードとバックテストで勝ちパターンを自動発見し、戦略を継続的に改善する。

## v3.1 最適化 (2026-03-20)

**方針: 「増やす」ではなく「削る」「止める」「絞る」+ proxy 依存ペナルティ**

- **戦略を16→1に削減**: vwap_reclaim のみ active (他15は off/watch/filter/supplement)
- **proxy feature フラグ付け**: 擬似特徴量に is_proxy フラグを付与、戦略ごとに proxy_usage_rate を算出
- **proxy 依存ペナルティ**: proxy_usage_rate に応じて confidence を最大 -0.15 減点
- **vwap_reclaim 保守化**: time_below 10→15, vol_reclaim 1.0→1.5, target 1.8→1.2 ATR
- **戦略 status 体系**: active / filter / supplement / watch / off を明示化
- **UI で proxy 依存度・data quality warning を表示**
- **backtest_summary に proxy_summary・overfitting warning を出力**

### 運用パラメータ

| 項目 | 値 | 理由 |
|------|-----|------|
| MAX_POSITIONS | 2 | リスク集中回避 |
| POSITION_SIZE | 25万円 | 1件あたり損失半減 |
| MAX_LOSS_PER_DAY | -1.5万円 | 資金の0.5%で停止 |
| MIN_CONFIDENCE | 0.65 | 高確信シグナルのみ |
| SCAN_INTERVAL | 300秒 | 無駄打ち抑制 |
| TOP_UNIVERSE | 8 | 流動性上位に集中 |
| FORCE_CLOSE_TIME | 14:50 | 余裕を持って決済 |

**重要な制約事項:**
- 本システムは intraday proxy (日足 + 擬似特徴量) で動作している
- proxy feature 依存が高い戦略は評価信頼度が低い
- 当面の主戦略は vwap_reclaim (proxy_usage_rate=1.0 だが他戦略より相対的に最良)
- optimization_results の順位を鵜呑みにしないこと (proxy 由来の見かけの強さがある)
- ORB / spread / orderbook 系は proxy 依存が高く off/watch に降格済み

## 仕組み

```
市場開場中 (09:00-15:30 JST)
  └─ run_paper.py が常時稼働
       ├─ 08:55 TDnet適時開示を事前取得
       ├─ 09:00 ウォッチリスト構築
       │    ├─ PreMarketScanner + StockSelector (流動性10億円+、上位8銘柄)
       │    ├─ 米国セクターバイアス計算 (前日ETF 11本 → 日本33業種へ伝播)
       │    ├─ レジーム判定 / 戦略auto on/off / 日次リセット
       │    └─ セクターバイアスで watchlist スコアに +/-0.10 加点
       ├─ 09:00~ vwap_reclaim 主軸トレード（300秒間隔）
       │    ├─ trend_follow フィルタ通過時のみ発火許可
       │    ├─ セクターバイアスで confidence に +/-0.05 補助
       │    ├─ 同時保有2件 / 1件25万円 / MIN_CONFIDENCE 0.65
       │    └─ 異常検知: 3連敗 or 直近8件-2万円超で停止
       ├─ 14:50 全ポジション強制決済
       └─ 15:30 日次ナレッジ抽出・学習・サマリー出力

バックグラウンド
  ├─ run_backtest_learn.py
  │    ├─ 過去3ヶ月 x 60銘柄で全戦略をシミュレーション
  │    ├─ intraday S/L・TP判定 + 含み損決済
  │    ├─ In-Sample / Out-of-Sample 分離で過学習を検出
  │    ├─ レジーム x 戦略 マトリクスで評価
  │    └─ Claude Code向けフィードバックパッケージ自動生成
  │         ├─ feedback_packet.json (メトリクス+異常+推奨ヒント)
  │         ├─ feedback_summary.md (LLMにコピペ可能なサマリ)
  │         └─ plots/ (equity curve, drawdown, OOS PF, regime heatmap等)
  │
  └─ run_optimize.py
       ├─ パラメータ空間をランダムサーチ (OOS PFで評価)
       └─ 最適パラメータをJSON保存

ダッシュボード (http://localhost:8502)
  └─ SPY前日・セクターバイアス・ウォッチリスト・レジーム
     戦略on/off・異常停止・トレード履歴 / 30秒自動更新
```

## 戦略構成 (v3.1)

| Status | 戦略 | Proxy率 | 説明 |
|--------|------|---------|------|
| **Active** | **vwap_reclaim** | 100% | 唯一の主戦略。proxy 依存だが他戦略より相対的に最良 |
| Filter | trend_follow | 0% | is_trending() で vwap_reclaim のゲートに使用。EMA/VWAP は real |
| Supplement | spread_entry | 100% | get_spread_boost() で微小ブースト (max 0.05)。単独発火しない |
| Watch | orb | 100% | intraday proxy 信頼性不足。将来再開用 |
| Watch | tdnet_event | 0% | イベント時参考。TDnet は real データ |
| Watch | gap_go | 0% | 擬似特徴量依存 |
| **Off** | 他10戦略 | 50-100% | 擬似intraday依存 / データ不足 / PF低下 |

## セットアップ

```bash
cd ~/dev/KabuAI-Daytrade
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# KABUAI_JQUANTS_API_KEY=your_key を記入
make setup
```

## 使い方

### 全部起動（推奨）
```bash
source .venv/bin/activate

# ペーパートレード（市場時間中に自動売買・学習）
nohup .venv/bin/python run_paper.py >> logs/paper_stdout.log 2>&1 &

# バックテスト学習（過去データで一括学習）
nohup .venv/bin/python run_backtest_learn.py >> logs/backtest_stdout.log 2>&1 &

# 戦略パラメータ最適化
nohup .venv/bin/python run_optimize.py > logs/optimize_stdout.log 2>&1 &

# ダッシュボード
streamlit run ui/app.py --server.port 8502 --theme.base dark
```

### ログ確認
```bash
tail -f logs/paper_trade.log                      # ペーパートレード
tail -f logs/optimize_stdout.log                 # 最適化進捗
cat knowledge/backtest_summary.json              # バックテスト結果
cat knowledge/paper_state.json                   # 稼働状態（UI用）
cat knowledge/sector_bias.json                   # 米国→日本セクターバイアス
cat knowledge/feedback_summary.md                # Claude Code向け改善サマリ
cat knowledge/feedback_packet.json               # 全メトリクス+推奨ヒント
ls  knowledge/plots/                             # 可視化チャート6枚
```

### Claude Code に改善を依頼する時
```bash
# バックテスト実行 → フィードバック自動生成
.venv/bin/python run_backtest_learn.py

# Claude Code に渡す
cat knowledge/feedback_summary.md    # これをコピペして「改善して」と伝える
```

## 自動化機能

| 機能 | 説明 |
|------|------|
| **銘柄選定** | PreMarketScanner(ギャップ・出来高・イベント) + StockSelector(売買代金10億+/出来高50万株+) |
| **米国セクターバイアス** | 前日の米国ETF11本の騰落率→日本33業種に伝播。watchlistに+/-0.10、confidenceに+/-0.05 |
| **レジーム判定** | trend_up/down, range, volatile, low_vol を日足SMA/ATR/BBから判定 |
| **戦略auto on/off** | ローリングPF < 0.9 or 勝率 < 40% で停止、PF > 1.2 and 勝率 > 48% で再開 |
| **銘柄相性学習** | strategy x ticker の勝率をDB蓄積、確信度に反映 |
| **異常検知** | 3連敗 / 直近8件で-2万円超ドローダウン → 自動停止 |
| **ナレッジ抽出** | 勝ち/負けパターンを自動マイニング |
| **改善フィードバック** | バックテスト後にJSON+Markdown+6plotsを自動生成。過学習警告・弱い戦略・データ品質を可視化 |

## アーキテクチャ

```
├── core/           設定・モデル・安全ガード・銘柄名マップ
├── db/             SQLite（トレード・ナレッジ・銘柄相性・戦略成績）
├── data_sources/   J-Quants API V2 + TDnet適時開示
├── brokers/        ペーパーブローカー / 立花証券API（将来）
├── scanners/       PreMarketScanner + StockSelector + ScoreEngine
├── strategies/     16戦略（vwap_reclaim主軸、他はfilter/supplement/off）
├── execution/      実行エンジン・リスク管理
├── analytics/      ナレッジ抽出・学習ループ・日次レポート・フィードバックパッケージ生成
├── tools/          特徴量計算・レジーム判定・セクターバイアス・コスト計算・バックテスト
├── ui/             Streamlitダッシュボード（セクターバイアス・ウォッチリスト・レジーム）
├── knowledge/      backtest_summary.json / feedback_packet.json / feedback_summary.md / plots/
└── tests/          ユニットテスト（73件全通過）
```

## データソース

| ソース | 状態 | 用途 |
|--------|------|------|
| J-Quants API V2 | 接続済み | 上場銘柄・日足・財務データ・33業種コード |
| TDnet | 接続済み | 適時開示（上方修正・自社株買い等の重要イベント） |
| Yahoo Finance Chart API | 接続済み | 米国セクターETF前日騰落率 (XLK/XLF/XLE等11本 + SPY) |
| 立花証券API | 将来実装 | 本番執行（リアルタイム板・約定・発注） |

## 安全設計

- `ALLOW_LIVE_TRADING=false` 固定（本番取引禁止）
- 日次損失上限 **-1.5万円** (資金の0.5%) で自動停止
- **3連敗** / 直近8件で **-2万円超** ドローダウンで異常停止
- 戦略auto on/off（PF < 0.9 / 勝率 < 40% で停止）
- **14:50** に全ポジション強制決済
- 同時保有 **最大2件** / 1件あたり **25万円**
- 市場時間外は自動待機（週末・祝日スキップ）
- 重複注文防止 / コスト込み評価

## ライセンス

MIT
