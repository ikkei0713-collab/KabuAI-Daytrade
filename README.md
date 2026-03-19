# KabuAI デイトレード

AI駆動型の日本株デイトレード学習システム。
ペーパートレードとバックテストで勝ちパターンを自動発見し、戦略を継続的に改善する。

## v3 保守的チューニング (2026-03-19)

**方針: 「増やす」ではなく「削る」「止める」「絞る」**

- **戦略を16→1に削減**: vwap_reclaim のみ active (他15は off/watch/filter)
- **同時保有 5→2**: リスク集中を回避
- **ポジションサイズ 50万→25万**: 1件あたり損失を半減
- **日次損失上限 -5万→-1.5万**: 資金の0.5%で停止
- **MIN_CONFIDENCE 0.3→0.65**: 高確信シグナルのみ通過
- **スキャン間隔 90秒→300秒**: 無駄打ち抑制
- **監視銘柄 20→8**: 流動性上位に集中 (売買代金10億円以上)
- **強制決済 15:20→14:50**: 余裕を持って決済
- **異常停止 3連敗 / 直近8件で-2万円超**: 早期に止める
- **vwap_reclaim**: max_distance 2.0→0.8%, target 1.5→1.2 ATR (飛びつき抑制)
- **trend_follow**: filter専用化 (単独発火せず、vwap_reclaimのゲートに使用)
- **spread_entry**: 補助専用化 (boost上限 0.15→0.05)
- **auto_toggle厳格化**: 停止 PF<0.9/WR<40%, 再開 PF>1.2/WR>48%

**重要な制約事項:**
- 本システムは intraday proxy (日足 + 擬似特徴量) で動作している
- ORB / spread / orderbook 系の評価信頼度は限定的
- 当面は vwap_reclaim 主軸で保守的に運用する

## 仕組み

```
市場開場中 (09:00-15:30 JST)
  └─ run_paper.py が常時稼働
       ├─ 08:55 TDnet適時開示を事前取得
       ├─ 09:00 ウォッチリスト構築 (PreMarketScanner + StockSelector)
       │        レジーム判定 / 戦略auto on/off / 日次リセット
       ├─ 09:00~ vwap_reclaim 主軸トレード（300秒間隔、上位8銘柄）
       │    ├─ trend_follow フィルタ通過時のみ発火許可
       │    ├─ 同時保有2件 / 1件25万円 / MIN_CONFIDENCE 0.65
       │    └─ 異常検知: 3連敗 or 直近8件-2万円超で停止
       ├─ 14:50 全ポジション強制決済
       └─ 15:30 日次ナレッジ抽出・学習・サマリー出力

バックグラウンド
  ├─ run_backtest_learn.py
  │    ├─ 過去3ヶ月 x 60銘柄で全戦略をシミュレーション
  │    ├─ intraday S/L・TP判定 + 含み損決済 (改善済)
  │    ├─ In-Sample / Out-of-Sample 分離で過学習を検出
  │    ├─ レジーム x 戦略 マトリクスで評価
  │    └─ 勝ち/負けパターンを自動抽出してDBに蓄積
  │
  └─ run_optimize.py
       ├─ パラメータ空間をランダムサーチ
       ├─ OOS PFで評価（過学習回避）
       └─ 最適パラメータをJSON保存

ダッシュボード (http://localhost:8502)
  └─ ウォッチリスト・レジーム・戦略on/off・異常停止・トレード履歴
     リアルDB表示 / 30秒自動更新
```

## 戦略構成 (v3)

| 状態 | 戦略 | 説明 |
|------|------|------|
| **Active** | **vwap_reclaim** | 唯一の主戦略。イベント重み + regime + trend filter |
| Filter | trend_follow | is_trending() で vwap_reclaim のゲートに使用 |
| Supplement | spread_entry | get_spread_boost() で微小ブースト (max 0.05) |
| Watch | orb, tdnet_event, gap_go | 将来再開用。現在は off |
| **Off** | 他10戦略 | 擬似intraday依存 / データ不足 / PF低下 |

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
```

## 自動化機能

| 機能 | 説明 |
|------|------|
| **銘柄選定** | PreMarketScanner(ギャップ・出来高・イベント) + StockSelector(スコアリング) |
| **レジーム判定** | trend_up/down, range, volatile, low_vol を日足SMA/ATR/BBから判定 |
| **戦略auto on/off** | ローリングPF < 0.7 or 勝率 < 30% で停止、回復で再開 |
| **銘柄相性学習** | strategy x ticker の勝率をDB蓄積、確信度に反映 |
| **異常検知** | 4連敗 / 直近10件で-3%超ドローダウン → 自動停止 |
| **ナレッジ抽出** | 勝ち/負けパターンを自動マイニング |

## アーキテクチャ

```
├── core/           設定・モデル・安全ガード・銘柄名マップ
├── db/             SQLite（トレード・ナレッジ・銘柄相性・戦略成績）
├── data_sources/   J-Quants API V2 + TDnet適時開示
├── brokers/        ペーパーブローカー / 立花証券API（将来）
├── scanners/       PreMarketScanner + StockSelector + ScoreEngine
├── strategies/     16戦略（階層化: 主/補/フィルタ/補助）
├── execution/      実行エンジン・リスク管理
├── analytics/      ナレッジ抽出・学習ループ・日次レポート
├── tools/          特徴量計算・レジーム判定・コスト計算・バックテスト
├── ui/             Streamlitダッシュボード（ウォッチリスト・レジーム・戦略on/off）
├── knowledge/      backtest_summary.json / paper_state.json
└── tests/          ユニットテスト（73件全通過）
```

## データソース

| ソース | 状態 | 用途 |
|--------|------|------|
| J-Quants API V2 | 接続済み | 上場銘柄・日足・財務データ |
| TDnet | 接続済み | 適時開示（上方修正・自社株買い等の重要イベント） |
| 立花証券API | 将来実装 | 本番執行（リアルタイム板・約定・発注） |

## 安全設計

- `ALLOW_LIVE_TRADING=false` 固定（本番取引禁止）
- 日次損失上限 -5万円で自動停止
- 4連敗 / 急速ドローダウンで異常停止
- 戦略auto on/off（PF劣化で自動停止）
- 15:20に全ポジション強制決済
- 市場時間外は自動待機（週末・祝日スキップ）
- 重複注文防止 / コスト込み評価

## ライセンス

MIT
