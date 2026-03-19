# KabuAI デイトレード

AI駆動型の日本株デイトレード学習システム。
ペーパートレードとバックテストで勝ちパターンを自動発見し、戦略を継続的に改善する。

## v2 改修内容 (2026-03-19)

- **random.shuffle 廃止** → PreMarketScanner + StockSelector で知的銘柄選定
- **simulate_current_price 廃止** → 日足終値ベースの判断に統一（擬似intraday依存を排除）
- **戦略階層化**: VWAP Reclaim(主) / ORB Continuation(補) / TrendFollow(フィルタ) / SpreadEntry(補助)
- **RegimeDetector 全戦略統合**: レジーム別に確信度調整
- **戦略auto on/off**: ローリングPF/勝率で自動停止・再開
- **銘柄相性学習**: strategy×ticker 勝率をDB蓄積し確信度に反映
- **異常検知強化**: 4連敗 / 急速ドローダウン(-3%) で自動停止
- **バックテスト改善**: intraday S/L・TP判定 + 含み損決済
- **UI拡張**: ウォッチリスト(採用理由付き) / レジーム / 戦略on/off / 異常停止表示

## 仕組み

```
市場開場中 (09:00-15:30 JST)
  └─ run_paper.py が常時稼働
       ├─ 08:55 TDnet適時開示を事前取得
       ├─ 09:00 ウォッチリスト構築 (PreMarketScanner + StockSelector)
       │        レジーム判定 / 戦略auto on/off / 日次リセット
       ├─ 09:00~ トレード開始（90秒間隔スキャン）
       │    ├─ 戦略階層に基づくシグナル優先度付け
       │    ├─ トレンドフィルタ / スプレッド補助 / 銘柄相性 を適用
       │    └─ 異常検知: 連敗・ドローダウンで自動停止
       ├─ 15:20 全ポジション強制決済
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

## 戦略階層

| 役割 | 戦略 | 説明 |
|------|------|------|
| **主戦略** | **VWAP Reclaim** | VWAP奪回 + イベント重み付け。最も安定 |
| 補助(継続型) | ORB Continuation | 前日バーの方向と一致するブレイクアウトのみ |
| フィルタ | Trend Follow | EMA/VWAP整列で他戦略の確信度を調整 |
| 補助シグナル | Spread Entry | スプレッド縮小時に他戦略にブースト付与 |
| イベント | TDnet / 決算 / 材料初動 | カタリストドリブン |
| その他 | Gap Go/Fade, VWAP Bounce, 逆張り系 | 通常稼働 |
| **無効化** | ~~Imbalance / 大口吸収 / Open Drive~~ | PF低下で停止 |

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
