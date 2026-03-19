# KabuAI デイトレード

AI駆動型の日本株デイトレード学習システム。
ペーパートレードとバックテストで勝ちパターンを自動発見し、戦略を継続的に改善する。

## 仕組み

```
市場開場中 (09:00-15:30 JST)
  └─ run_paper.py が常時稼働
       ├─ 08:55 TDnet適時開示を事前取得
       ├─ 09:00 トレード開始（90秒間隔スキャン）
       ├─ 銘柄スコアリング(Stocks in Play) → 有効戦略でスキャン → ペーパー売買
       ├─ 15:20 全ポジション強制決済（クロージングオークション前）
       └─ 15:30 日次ナレッジ抽出・学習・サマリー出力

バックグラウンド
  ├─ run_backtest_learn.py
  │    ├─ 過去3ヶ月 × 60銘柄で全戦略をシミュレーション
  │    ├─ コスト込み（スプレッド・スリッページ・マーケットインパクト）
  │    ├─ In-Sample / Out-of-Sample 分離で過学習を検出
  │    ├─ レジーム別（trend_up/down, range, volatile, low_vol）に評価
  │    └─ 勝ち/負けパターンを自動抽出してDBに蓄積
  │
  └─ run_optimize.py
       ├─ パラメータ空間をランダムサーチ（5戦略 × 各3-5パラメータ）
       ├─ OOS PFで評価（過学習回避）
       └─ 最適パラメータをJSON保存

ダッシュボード (http://localhost:8502)
  └─ リアルDB表示（トレード履歴・戦略成績・ナレッジ・ログ・30秒自動更新）
```

## 最新バックテスト結果

赤字戦略無効化後 / 73トレード / 60銘柄 / コスト調整後:

| 指標 | 全体 | In-Sample (60%) | Out-of-Sample (40%) |
|------|------|-----------------|---------------------|
| トレード数 | 73 | 56 | 17 |
| 勝率 | **56.2%** | 58.9% | 47.1% |
| PF | **1.86** | 2.32 | 1.26 |
| 損益(コスト後) | **+252K** | +219K | +33K |

**有効な戦略（データ裏付けあり）:**
- `vwap_reclaim`: IS 56% / OOS 55%（最も安定、PF 1.34→1.71）
- `spread_entry`: IS 60%（OOS劣化あり → パラメータ最適化中）
- `trend_follow`: 1件のみだがPF高（データ蓄積中）
- `vwap_bounce`: 3件 67%勝率（データ蓄積中）

**無効化済み（赤字）:**
- `orderbook_imbalance`: 119件 44%勝率 PF 0.77（ダミーデータ依存）
- `large_absorption`: 同上
- `open_drive`: 17件 35%勝率 PF 0.23

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
nohup python run_paper.py >> logs/paper_stdout.log 2>&1 &

# バックテスト学習（過去データで一括学習）
nohup python run_backtest_learn.py >> logs/backtest_stdout.log 2>&1 &

# 戦略パラメータ最適化（指定時間ランダムサーチ）
nohup python run_optimize.py > logs/optimize_stdout.log 2>&1 &

# ダッシュボード
streamlit run ui/app.py --server.port 8502 --theme.base dark
```

### ログ確認
```bash
tail -f logs/paper_stdout.log                    # ペーパートレード
tail -f logs/optimize_stdout.log                 # 最適化進捗
grep "NEW BEST" logs/optimize_stdout.log         # 改善のみ
cat knowledge/backtest_summary.json              # バックテスト結果
cat knowledge/optimization_results.json          # 最適化結果
```

## 搭載戦略（16種 / 13有効）

| カテゴリ | 戦略 | 概要 | 状態 |
|---------|------|------|------|
| 寄付 | **ORB** | 5分足レンジブレイクアウト（主戦略） | 有効・最適化中 |
| モメンタム | **VWAP Reclaim** | VWAP下落後の奪還 | **★最優秀** |
| モメンタム | VWAP Bounce | VWAPサポートバウンス | 有効 |
| モメンタム | Trend Follow | EMA9>EMA21トレンド追従 | 有効 |
| 板需給 | Spread Entry | スプレッド縮小ブレイク | 有効・最適化中 |
| ギャップ | Gap Go / Gap Fade | ギャップ系 | 有効 |
| 逆張り | Overextension / RSI / 急落リバウンド | 各種逆張り | 有効（条件付き） |
| イベント | TDnet / 決算 / 材料初動 | イベントドリブン | 有効 |
| 板需給 | ~~Imbalance / 大口吸収~~ | 板情報系 | **無効化** |
| 寄付 | ~~Open Drive~~ | 寄付ドライブ | **無効化** |

## アーキテクチャ

```
├── core/           設定・モデル・安全ガード・コストモデル・銘柄名マップ
├── db/             SQLite（トレード・ナレッジ・戦略成績・改善候補）
├── data_sources/   J-Quants API V2 + TDnet適時開示スクレイピング
├── brokers/        ペーパーブローカー / 立花証券API（将来・本番執行用）
├── scanners/       銘柄スコアリング（Stocks in Play） + 戦略スコアリング
├── strategies/     16戦略（BaseStrategy統一IF、レジーム連動）
├── execution/      実行エンジン・リスク管理（セクター集中制限等）
├── analytics/      ナレッジ抽出・学習ループ・日次レポート
├── tools/          特徴量計算・レジーム判定・コスト計算・バックテスト
├── ui/             Streamlitダッシュボード（リアルDB・ログ・自動更新）
└── tests/          ユニットテスト（安全ガード・戦略・スコアエンジン）
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
- 15:20に全ポジション強制決済（クロージングオークション前）
- 市場時間外は自動待機（週末・祝日スキップ）
- 重複注文防止 / Idempotent設計
- コスト込み評価（スプレッド・スリッページ・インパクト）

## PC移行

```bash
# 旧PCからコピー（学習データ引き継ぎ）
scp db/kabuai.db 新PC:~/dev/KabuAI-Daytrade/db/
scp data/cache/ticker_names.json 新PC:~/dev/KabuAI-Daytrade/data/cache/
```

## ライセンス

MIT
