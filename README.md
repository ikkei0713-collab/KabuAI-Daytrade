# KabuAI デイトレード

AI駆動型の日本株デイトレード学習システム。
ペーパートレードとバックテストで勝ちパターンを自動発見し、戦略を継続的に改善する。

## 仕組み

```
市場開場中 (09:00-15:00)
  └─ run_paper.py が常時稼働
       ├─ 90秒ごとにJ-Quantsから実データ取得
       ├─ TDnet適時開示を毎朝取得（上方修正・自社株買い等）
       ├─ 銘柄スコアリング → 16戦略でスキャン → ペーパー売買
       ├─ 14:50に全ポジション強制決済
       └─ 15:00に日次ナレッジ抽出・学習

バックグラウンド
  └─ run_backtest_learn.py
       ├─ 過去3ヶ月の実データで全戦略をシミュレーション
       ├─ コスト込み（スプレッド・スリッページ）で現実的に評価
       ├─ In-Sample / Out-of-Sample 分離で過学習を検出
       ├─ レジーム別（トレンド/レンジ/高ボラ）に戦略を評価
       └─ 勝ちパターン・負けパターンを自動抽出してDBに蓄積

ダッシュボード (http://localhost:8502)
  └─ リアルタイムで稼働状況・トレード履歴・ナレッジを確認
```

## 最新バックテスト結果

180トレード / 28銘柄 / コスト調整後:

| 指標 | 全体 | In-Sample (60%) | Out-of-Sample (40%) |
|------|------|-----------------|---------------------|
| トレード数 | 180 | 135 | 45 |
| 勝率 | 46.7% | 47.4% | 44.4% |
| PF | 0.95 | 1.09 | 0.73 |
| 損益(コスト後) | -51K | +62K | -113K |

**有効な戦略:**
- `vwap_reclaim`: IS 60% → OOS 60%（安定、PF 2.65）
- `spread_entry`: IS 58%（OOS劣化あり、要改善）

**無効化推奨:**
- `orderbook_imbalance`: 119件中44%勝率、ダミーデータ依存
- `open_drive`: 35%勝率、PF 0.23

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

# ペーパートレード（市場時間中に自動売買）
nohup python run_paper.py >> logs/paper_stdout.log 2>&1 &

# バックテスト学習（バックグラウンドで高速実行）
nohup python run_backtest_learn.py >> logs/backtest_stdout.log 2>&1 &

# ダッシュボード
streamlit run ui/app.py --server.port 8502 --theme.base dark
```

### ログ確認
```bash
# ペーパートレードのリアルタイムログ
tail -f logs/paper_stdout.log

# バックテスト結果
cat knowledge/backtest_summary.json
```

### 個別実行
```bash
make ui          # ダッシュボードのみ
make trade       # トレードエンジン
make backtest    # バックテスト
make test        # テスト
```

## 搭載戦略（16種）

| カテゴリ | 戦略 | 概要 | 評価 |
|---------|------|------|------|
| 寄付 | **ORB** | 5分足レンジブレイクアウト（主戦略） | 改善中 |
| モメンタム | **VWAP Reclaim** | VWAP下落後の奪還 | ★最優秀 |
| モメンタム | VWAP Bounce | VWAPサポートバウンス | 有効 |
| モメンタム | Trend Follow | EMA9>EMA21トレンド追従 | 有効 |
| 板需給 | Spread Entry | スプレッド縮小ブレイク | 要改善 |
| ギャップ | Gap Go | GU 2%超→ブレイク | 検証中 |
| ギャップ | Gap Fade | GU失速フェード | 検証中 |
| 寄付 | Open Drive | 寄付ドライブ | 非推奨 |
| 逆張り | Overextension / RSI / 急落リバウンド | 各種逆張り | 条件付き |
| 板需給 | Imbalance / 大口吸収 | 板情報系 | 非推奨 |
| イベント | TDnet / 決算 / 材料初動 | イベントドリブン | 検証中 |

## アーキテクチャ

```
├── core/           設定・モデル・安全ガード・コストモデル
├── db/             SQLite（トレード・ナレッジ・戦略成績）
├── data_sources/   J-Quants API V2 + TDnet適時開示
├── brokers/        ペーパーブローカー / 立花証券API（将来）
├── scanners/       銘柄スコアリング（Stocks in Play）
├── strategies/     16戦略（BaseStrategy統一インターフェース）
├── execution/      実行エンジン・リスク管理
├── analytics/      ナレッジ抽出・学習ループ
├── tools/          特徴量計算・レジーム判定・バックテスト・コスト計算
├── ui/             Streamlitダッシュボード（リアルDB表示）
└── tests/          ユニットテスト
```

## データソース

| ソース | 状態 | 用途 |
|--------|------|------|
| J-Quants API V2 | 接続済み | 上場銘柄・日足・財務データ |
| TDnet | 接続済み | 適時開示（上方修正・自社株買い等） |
| 立花証券API | 将来実装 | 本番執行（リアルタイム板・約定） |

## 安全設計

- `ALLOW_LIVE_TRADING=false` 固定（本番取引禁止）
- 日次損失上限 -5万円で自動停止
- 14:50に全ポジション強制決済
- 市場時間外は自動待機（週末スキップ）
- 重複注文防止 / Idempotent設計

## ライセンス

Private - 個人利用限定
