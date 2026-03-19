# KabuAI デイトレード

AI駆動型の日本株デイトレード学習システム。ペーパートレードで勝ちパターンを自動発見し、戦略を継続的に改善する。

## 概要

16種類のデイトレ戦略を実装し、J-Quants APIの実データでバックテスト・ペーパートレードを実行。トレード結果からナレッジを自動抽出し、勝率とPF（プロフィットファクター）を改善し続ける。

**本番取引は禁止（`ALLOW_LIVE_TRADING=false`固定）**

## 搭載戦略（16種）

| カテゴリ | 戦略 | 概要 |
|---------|------|------|
| ギャップ | Gap Go | GU 2%超 → 5分足ブレイク |
| ギャップ | Gap Fade | GU 3%超の失速フェード |
| 寄付 | Open Drive | 寄付き方向ドライブ継続 |
| 寄付 | ORB | 5分足レンジブレイクアウト |
| モメンタム | VWAP Reclaim | VWAP下落後の奪還 |
| モメンタム | VWAP Bounce | VWAPサポートバウンス |
| モメンタム | Trend Follow | EMA9>EMA21トレンド追従 |
| 逆張り | Overextension | 3ATR超過の平均回帰 |
| 逆張り | RSI逆張り | RSI(5)極端値からの反転 |
| 逆張り | 急落リバウンド | 5%超急落後のリバウンド |
| 板需給 | Imbalance | 板の偏りトレード |
| 板需給 | 大口吸収 | 大口注文吸収検知 |
| 板需給 | スプレッド縮小 | スプレッド縮小ブレイク |
| イベント | TDnet | 上方修正・自社株買い |
| イベント | 決算モメンタム | 決算後モメンタム |
| イベント | 材料初動 | ニュース初動 |

## セットアップ

```bash
cd ~/dev/KabuAI-Daytrade

# Python仮想環境
python3.13 -m venv .venv
source .venv/bin/activate

# 依存パッケージ
pip install -e ".[dev]"

# .envにAPIキーを設定
cp .env.example .env
# KABUAI_JQUANTS_API_KEY=your_api_key_here を記入

# DB初期化
make setup
```

## 使い方

### ダッシュボード
```bash
make ui
# http://localhost:8501 で開く
```

### バックテスト＋ナレッジ蓄積
```bash
python run_backtest_learn.py
```
過去3ヶ月の実データで全戦略をシミュレーション。トレード結果からナレッジを自動抽出してDBに保存。

### リアルタイムペーパートレード
```bash
python run_paper.py
```
バックグラウンドで常時稼働。90秒ごとにスキャン→売買→学習。

## 学習ループ

```
1. バックテスト or ペーパートレード → トレード結果蓄積
2. ナレッジ自動抽出（勝ち/負けパターン）
3. 戦略別パフォーマンス更新（勝率・PF・平均損益）
4. 改善候補生成（条件変更・閾値調整）
5. UIで改善候補を承認/却下
6. 承認された変更のみ戦略に反映
```

## フォルダ構成

```
├── core/          コア基盤（モデル・設定・安全ガード）
├── db/            SQLiteデータベース
├── data_sources/  J-Quants API V2・TDnet
├── brokers/       ペーパーブローカー・立花API（将来）
├── scanners/      ユニバース・プレマーケットスキャン
├── strategies/    全16戦略の実装
├── execution/     実行エンジン・リスク管理
├── analytics/     トレード分析・ナレッジ抽出・学習ループ
├── tools/         特徴量エンジニアリング・バックテスト
├── ui/            Streamlitダッシュボード
└── tests/         ユニットテスト
```

## 安全設計

- `ALLOW_LIVE_TRADING=false` 固定（本番取引禁止）
- 日次損失上限 -5万円で自動停止
- 最大ポジション数制限（5件）
- 14:50に全ポジション強制決済
- 重複注文防止
- Idempotent設計

## データソース

- **J-Quants API V2**: 上場銘柄・日足・財務データ（APIキー認証）
- **TDnet**: 適時開示情報（将来実装）
- **立花証券API**: リアルタイム板・約定（将来実装）

## テスト

```bash
pytest tests/ -v
```

## ライセンス

Private - 個人利用限定
