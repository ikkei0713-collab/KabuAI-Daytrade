# KabuAI デイトレード

AI駆動型の日本株デイトレード学習システム。
ペーパートレードとバックテストで勝ちパターンを自動発見し、戦略を継続的に改善する。

## 後場 PM-VWAP reclaim (2026-03-25)

主戦略 `vwap_reclaim` を**別戦略に増やさず**拡張し、東証の後場（12:30〜）で再計算した **PM-VWAP** に対して「下から上への reclaim のち、その上で数本定着」「13:00以降」「後場の出来高・売買代金が十分」である場合のみエントリーを許可する。

- **重視するもの**: 後場セッション、出来高・売買代金、PM-VWAP 上での定着（一瞬のブレイクでは入らない）、`trend_follow.pm_trend_filter` によるフィルタ。
- **低位株（例: 100〜500円）は主条件にしない**。任意の小さな `pm_low_price_bonus` のみ。
- **intraday 分足が取れるとき** PM 特徴量は高信頼。日足のみの proxy では `pm_intraday_is_proxy` が真となり、品質閾値未満の PM エントリーは行わない。

主要な設定は `core/config.py`（`PM_*`, `LOW_PRICE_BONUS_*`）を参照。

## v3.3 収束フィルタ導入 (2026-03-22)

**方針: 「拡散に飛び乗らず、収束後の再拡大だけを狙う」**

### v3.3 変更点
- **MA収束フィルタ**: MA群(5/10/20)と価格の収束度・ボラ圧縮度を計算し、拡散飛び乗りを拒否
- **vwap_reclaim 改善**: 収束フィルタ通過時のみ confidence 加点、GC/DC直後拡散は減点
- **trend_follow 進化**: 「強いから入る」→「収束後だけ許可するフィルタ」へ (`convergence_filter()` 追加)
- **ORB 収束評価**: watch のまま、収束後 continuation のみ評価
- **銘柄選定に収束加点**: event + MA収束 + 価格圧縮 → 軽く加点 (weight=0.03)
- **feedback に収束分析追加**: 収束あり/なし別 PF, WR, avg_pnl を自動比較
- **UI に収束表示**: watchlist に収束スコア・収束後候補ラベル・拡散注意ラベルを追加
- **収束特徴量は日足から直接計算** (proxy ではない) → proxy 依存度を増やさない
- **config 化**: 全閾値を `core/config.py` の Settings で管理

### 最適化結果 (2026-03-23, 6ヶ月データ, 219回試行)

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| min_time_below_vwap_min | 10 | OOS PF=1.92 |
| min_volume_at_reclaim | 1.0 | 件数確保 |
| target_atr_multiple | 1.5 | 利確を適度に |
| reclaim_buffer_pct | 0.20 | ノイズ除去 |
| max_distance_from_vwap_pct | 2.0 | 広めに許容 |
| MAX_MA_SPREAD_PCT | 0.03 | 緩め=件数確保 |
| MIN_MA_CONVERGENCE_SCORE | 0.55 | 維持 |
| MIN_RANGE_COMPRESSION | 0.30 | 緩和 |
| MIN_VOL_COMPRESSION | 0.50 | やや厳格 |
| CONVERGENCE_BOOST | 0.04 | 控えめ |
| EXPANSION_PENALTY | 0.10 | 控えめ |

### 収束フィルタの仕組み
```
vwap_reclaim (主戦略)
  ├── trend_follow.is_trending() → EMA/VWAP alignment check
  ├── trend_follow.convergence_filter() → MA収束/ボラ圧縮/GC直後拡散チェック
  │     ├── ma_spread_pct > 閾値 → 拒否 (拡散しすぎ)
  │     ├── extension_from_ma_score < 0.35 → 拒否 (乖離しすぎ)
  │     ├── post_cross_expansion_flag → 拒否 (GC/DC直後の拡散)
  │     ├── ma_convergence_score ≥ 0.55 → 加点
  │     ├── range/vol compression ≥ 閾値 → 加点
  │     └── squeeze_breakout_ready → 最優先加点
  └── confidence 最終判定 → MIN_CONFIDENCE 0.65 でゲート
```

**重要**: intraday proxy 制約があるため、収束特徴量の評価も過信しないこと。
日足ベースの MA/ATR 計算は比較的安定しているが、intraday の精度とは異なる。

## v3.2 論文準拠フィードバック強化 (2026-03-20)

**方針: 「増やす」ではなく「削る」「止める」「絞る」+ proxy 依存ペナルティ + 論文準拠フィードバック**

### v3.2 新機能 (論文知見の最小差分適用)
- **Feature Statistics**: 22特徴量 × 15統計量 (count/mean/std/percentiles/skew/kurtosis/missing_ratio/zero_ratio/is_proxy)
- **IC/ICIR 近似計算**: Spearman(selector_score, realized_return) で銘柄選定の有効性を定量化
- **Net Exposure**: 方向/戦略/銘柄の偏り度合いを可視化
- **Sharpe Ratio**: 全メトリクス計算に追加
- **Cumulative IC プロット**: selector_score の予測力の推移を可視化
- **Net Exposure プロット**: Long/Short バイアスの推移を可視化
- **Prompt テンプレート**: prompts/ に Prompt1(基本)/Prompt2(追加情報)/Prompt3(プロット付き) を配置
- **フィードバック設計 3段階**: 基本テキスト → +feature stats/IC/exposure → +8枚のプロット

### v3.1 (proxy 依存ペナルティ)
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
  │    └─ Claude Code向けフィードバックパッケージ自動生成 (論文準拠3段階)
  │         ├─ feedback_packet.json (全メトリクス+feature stats+IC/ICIR+net exposure)
  │         ├─ feedback_summary.md (LLMにコピペ可能なサマリ)
  │         └─ plots/ (equity curve, drawdown, OOS PF, regime heatmap,
  │              cumulative IC, net exposure, confidence vs PnL, feature compare)
  │
  └─ run_optimize.py
       ├─ パラメータ空間をランダムサーチ (OOS PFで評価)
       └─ 最適パラメータをJSON保存

ダッシュボード (http://localhost:8502)
  └─ SPY前日・セクターバイアス・ウォッチリスト・レジーム
     戦略on/off・異常停止・トレード履歴 / 30秒自動更新
```

## 戦略構成 (v3.3)

| Status | 戦略 | Proxy率 | 説明 |
|--------|------|---------|------|
| **Active** | **vwap_reclaim** | 100% | 唯一の主戦略。収束フィルタで拡散飛び乗り抑制 |
| Filter | trend_follow | 0% | is_trending() + convergence_filter() で vwap_reclaim のゲートに使用 |
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
cat knowledge/feedback_packet.json               # 全メトリクス+IC/ICIR+feature stats
ls  knowledge/plots/                             # 可視化チャート8枚
cat prompts/prompt3_with_plots.md                # Prompt3テンプレート（プロット付き）
```

### Claude Code に改善を依頼する時
```bash
# バックテスト実行 → フィードバック自動生成
.venv/bin/python run_backtest_learn.py

# 3段階のフィードバック（論文準拠）
# Prompt1: 基本テキスト
cat knowledge/feedback_summary.md

# Prompt2: + feature stats / IC / net exposure
cat knowledge/feedback_packet.json | python -m json.tool

# Prompt3: + 8枚のプロット
ls knowledge/plots/
#   equity_curve.png, drawdown_curve.png, strategy_oos.png,
#   regime_heatmap.png, cumulative_ic.png, net_exposure.png,
#   confidence_vs_pnl.png, feature_win_loss_compare.png

# テンプレートを参照
cat prompts/prompt1_basic.md         # 基本フィードバック
cat prompts/prompt2_extended.md      # 追加情報付き
cat prompts/prompt3_with_plots.md    # プロット付き
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
| **改善フィードバック** | バックテスト後にJSON+Markdown+8plotsを自動生成。feature stats・IC/ICIR・net exposure・過学習警告・proxy依存度を可視化 (論文準拠3段階Prompt) |

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
├── prompts/        Claude Code向けPromptテンプレート（Prompt1/2/3の3段階）
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

## 参考文献

本システムの設計・実装に以下の学術論文の知見を活用しています。

### [1] GA を用いた適合度関数と相場変化に着目したシステムトレード
- **著者**: 荒井裕也, 折原良尚, 中川博之, 清雄一, 田原康之, 大須賀昭彦
- **所属**: 電気通信大学大学院 情報システム学研究科
- **出典**: 人工知能学会 SIG-FIN-010, pp.55-60, 2013
- **URL**: https://www.jstage.jst.go.jp/article/jsaisigtwo/2013/FIN-010/2013_10/_pdf
- **適用箇所**:
  - **損失回避型適合度関数 (fit3)**: `-Loss + 0.01 * Profit` → `run_optimize.py` のパラメータ評価に使用。損失を強く罰することで過学習を抑制し、安定したパラメータを発見する
  - **レジーム別インジケータ切替**: トレンド相場ではトレンドフォロー系、レンジ相場ではオシレータ系の指標が有効 → `market_regime.py` の `get_convergence_params()` でレジーム別に収束フィルタ閾値を切り替える設計に反映
  - **短期学習ウィンドウ**: ハイブリッド切替は1-2ヶ月の短い学習期間が良い → ローリングウィンドウの設計方針に反映

### [2] 部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略
- **著者**: 中川慧, 竹本悠城, 久保健治, 加藤真大
- **所属**: 大阪公立大学 / MATSUO Institute / 東京大学 / みずほDLフィナンシャルテクノロジー
- **出典**: 人工知能学会 SIG-FIN-036, pp.76-83, 2026
- **URL**: https://www.jstage.jst.go.jp/article/jsaisigtwo/2026/FIN-036/2026_76/_pdf/-char/ja
- **適用箇所**:
  - **米国→日本セクターバイアス**: 前日の米国セクターETF騰落率から日本業種への波及を予測 → `tools/sector_bias.py` の lead-lag signal に反映。論文は部分空間正則化PCA (K=3, λ=0.9) を使うが、本システムでは軽量な手動マッピング + 線形加重で近似実装
  - **IC/ICIR 評価フレームワーク**: 銘柄選定スコアの予測力を Spearman IC で定量化 → `analytics/feedback_packet.py` の `_ic_icir()` に反映
  - **セクターローテーション**: GICS セクター分類による循環/ディフェンシブ区分の考え方 → `sector_bias.py` のマッピングテーブル設計に参考

## ライセンス

MIT
