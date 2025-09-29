# Mahjong Attention AI (日本語版)

Mahjong Attention AI は、日本式リーチ麻雀に特化したトランスフォーマーモデルのMVPです。教師あり模倣学習からスタートし、将来的に PPO を用いた強化学習へ自然に拡張できるアーキテクチャを採用しています。本リポジトリは、合成牌譜データ、評価ユーティリティ、そして uv を用いた再現性の高いワークフローを同梱し、エンコード→学習→評価→バックテストまでを一貫して実行できます。

## 主な特徴

- 盤面トークン列と行動候補に対して **Transformer Encoder + Cross-Attention** を適用し、Policy / Value / Auxiliary ヘッドを同時に学習。
- 合成牌譜を用いた **教師あり学習パイプライン**（encode → train → eval）。Top-1 / Top-3 精度を標準レポート。
- 同配分に対応した **バックテスト基盤**。ブートストラップによる95%信頼区間、A/B比較、JSON/CSV 出力をサポート。
- **taskipy + ruff + mypy + pytest** によるモダンな開発体験と品質管理。
- 将来の自動対局強化のための **PPOスタブとシミュレータブリッジ** を用意。

## クイックスタート（uv 使用）

```bash
uv venv
uv sync
uv run python -m mahjong_attn_ai.cli encode --config configs/default.yaml
uv run python -m mahjong_attn_ai.cli train  --config configs/default.yaml
uv run python -m mahjong_attn_ai.cli eval   --config configs/default.yaml --ckpt runs/latest/best.ckpt
uv run python -m mahjong_attn_ai.cli backtest --config configs/eval_backtest.yaml --ckpt runs/latest/best.ckpt
uv run python -m mahjong_attn_ai.cli abtest   --config configs/eval_backtest.yaml --ckpt-a runs/latest/best.ckpt --ckpt-b runs/latest/baseline.ckpt
```

`taskipy` でも同等のコマンドを利用できます：

```bash
uv run task setup
uv run task encode
uv run task train
uv run task eval
uv run task backtest
uv run task abtest
uv run task lint
uv run task typecheck
uv run task test
```

## 設定ファイルの概要

- `configs/default.yaml`: 学習全般の基本設定（データパス、Transformer ハイパーパラメータ、シミュレータ初期値、バックテストの相手Botなど）。
- `configs/eval_backtest.yaml`: 評価/バックテスト向けの推奨設定（シード数やブートストラップ試行を増加）。

設定ファイルは `mahjong_attn_ai.cli` が解釈する共通スキーマ（`data`、`dataset`、`model`、`training`、`simulator`、`backtest` セクション）を共有します。

## 合成データ

`data/sample_kifus/` に収録された合成牌譜は `SyntheticKifuParser` で読み込みます。牌・アクションを文字列表現で記述しても自動で ID 化され、`dataset.auto_generate` を有効にすると不足分をランダム生成データで補えます。実牌譜を利用する場合は、このパーサ部を差し替えることで既存パイプラインを活かしつつ移行できます。

## モデル構成

- **盤面エンコーダ**: トークン埋め込み + 位置埋め込み → Pre-LN `nn.TransformerEncoder`。
- **候補エンコーダ**: 行動候補トークンの埋め込みと位置埋め込みを適用。
- **Cross-Attention**: 行動列を Query、盤面列を Key/Value としてアテンション。
- **ヘッド**:
  - Policy ヘッド: 合法手のみを残してロジット→log_softmax。
  - Value ヘッド: 半荘/局 EV を回帰。
  - Auxiliary ヘッド: 危険度（放銃確率）などの追加監督信号を推定。
- 損失関数: `L = CE(policy) + λ_v * MSE(value) + λ_aux * MSE(aux)`。係数は YAML で変更可能。

## 評価・バックテスト

`mahjong_attn_ai.eval.backtest.BacktestRunner` は以下を提供します：

- シード制御付きの自己対戦（同配分対応）。
- KPI ログ（平均順位・素点EV・和了率・放銃率・立直率・平均打点など）。
- ブートストラップによる95%信頼区間、A/B比較の平均差とCI。
- `runs/eval/<timestamp>/` 配下に JSON (`summary.json`) と CSV (`table.csv`) を出力。

Rich での出力例：

```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric             ┃ Value  ┃
┣━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━┫
┃ average_rank       ┃ 2.1000 ┃
┃ score_ev           ┃ 320.0000 ┃
┃ kpi_win_rate       ┃ 0.2450 ┃
┃ rank_ci_low        ┃ 1.9500 ┃
┃ rank_ci_high       ┃ 2.2500 ┃
┃ score_ci_low       ┃ 180.0000 ┃
┃ score_ci_high      ┃ 460.0000 ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━┛
```

`abtest` サブコマンドは、2つのポリシーのサマリーと差分（CI 付き）を表示し、`summary.json` を保存します。

## テスト

Pytest では、特徴量エンコードの形状、合法手マスク、前向き推論のテンソル形状、バックテスト API のスモークテストをカバーしています。`uv run task test` で一括実行できます。

## 今後の拡張

- `parser.py` / `dataset.py` を拡張し、実牌譜（例: mjai）の読み込みに対応。
- `env.simulator_stub.Simulator` を Rust や mjai 連携に差し替えて実戦レベルのシミュレーションを実現。
- シミュレータから逐次サンプルを受け取れるようになったら `training/rl_stub.py` に PPO を実装。
- Auxiliary タスク（向聴数、受け入れ枚数など）の追加や Weights & Biases 連携。

## 想定される出力

- **学習**: `runs/latest/best.ckpt`（ベストモデル）と Top-1/Top-3 を含む学習ログ。
- **評価**: `eval` 実行時の JSON 出力およびコンソールログ。
- **バックテスト**: `runs/eval/<timestamp>/summary.json` と `runs/eval/<timestamp>/table.csv`。
- **A/Bテスト**: `runs/eval/abtest-*/summary.json` にポリシーA/Bおよび差分指標を記録。
