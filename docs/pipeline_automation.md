# Pipeline Automation Runner

`scripts/run_pipeline_loop.py` は p0 → p1 → p2 → p3 → report の各 CLI を 1 つのラッパーで順次実行し、15 分間隔などの定期ジョブとして運用できるようにするためのスクリプトです。

## 主な特徴
- `configs/pipeline_defaults.yaml` でステージ別の既定引数やリトライ回数を集中管理。
- 状態ファイル (`tmp/pipeline_state.json`) に前回実行時刻や失敗回数を記録し、再起動時に `--since` を自動調整。
- 出力ファイルは `--resume` / `--merge-existing` を常時付与し、同一日付のファイルに追記する運用を維持。
- 実行結果メトリクスは `out/profiles/<profile>/metrics/pipeline/pipeline_<date>.jsonl` に JSON Lines 形式で追記。
- ロックファイル (`/tmp/modbot_pipeline.lock`) により重複起動を防止。

## 使い方
```bash
# 単発実行
python -m scripts.run_pipeline_loop --once

# ループ実行（既定 15 分間隔、Ctrl+C で停止）
python -m scripts.run_pipeline_loop --loop

# 30 分間隔、ログファイル出力、dry-run
PIPELINE_INTERVAL_MINUTES=30 \
python -m scripts.run_pipeline_loop --loop --dry-run --log-file out/pipeline.log
```

- `--interval-minutes` > 環境変数 `PIPELINE_INTERVAL_MINUTES` > `configs/pipeline_defaults.yaml` > 既定 15 分 の優先順位で周期を決定します。
- `--dry-run` はコマンド生成のみ行い、状態やメトリクスを更新しません。
- `--once` と `--loop` は排他。指定が無い場合は単発実行として動作します。
- `--quiet` / `--verbose` で標準出力ログレベルを調整できます。ログは JSON 1 行形式です。

## 設定ファイル (`configs/pipeline_defaults.yaml`)
```yaml
p0:
  resume: true
  since_fallback_minutes: 1440
  retries: 2

p1:
  batch_size: 16
  concurrency: 1
  qps: 4
  threads: 8
  raw_general_topk: 256
  rules_config: configs/rules.yaml
  retries: 1
  merge_existing: true

p2:
  batch_size: 16
  concurrency: 1
  qps: 4
  buffered: true
  nudenet_mode: always
  rules_config: configs/rules.yaml
  retries: 1
  merge_existing: true

p3:
  rules: configs/rules_v2.yaml
  retries: 1
  merge_existing: true

report:
  enabled: true
  retries: 1
```

- `enabled`, `retries`, `timeout_sec`, `extra_args` は制御項目として扱います。それ以外のキーは `--key value` 形式に変換され CLI 引数として渡されます。
- 真偽値は `true` のときのみフラグを付与 (`merge_existing: true` → `--merge-existing`)。
- `p0.since_fallback_minutes` は初回実行時に `now - minutes` を `--since` として利用します（UTC）。

## 状態とロック
- 状態ファイル: `tmp/pipeline_state.json`
  - `last_started_at` / `last_completed_at` は UTC ISO 形式で保存。
  - 破損時は `.bak` から自動復旧します。
- ロックファイル: `/tmp/modbot_pipeline.lock`
  - 別プロセスが動作中の場合は起動を中止。PID が存在しない場合は自動的にロックを削除します。

## メトリクスとレポート
- メトリクスは `pipeline_<date>.jsonl` に 1 ランごとに追記されます。
- `stages` 配列には各ステージの `status` / `duration_sec` / `records` / `attempts` を記録します。
- `report` ステージは `configs/pipeline_defaults.yaml` の `report.enabled` が `true` のときのみ実行されます。

## テスト
- `tests/scripts/test_run_pipeline_loop.py` に時刻計算や設定読込、メトリクス出力、状態ファイル処理のユニットテストを追加しています。
- 依存関係が揃っていないローカル環境でもテストを実行できるように、`pydantic` / `pydantic_settings` の最小スタブをテスト側で定義しています。

## 既知の注意点
- 実行するプロファイルの Python 仮想環境には pydantic / pydantic-settings / PyYAML などの依存をインストールしておいてください。
- ステージ固有の追加オプションが必要な場合は `configs/pipeline_defaults.yaml` に追記し、`scripts/run_pipeline_loop.py` の `build_stage_command` に反映されることを確認してください。
