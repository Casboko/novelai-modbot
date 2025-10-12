# Pipeline Notifications Runner

`scripts/run_notifications.py` は p3 の findings とパイプラインメトリクスを監視し、Discord へ通知を送信するためのスクリプトです。フェーズ①で生成された成果物を基に、モデレーター向けのアラートや日次サマリを自動化します。

## 主な機能

- `configs/notifications.yaml` でチャネル設定・テンプレート・メンションロールを管理。
- findings JSONL から severity >= yellow の新着レコードを抽出し、Webhook または Bot API（REST）で通知。
- `out/profiles/<profile>/metrics/pipeline/pipeline_<date>.jsonl` を読み取り、ステージ失敗時に pipeline アラートを送信（stderr は末尾 1024 Byte をUTF-8で保持。非対応文字は base64 表記）。
- `tmp/notification_state.json` に最終通知時刻・pending payload・失敗回数を保存。バックアップ (`.bak`) と `.tmp` を利用した安全な書き出し。
- `--loop` 実行時は `--interval-minutes`（既定: `alert_cooldown_minutes`）ごと、または CRON/systemd からの単発呼び出しも可能。

## 使い方

```bash
# 単発実行（dry-run）
python -m scripts.run_notifications --dry-run --profile current --date 2025-10-12

# 15 分間隔でループ実行（実送信）
python -m scripts.run_notifications \
  --loop \
  --interval-minutes 15 \
  --log-file out/notifications.log
```

- `--only findings` や `--only alerts` を複数回指定すると通知種別を絞り込めます。
- `--state-file` / `--lock-file` で状態ファイルや排他ファイルの場所を変更できます。
- ログは JSON 形式で stdout と任意のファイルに出力されます（`notification_sent` / `notification_error` など）。

## 設定 (`configs/notifications.yaml`)

- `defaults`: タイムゾーン、メッセージ URL prefix、レート制御、リトライ回数、アラートのクールダウンを定義。
- `channels`: チャンネルごとに `type`（webhook/bot）、severity 閾値、テンプレート、ロールメンションを指定。環境変数 `${VAR}` 記法に対応。
- `templates`: Embed のタイトル/本文/フィールドを `str.format` で記述。`{reasons}` や `{rating_explicit:.2f}` などレコードの値を埋め込めます。

## ステートファイル (`tmp/notification_state.json`)

```json
{
  "profile": "current",
  "last_findings_at": "2025-10-12T03:20:00Z",
  "last_findings_id": "1234567890:feedface",
  "last_alert_at": "2025-10-12T03:30:00Z",
  "last_alert_run_id": "run-42",
  "pending_notifications": [],
  "failure_count": 0
}
```

- `pending_notifications` には送信失敗した payload が保持され、次回起動で再送されます。
- ファイル破損時は `.bak` から自動復旧します。

## パイプライン連携

- `scripts/run_pipeline_loop.py` で失敗したステージの stderr（末尾 1024 Byte）と `returncode` を `stage_metrics[stage]` に保存するため、runner 側で詳細なエラーを通知に含められます。
- `PartitionPaths.pipeline_metrics_path()` を利用して対象日のメトリクス JSONL を一貫して参照します。

## テスト

- `tests/notify/test_pipeline_notifier.py` にユニットテストを追加。テンプレート展開、finding 判定、state更新、再送キューの挙動などをカバーしています。
