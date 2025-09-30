# Repository Guidelines

## プロジェクト構成とモジュール
本リポジトリは Discord サーバー向けモデレーション bot の処理パイプラインを Python で実装しています。主要なモジュールは `app/` 以下にまとまり、`main.py` が Discord クライアントとルールエンジンを起動します。`p0_scan.py` → `cli_wd14.py` → `analysis_merge.py` → `triage.py` の順で添付画像取得から検知レポート生成までを担当します。
- `configs/` にはルール定義 `rules.yaml` や NudeNet/交差シグナル構成があり、運用パラメータ調整の際に編集します。
- `models/wd14/` は WD14 EVA02 重みとラベル情報のキャッシュ置き場です。初回推論時に自動的にダウンロードされ、差し替えはリビジョン指定で制御します。
- `out/` はスキャン結果や生成 CSV を置くためのワークスペースで、各フェーズごとの中間成果物を確認できます。
- `docs/` とルート直下の P0/P1/P2 文書は運用手順の補足資料であり、作業スケジュールやエスカレーションの参考になります。
- 依存は `requirements.base.txt`（共通）と `requirements-cpu.txt` / `requirements-gpu.txt`（いずれか片方を使用）に分離されています。`requirements.txt` は CPU プロファイルを指し、`app/cache_*.sqlite` が推論キャッシュです。削除すると再推論が走ります。

## ビルド・テスト・開発コマンド
- `python -m app.p0_scan --since 2024-01-01 --out out/p0_scan.csv` : Discord から添付ファイルを収集し CSV を蓄積します (`.env` のトークン必須)。
- `python -m app.cli_wd14 --input out/p0_scan.csv` : WD14 推論を実行し JSONL とメトリクスを出力します (ローカルモデルは自動取得)。
- `python -m app.analysis_merge --scan out/p0_scan.csv --wd14 out/p1_wd14.jsonl --metrics out/p2_metrics.json` : WD14 と NudeNet の結果を統合して解析 JSONL を生成します。
- `python -m app.cli_scan --analysis out/p2_analysis.jsonl --findings out/p3_findings.jsonl` : ルール評価を行いサマリーを表示します。
- `python -m app.cli_report --findings out/p3_findings.jsonl --severity red` : CSV レポートを作成し、重大度フィルタ付きでエクスポートします。
- `python -m app.cli_report --help` や `python -m app.cli_scan --help` を実行し、追加オプションやチャンネルフィルタを確認してください。
- NSFW 辞書は `configs/rules.yaml` の `groups.nsfw_general` を唯一の編集箇所とし、更新後は **p1（必要な場合）→p2→p3** の順に再実行して反映を確認してください。

## GPU 実行クイックスタート（WD14）
- 依存導入は必ずどちらか一方のみを選択します。
  - CPU 環境: `pip install -r requirements-cpu.txt`
  - GPU 環境 (CUDA/TensorRT): `pip install -r requirements-gpu.txt`
- 実行前に `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` で `CUDAExecutionProvider` や `TensorrtExecutionProvider` が列挙されることを確認してください。
- GPU 実行例:
  `python -m app.cli_wd14 --input out/p0/shard_00.csv --out out/p1/p1_wd14_00.jsonl --metrics out/metrics/p1_00.json --provider cuda --batch-size 48 --concurrency 24 --qps 4.0`
- `WD14_PROVIDER`（既定 `cpu`）で CLI 未指定時のプロバイダを切り替えられます。環境に CUDA/TensorRT が存在しない場合は WARN を 1 度だけ出して自動的に CPU へフォールバックします。
- メトリクス JSON (`--metrics`) には `infer_ms_avg` と `img_per_sec` が追加されており、CPU/GPU の性能比較に利用できます。必要に応じて `.env` の `WD14_CACHE_SUFFIX` を設定すると WD14 キャッシュファイルを分岐できます。

## シャーディング実行フロー（20,000枚規模）
1. **分割**: `python scripts/split_index.py --input out/p0_scan.csv --out-dir out/p0 --shards 10`
2. **p1 (WD14)**:
   ```bash
   python scripts/run_p1_sharded.py \
     --shard-glob "out/p0/shard_*.csv" \
     --out-dir out \
     --provider cuda --batch-size 48 --concurrency 24 --qps 4.0 \
     --parallel 2 --resume \
     --status-file out/status/p1_manifest.json
   ```
   - `.tmp` ファイルは完了時に自動 rename（失敗時は残置）。
   - `--resume` を付けると既存の最終ファイルをスキップし、途中停止後の再開が可能です。
   - 429 や 5xx が多い場合は `--qps` を下げるか `--parallel 1` に落として調整します。
3. **p2 (analysis merge)**:
   ```bash
   python scripts/run_p2_sharded.py \
     --shard-glob "out/p0/shard_*.csv" \
     --wd14-dir out/p1 \
     --out-dir out \
     --qps 4.0 --concurrency 16 \
     --parallel 2 --resume \
     --status-file out/status/p2_manifest.json \
     --extra-args "--nudenet-mode auto"
   ```
   - ランナーは `--rules-config configs/rules.yaml` を自動付与します。別の辞書を使う場合は `--extra-args` で明示してください。
4. **マージ（任意）**:
   ```bash
   python scripts/merge_jsonl.py --glob "out/p1/p1_wd14_*.jsonl" --out out/p1/p1_wd14_all.jsonl
   python scripts/merge_jsonl.py --glob "out/p2/p2_analysis_*.jsonl" --out out/p2/p2_analysis_all.jsonl
   ```

- 各ランナーは manifest JSON を更新し、`queued/running/done/failed` を追跡します。ファイルは `out/status/` に保存され、再実行時も引き継がれます。
- SQLite キャッシュは WAL + 60 秒 timeout に設定済みですが、ロック待ちが発生する場合は `--parallel 1` に落として運用してください。
- 失敗 shard は末尾で 1 回リトライします。複数回失敗する場合は manifest を確認し、`--resume` 付きでもう一度実行すると該当 shard のみが再試行されます。
- `--extra-args "--nudenet-mode auto"` を指定すると、p2 での NudeNet 実行を WD14 スコアに基づいてゲートできます。`auto`（既定）は `rating.questionable ≥ 0.35` または `rating.explicit ≥ 0.20` のときのみ推論を実行し、`always` は全件実行、`never` はキャッシュヒット以外をスキップします。
- メトリクスには従来の `average_nudenet_latency_ms` / `from_cache` に加えて、`nudenet.executed` / `nudenet.skipped` / `nudenet.p95_latency_ms` などの詳細指標が出力されます。ゲート調整時は `nudenet.executed` と `nudenet.skipped` を並行して確認してください。

### Runpod バッチ運用
- Runpod 上での大規模加工手順は `docs/runpod_batch_runbook.md` を参照してください。
- Network Volume, S3 同期、レート制御、A/B 比較、レビュー用 CSV の出力までを一冊にまとめています。

## コーディングスタイルと命名規約
Python 3.11 を想定し、PEP 8 準拠の 4 スペースインデントを徹底してください。関数・変数は snake_case、クラスは PascalCase で統一し、型ヒントと `from __future__ import annotations` を前提に遅延評価を維持します。自動整形ツールは同梱されていないため、`ruff` や `black` をローカルで実行し差分がない状態でコミットしてください。非同期処理では `asyncio` ループのキャンセル管理が重要なので、タイムアウト値や `RateLimiter` 利用箇所の命名を明確にしましょう。

## テスト指針
現時点で自動テストスイートは未整備です。新規機能を追加する際は `tests/` ディレクトリを作成し `pytest` ベースのテストを導入してください。I/O を伴う処理は Discord クライアントや Hugging Face API をモックし、副作用のないユニットテストを優先します。主要フローごとに `test_<module>.py` 形式で命名し、スキャンからルール評価までのハッピーパスと代表的なエラーケースを最低限カバーしてください。回帰を防ぐため、重い推論は `monkeypatch` でキャッシュ層にスタブを挿入し、データサンプルは `tests/fixtures/` に配置しましょう。

## DSL レイヤ導入の進め方
- `configs/rules.yaml` に `version: 2` を記載すると DSL が有効になり、既存ロジックと併用されます（未指定/`1` は従来どおり）。
- DSL では `groups`（タグ/ワイルドカード定義）、`features`（中間式）、`rules`（when/reasons）を記述します。例:
  ```yaml
  version: 2
  groups:
    nsfw_general: ["bikini", "see_through", "underboob"]
  features:
    combo: "rating.explicit * exposure_peak"
  rules:
    - id: RED-NSFW-101
      severity: red
      priority: 10
      when: "(!channel.is_nsfw) && (rating.explicit >= 0.6 || sum('nsfw_general') >= 0.25)"
      reasons: ["exp={rating.explicit:.2f}", "sum={sum('nsfw_general'):.2f}"]
  ```
- 利用可能な変数・関数
  - 変数: `rating.*`, `exposure_peak`, `minors_peak`, `channel.is_nsfw`, `message.is_spoiler`, `attachment_count` など（既存メトリクスは `metrics` 経由で DSL にバインド済み）。
  - 関数: `score(tag)`, `sum(group)`, `max(group)`, `any(group, gt=0.35)`, `count(group, gt=0.35)`, `topk_sum(group, k, gt=0.35)`, `clamp(x, lo, hi)`, `nude.has(flag)`, `nude.any(prefix="EXPOSED_", min=1)` など。
  - 新規メトリクス: `exposure_area` / `exposure_count` （`analysis_merge` が `nudity_area_ratio` / `nudity_box_count` を計上した値）。
- `app/engine/` 配下に Safe AST ベースの DSL ランタイムを実装済み。`DslProgram.evaluate()` の結果は `metrics.dsl` / `metrics.winning` に格納され、最終判定は DSL ベースで一貫します（legacy は `--allow-legacy` 併用時のフォールバックのみ）。
- 厳格モードが必要な場合は `dsl_mode: strict` を `rules.yaml` に追加すると未知変数やゼロ除算で即エラーになります（既定は `warn` モードで 0/False にフォールバック）。
- DSL の単体テストは `tests/engine/test_dsl_program.py` を参考に追加してください。主要ケース（命中、非命中、安全性と禁止ノード）を網羅することが推奨です。

## p3 スキャン（DSL対応）
- 本番実行例:
  ```bash
  python -m app.cli_scan \\
    --analysis out/p2/p2_analysis_all.jsonl \\
    --findings out/p3/findings.jsonl \\
    --rules configs/rules.yaml \\
    --metrics out/metrics/p3_run.json
  ```
- ドライラン（結果を出さずメトリクスのみ確認）:
  ```bash
  python -m app.cli_scan \\
    --analysis out/p2/p2_analysis_all.jsonl \\
    --rules configs/rules.yaml \\
    --metrics out/metrics/p3_dry.json \\
    --dry-run
  ```
- `--print-config` で DSL グループ/ルール数を要約表示できます。`--limit` / `--offset` を使うとサンプリング実行が可能です。

## A/B 比較フロー
1. 現行ルール（A）と候補ルール（B）を用意
2. A/B を実行し、差分を確認
   ```bash
   python -m app.cli_rules_ab \\
     --analysis out/p2/p2_analysis_all.jsonl \\
     --rulesA configs/rules.yaml \\
     --rulesB configs/rules_v2.yaml \\
     --out-dir out/exports \\
     --sample-diff 200 \\
     --samples-minimal \\
     --samples-redact-urls
   ```
3. `p3_ab_compare.json` で counts/delta/混同行列、`p3_ab_diff.csv` で差分列、`p3_ab_diff_samples.jsonl` でレビュー用サンプルを確認
4. 差分レビュー後、合意したルールを `configs/rules.yaml` に昇格

## strict / warn モード
- `warn`（既定）: 未知識別子や式エラーは 0/False にフォールバックし WARN を 1ルール×エラー種あたり最大5回表示
- `strict`: ローダ/評価器のいずれかで例外化し、問題ルールを即時洗い出し
- モードの優先順位は **`--lock-mode` > CLI `--dsl-mode` > ENV `MODBOT_DSL_MODE` > YAML `dsl_mode` > warn** です。`--lock-mode` を使うと A/B 両側を同一ポリシーで強制実行します。
- strict を常用したい場合は `MODBOT_DSL_MODE=strict` を環境変数に設定するか、CLI で `--dsl-mode strict` を指定してください。両者を固定したい場合は `--lock-mode strict` を利用してください。
- legacy ルール（`version: 1`）が混ざった場合は既定で比較を停止 (exit code 2) します。レビューだけ続行したい場合は `--allow-legacy` を付けると `p3_ab_compare.json` に `note="skipped due to legacy ruleset"` が追記され、CSV/サンプルは生成されません。
- `--samples-minimal` を付けるとサンプル JSONL が最小ビュー（severity/rule/reasons/metrics のみ）で出力されます。`--samples-redact-urls` を併用すると URL フィールドは `null`、文中の URL は `[URL]` に置換されます。
- `--out-dir DIR` を指定すると `DIR/p3_ab_compare.json`・`DIR/p3_ab_diff.csv`・`DIR/p3_ab_diff_samples.jsonl` が自動生成されます（個別指定よりも運用が容易）。
- レガシー構成（`version: 1`）が残っている場合は `--allow-legacy --fallback green` で強制的に `severity=green` として出力できます（もしくは `--fallback skip` で書き込みを抑止）。
- WARN が多い場合は strict で健全性を確認してから本番に適用してください

## コミットとプルリクエスト
コミットメッセージは `feat(scope): summary` 形式の Conventional Commits が利用されています。変更内容と影響範囲を明確にするため `fix`, `chore`, `docs` なども適宜活用してください。プルリクエストには目的、主な変更点、検証ログ (`python -m app.cli_scan --help` など) を記載し、関連 issue やチケット番号をリンクします。レビューではキャッシュ破棄の有無や Discord API 制限への影響を説明し、Discord 出力や通知内容を変更した場合はスクリーンショット、サンプルログ、生成 CSV の抜粋を添付してください。

## セキュリティと設定
`.env` に `DISCORD_BOT_TOKEN`、`GUILD_ID`、`LOG_CHANNEL_ID` などの機密値を設定します。ファイルは git 管理外で保持し、共有は安全な秘密情報マネージャー経由で行ってください。外部モデルを更新する際は `configs/` と `models/` の差分を確認し、秘匿情報を誤ってコミットしないよう `git status` と `.gitignore` を実行前に点検しましょう。ローカル検証ではテスト用トークンを使用し、本番トークンは CI やホスティング環境のシークレットストアに限定してください。
