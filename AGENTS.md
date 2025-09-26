# Repository Guidelines

## プロジェクト構成とモジュール
本リポジトリは Discord サーバー向けモデレーション bot の処理パイプラインを Python で実装しています。主要なモジュールは `app/` 以下にまとまり、`main.py` が Discord クライアントとルールエンジンを起動します。`p0_scan.py` → `cli_wd14.py` → `analysis_merge.py` → `triage.py` の順で添付画像取得から検知レポート生成までを担当します。
- `configs/` にはルール定義 `rules.yaml` や NudeNet/交差シグナル構成があり、運用パラメータ調整の際に編集します。
- `models/wd14/` は WD14 EVA02 重みとラベル情報のキャッシュ置き場です。初回推論時に自動的にダウンロードされ、差し替えはリビジョン指定で制御します。
- `out/` はスキャン結果や生成 CSV を置くためのワークスペースで、各フェーズごとの中間成果物を確認できます。
- `docs/` とルート直下の P0/P1/P2 文書は運用手順の補足資料であり、作業スケジュールやエスカレーションの参考になります。
- ルートの `requirements.txt` が公式依存関係リスト、`app/cache_*.sqlite` が推論キャッシュです。削除すると再推論が走ります。

## ビルド・テスト・開発コマンド
- `python -m app.p0_scan --since 2024-01-01 --out out/p0_scan.csv` : Discord から添付ファイルを収集し CSV を蓄積します (`.env` のトークン必須)。
- `python -m app.cli_wd14 --input out/p0_scan.csv` : WD14 推論を実行し JSONL とメトリクスを出力します (ローカルモデルは自動取得)。
- `python -m app.analysis_merge --scan out/p0_scan.csv --wd14 out/p1_wd14.jsonl --metrics out/p2_metrics.json` : WD14 と NudeNet の結果を統合して解析 JSONL を生成します。
- `python -m app.cli_scan --analysis out/p2_analysis.jsonl --findings out/p3_findings.jsonl` : ルール評価を行いサマリーを表示します。
- `python -m app.cli_report --findings out/p3_findings.jsonl --severity red` : CSV レポートを作成し、重大度フィルタ付きでエクスポートします。
- `python -m app.cli_report --help` や `python -m app.cli_scan --help` を実行し、追加オプションやチャンネルフィルタを確認してください。

## コーディングスタイルと命名規約
Python 3.11 を想定し、PEP 8 準拠の 4 スペースインデントを徹底してください。関数・変数は snake_case、クラスは PascalCase で統一し、型ヒントと `from __future__ import annotations` を前提に遅延評価を維持します。自動整形ツールは同梱されていないため、`ruff` や `black` をローカルで実行し差分がない状態でコミットしてください。非同期処理では `asyncio` ループのキャンセル管理が重要なので、タイムアウト値や `RateLimiter` 利用箇所の命名を明確にしましょう。

## テスト指針
現時点で自動テストスイートは未整備です。新規機能を追加する際は `tests/` ディレクトリを作成し `pytest` ベースのテストを導入してください。I/O を伴う処理は Discord クライアントや Hugging Face API をモックし、副作用のないユニットテストを優先します。主要フローごとに `test_<module>.py` 形式で命名し、スキャンからルール評価までのハッピーパスと代表的なエラーケースを最低限カバーしてください。回帰を防ぐため、重い推論は `monkeypatch` でキャッシュ層にスタブを挿入し、データサンプルは `tests/fixtures/` に配置しましょう。

## コミットとプルリクエスト
コミットメッセージは `feat(scope): summary` 形式の Conventional Commits が利用されています。変更内容と影響範囲を明確にするため `fix`, `chore`, `docs` なども適宜活用してください。プルリクエストには目的、主な変更点、検証ログ (`python -m app.cli_scan --help` など) を記載し、関連 issue やチケット番号をリンクします。レビューではキャッシュ破棄の有無や Discord API 制限への影響を説明し、Discord 出力や通知内容を変更した場合はスクリーンショット、サンプルログ、生成 CSV の抜粋を添付してください。

## セキュリティと設定
`.env` に `DISCORD_BOT_TOKEN`、`GUILD_ID`、`LOG_CHANNEL_ID` などの機密値を設定します。ファイルは git 管理外で保持し、共有は安全な秘密情報マネージャー経由で行ってください。外部モデルを更新する際は `configs/` と `models/` の差分を確認し、秘匿情報を誤ってコミットしないよう `git status` と `.gitignore` を実行前に点検しましょう。ローカル検証ではテスト用トークンを使用し、本番トークンは CI やホスティング環境のシークレットストアに限定してください。
