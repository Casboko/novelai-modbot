# Repository Guidelines

## プロジェクト構成とモジュール整理
- `src/bot/` Discord Bot エントリーポイントと Slash コマンド。`main.py` から `asyncio.run` で起動。
- `src/pipeline/` メッセージ履歴クロール、pHash 重複排除、CSV/SQLite への永続化を担当。
- `src/analyzers/` WD14・NudeNet ラッパーとタグ正規化。モデルは `assets/models/` にキャッシュ。
- `src/rules/` 判定ロジックと通知テンプレを集約。文面は `notify_templates/ja.md` を基準に管理。
- `tests/` 下に pytest を配置。Discord API 呼び出しは `tests/fixtures/discord_mocks.py` でモック化。
- サンプル画像やログ雛形は `assets/samples/`、ドキュメントは `docs/` に置き、個人情報を持ち込まない。

## ビルド・テスト・開発コマンド
- `poetry install` 依存を導入。Python 3.11 系を前提にし、poetry.lock を常に最新に保つ。
- `poetry run pre-commit run --all-files` でフォーマッタ（black）、静的解析（ruff, mypy）を一括実行。
- `poetry run pytest tests/ --maxfail=1 --ff` 単体・統合テストを最速で確認。
- `poetry run pytest --cov=src --cov-report=term-missing` でカバレッジを取得し、85% 以上を維持。
- `poetry run python -m bot.cli scan --guild <guild_id>` 過去履歴クロールをローカルで検証。
- `poetry run python -m bot.cli report --limit 50` 最新スキャン結果を要約出力し、レビュー用ログを生成。

## コーディングスタイルと命名規約
- 4 スペースインデント、`from __future__ import annotations` をデフォルトで追加し型注釈を簡潔に。
- ファイル・モジュールは `snake_case.py`、クラスは `PascalCase`、内部関数は `_snake_case`。
- 例外メッセージは日本語で書き、Discord ログは構造化 JSON（`structlog`）で `logs/` に出力。
- black・ruff・mypy の設定は `pyproject.toml` に集約。PR 前に `poetry run pre-commit` を必ず通す。
- 環境設定は `.env` に保持し、`pydantic.BaseSettings` で読み込む。秘密情報はコードに直書きしない。

## テスト方針
- pytest + pytest-asyncio を使用。命名は `test_<対象>_<条件>`、統合テストは `tests/integration/` に配置。
- Discord API 呼び出しは `pytest-httpx` や `responses` でモックし、429 リトライ挙動の回帰テストを含める。
- 大型フィクスチャは `tests/fixtures/` に保存し、3 MB 未満のサンプル画像のみコミットする。
- バグ再発防止のため、報告再現用の CLI は `tests/e2e/` で `poetry run python -m tests.e2e.<name>` として提供。

## コミットとプルリクエスト
- コミットは Conventional Commits を準拠。例: `feat(scanner): add thread resume cursor`。
- 設定・CI 変更は `chore(config): ...`、ドキュメントは `docs:` プレフィックスで分離。
- PR 説明は「背景」「変更点」「検証」の 3 項目を必須。関連 Issue、スキャン結果ログ、スクリーンショットを添付。
- レビュー前に `poetry run pre-commit run --all-files` と `poetry run pytest` の結果を PR テンプレートに記載。
- モデル更新や閾値調整の際は、影響ギルドとロールアウト手順を PR に明記してからマージ。

## セキュリティと設定管理
- `DISCORD_BOT_TOKEN`, `DATABASE_URL`, `HF_TOKEN` などの資格情報は `.env` で管理し、`.env.example` を更新。
- モデルキャッシュは `assets/models/` に置き、`assets/models/VERSION` に SHA256 を記録して改ざん検出を行う。
- ローカル永続化は `data/` 配下に限定し、個人識別情報を含むファイルは `.gitignore` に追加。
- 本番ギルド設定は `config/prod.yaml` を参照し、暗号化ストレージ経由で配布。平文共有は禁止。
