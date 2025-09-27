# モデレーション通知チケット管理ビュー 実装計画

## 目的
- `notify` 済みで削除待ち（`status="notified"`）のチケットを、Discord 内で確認・管理できるようにする。
- `/report` と同じビジュアルでカード表示し、モデレータが期限前にキャンセルできる UI を提供する。

## 現状整理
- `app/main.py` では `/notify` コマンドとレポート埋め込み (`ReportPaginator`) から `process_notification` を呼び出し、チケット (`tickets.status = notified`) を発行している。
- 期限到来後の自動削除は `ticket_watcher` が `TicketStore.fetch_due_tickets()` を用いて実行。キャンセル手段は未実装。
- `TicketStore` (`app/store.py`) には `register_notification`, `update_status`, `fetch_due_tickets`, `append_log` があるが、一覧取得/キャンセル専用 API がない。

## 概要設計
1. **データアクセス拡張**
   - `TicketStore` に以下を追加:
     - `fetch_active_tickets()` : `status = 'notified'` を抽出し、期限順に返す。
     - `append_log(...)` 既存を流用し、キャンセル記録用に利用。
     - `cancel_ticket(ticket_id, actor_id)` : `update_status(..., status='cancelled')` をラップし、ログ (`action='cancel'`) を残す補助関数として実装。
   - `tickets` テーブルのスキーマ変更は不要。`status` に `cancelled` を追加利用する想定。

2. **Discord UI**
   - Slash コマンド `/tickets` (仮) を追加。オプションとして `format` (`embed`/`csv`) を `/report` 命令と合わせるか検討（初期は embed のみでも可）。
   - `ReportPaginator` を再利用可能なように拡張するか、チケット専用ビュー `TicketPaginator` を新設:
     - `records` の代わりに `Ticket` と関連記録 (`find_record_for_message` の結果) を保持。
     - ボタン: 前/次, 「キャンセルする」, （必要ならログ転送）。
     - `キャンセル` ボタン押下で `TicketStore.cancel_ticket` を呼び出し、`TicketStore.append_log(action='cancelled')` と `send_ticket_log` を流用してモデログ通知。
   - 埋め込み構成: `build_record_embed` を流用し、`Ticket` から `record` を再構成 (`_ensure_record_defaults`) して表示。フッターに `due_at`, `status`, `ticket_id` を追加表示。

3. **コマンドハンドラ**
   - `/tickets` 実装フロー:
     1. `TicketStore.fetch_active_tickets()` でレコード取得。
     2. 件数ゼロならメッセージを返す。
     3. `TicketPaginator` を生成し、エフェメラルで表示。
   - `/tickets` の権限は `/report` 同様にギルド限定。必要なら `default_member_permissions` でモデレーター権限のみ許可。

4. **キャンセル操作**
   - キャンセルボタンで:
     1. `process_cancel_ticket` (新規ヘルパ) を呼び、`TicketStore.update_status(..., 'cancelled')`。
     2. `TicketStore.append_log` に `action='cancel'`, `detail='モデレーター操作'` 等を残す。
     3. `send_ticket_log` を使ってモデログへ通知 (`action="キャンセル"`, `result="モデレーターが取消"`)。
     4. UI 上で現在のカードを更新 or 次のカードへ移動。

5. **CSV 出力 (オプション)**
   - `/tickets` に `format=csv` を追加する場合、`write_report_csv` を流用しチケット用のフィールドを組む。
   - 初期リリースでは embed のみ対応し、CSV は将来拡張に回す案が手軽。

6. **状態遷移**
   - `notified` → `cancelled` （モデ操作）
   - `notified` → `bot_deleted` / `author_deleted` / `failed`（自動処理既存）
   - `cancelled` 状態のチケットはウォッチャーで対象外（`fetch_due_tickets` 条件が `status='notified'` のみのため追加対応不要）。

## 実装タスク
1. `TicketStore` 拡張
   - `fetch_active_tickets`, `cancel_ticket` を追加。
   - `TicketStore._row_to_ticket` / dataclass はそのまま利用。
2. `app/main.py`
   - `process_notification` 付近に `process_ticket_cancel` ヘルパーを追加。
   - `TicketPaginator` (新規 View) を実装。`ReportPaginator` のコードを参考に、`_build_embed` で埋め込み生成。
   - `/tickets` Slash コマンドを登録（`register_commands` 内）。
   - `send_ticket_log` を流用し、キャンセル時のログ出力を実装。
3. ロガー/メッセージ
   - キャンセル時にユーザーへエフェメラルメッセージで結果通知。
   - 既存ログ (`ticket_logs` と Discord notify) に `action='cancel'` を追加。
4. テスト/検証
   - 既存DBに `notified` チケットを手動挿入し、コマンドで一覧が表示されるか確認。
   - キャンセルボタンでステータスが `cancelled` へ更新され、ウォッチャー対象外になることを確認。
   - モデログへ通知が届くことを確認。

## 追加検討事項
- 権限管理: `/notify`, `/report` 同等のロールに限定するかどうか。
- 期限超過後にキャンセルする場合の扱い（実装上は実行可能だが、ログに "期限超過後" 等のメッセージを入れるか要検討）。
- 将来的な `format=csv` やフィルタ（日付・チャンネル別）機能の追加余地をコメントとして残す。

---
この計画に沿って実装を行う。
