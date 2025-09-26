# 非公式 NovelAI サーバー向け「過去画像モデレーション BOT」実装計画書（v0.9）

> 目的：**サーバー内の「過去に投稿された全画像」を機械走査し、規約抵触の疑いがある投稿だけをピックアップ → 投稿者本人へ削除依頼 → 期限超過でモデレーターが削除**という運用を、低負荷・高リコールで実現する。
> スコープ：**画像**（添付・埋め込み・絵文字・スタンプを含む）。テキストは今回は対象外（AutoMod 連携は付録）。
> タイムゾーン：Asia/Tokyo（JST）

---

## 1. 規約・権限・設計ポリシー

* **セルフボット禁止**：ユーザートークンでの自動化は不可。必ず **OAuth2/Bot** で実装。([Discordサポート][1])
* **削除に必要な権限**：他者メッセージの削除は **Manage Messages** が必要。([Discord][2])
* **メッセージ内容 Intent**：100サーバー以上のボットは *Message Content* が**特権 Intent**（審査対象）。100未満は開発者ポータルで有効化のみで可。今回の走査は主に**添付/埋め込みのメタ**を使うが、将来のテキスト規則に備え**ONを推奨**。([Discord 開発者サポート][3])
* **レート制御**：グローバル 50 req/s。HTTP 429 の `retry_after` に従いキュー制御。([Discord][4])
* **ジャンプリンク**：各ヒットには `https://discord.com/channels/{guild}/{channel}/{message}` を保存・提示（＝**返信・モデ操作の基点**）。([discord.js][5])
* **返信設計**：Bot の削除依頼は **message reply**（`message_reference`）で元投稿に直結、**allowed_mentions** で**投稿者のみ**に通知（`replied_user=false`）。([Discord][6])
* **スレッド対応**：アーカイブ済みも含め**一覧→必要なら join→返信**の順で処理。([Discord][7])

---

## 2. 成果物（Deliverables）

1. Discord Bot（Python/discord.py）
2. 画像解析ワーカー（WD14 タガー + NudeNet、オプションでクラウド二次判定）
3. ルールエンジン（赤/橙/黄の三段階）
4. 永続化（SQLite/CSV）・**メッセージジャンプリンク**・**返信テンプレ**を含むレポート
5. スラッシュコマンド UI（/scan, /notify, /remind, /escalate, /report）
6. 運用 SOP & 監査ログ

---

## 3. 機能要件（FR）

* **F1 収集**：全テキストチャンネル＋スレッド（**アーカイブ済み含む**）の**過去全履歴**を走査。メッセージの
  `attachments[*] (image/*)`、`embeds[*].image` を抽出。**カスタム絵文字/スタンプ**も別ジョブで取得。([discord.js][8])
* **F2 重複回避**：URL 正規化＋**pHash** キャッシュで既処理をスキップ。([GitHub][9])
* **F3 画像解析**：

  * **WD14 タガー**で Danbooru 系タグ・**rating**（general/sensitive/questionable/explicit）を推定。([Hugging Face][10])
  * **NudeNet** で露出部位検出（アニメ絵の補助判定）。([GitHub][11])
* **F4 ルール判定**：

  * **赤**：未成年を想起させるタグ（`loli, shota, child, kid, toddler, teen, young` など）が所定閾値以上。
  * **橙**：成人向け可能性（WD14: questionable/explicit または NudeNet 高スコア）**かつ** チャンネルが 18+ でない。
  * **黄**：弱い露出 + 若年風特徴タグの複合などグレー。
* **F5 依頼→期限→削除**：

  * ヒットごとに**ジャンプリンク**を付けて **元投稿へ返信**で「削除依頼」（@投稿者のみメンション）。**既定期限**（例：72h）を記録。
  * 期限超過で**存在確認→削除**（監査ログに理由）。([Discord][2])
* **F6 レポーティング**：CSV/JSON / **メッセージリンク** / 判定根拠（タグ・スコア） / ステータス（通知済・リマインド・投稿者削除・モデ削除）。
* **F7 再開可能**：チャンネルごとのカーソル（最古→新しい順）。中断・再開に強い。

---

## 4. 非機能要件（NFR）

* **N1** 高リコール優先（誤検知は人手レビューで吸収／自動削除はしない）
* **N2** データ最小化：画像のローカル保存は**最小限**（ハッシュ・URL・メタ中心）。Discord CDN URL は**期限がある場合がある**ため、必要時は再取得で更新。([Reddit][12])
* **N3** レート制御（グローバル 50 rps）・指数バックオフ。([Discord][4])
* **N4** 監査可能性：誰が・いつ・何に・何をしたか（削除依頼／削除／免責）。

---

## 5. アーキテクチャ

```
[Discord API] 
   │  (history, threads, emojis/stickers) 
   ▼
[Crawler]──┬──> [Dedup(pHash)]
            │
            └──> [Downloader] ──> [Analyzer] = [WD14] + [NudeNet] ( + optional Cloud )
                                      │
                                      ▼
                                 [Rule Engine]
                                      │
                                      ├──> [Queue: 赤/橙/黄]
                                      ├──> [Notifier: reply + allowed_mentions]
                                      └──> [Scheduler: remind/escalate(delete)]
                                     
[Store(SQLite/CSV)] <─── all events / jump links / due_at / status
[Slash Commands]  /scan /status /notify /remind /escalate /report
```

* **Threads（アーカイブ含む）**は API で列挙・参加可能。([Discord][7])
* **allowed_mentions / reply** で**投稿者のみに通知**しつつ、**元投稿に紐付いた会話**を維持。([discord-api-types documentation][13])

---

## 6. モジュール設計（Python / discord.py）

### 6.1 Crawler

* 全テキストチャンネル列挙 → `history(limit=None, oldest_first=True)` で遡及収集
* スレッド：`active+archived` をページング列挙 → 必要に応じ `join` → 収集
* 取得項目：`message_id, channel_id, guild_id, author_id, timestamp, attachments, embeds.image, is_nsfw_channel`

### 6.2 Dedup（pHash）

* `imagehash.phash(image)` を 64bit hex で保存。距離 ≤ 5 などで重複扱い。([GitHub][9])

### 6.3 Analyzer

* **WD14**：`SmilingWolf/wd-v1-4-*-tagger-*` のいずれか（ViT/SwinV2 等）。rating と年少関連タグを取得。([Hugging Face][10])
* **NudeNet**：露出部位検出（confidence 出力）。([GitHub][11])
* （任意）クラウド二次判定は後日拡張点としてフックを用意。

### 6.4 Rule Engine

* 赤/橙/黄 に分類。各ヒットに**ジャンプリンク**を付与（ `https://discord.com/channels/{guild}/{channel}/{message}` ）。([discord.js][5])

### 6.5 Notifier / Scheduler

* **削除依頼**：

  * `message.reply()`（内部で `message_reference`）＋ `AllowedMentions(users=[author], replied_user=False)`。([Discord][6])
  * 期限 `due_at`（既定 72h）を保存。
* **期限管理**：cron/scheduler が `due_at < now` の未対応を**存在確認→削除**。必要権限は **Manage Messages**。([Discord][2])

### 6.6 Storage（SQLite 例）

* `hits(message_id PK, channel_id, guild_id, author_id, ts, jump_link, tags, rating, nude_scores, rule, due_at, status)`
* `hash_cache(attachment_id PK, phash, first_seen)`
* `cursors(channel_id PK, last_message_id)`

---

## 7. スラッシュコマンド仕様（抜粋）

* `/scan start scope:<all|channel|since:date>` … 初回フル or 差分走査
* `/scan status` … 進捗（件数/残）
* `/notify <message_link> [due:hours]` … 当該投稿に**返信**で依頼（テンプレ使用）
* `/remind <message_link>` … リマインド返信
* `/escalate <message_link>` … 期限超過で**削除**（監査理由を付与）
* `/report format:<csv|json>` … ヒット一覧を出力

---

## 8. 返信テンプレ（JST）

> @{user} こちらの投稿は Discord ガイドライン/当サーバールールに抵触する可能性があります。**ご本人による削除**をお願いします。
> 期限：{YYYY-MM-DD HH:mm JST}
> ※期限までに対応がない場合、モデレーターが削除します。（この返信スレッドでご連絡ください）

※ 返信は**元投稿に紐付く**ため、**後から見返しやすい**／通知も届きやすい。([Discord][6])

---

## 9. しきい値（初期案・運用で調整）

* **WD14 rating**：`questionable >= 0.35` or `explicit >= 0.20` → 「橙」基準
* **年少ワード集合**（EN/JA）score 合計 ≥ `T_minor` → 「赤」
* **NudeNet**：局所スコア ≥ `0.6` が N 箇所以上 → 「橙」
* **黄→手動レビュー**：弱い露出 + 若年風タグ（例：`flat_chest + petite + small_shoulders` など）の複合

---

## 10. テスト計画

* **小規模チャンネルでパイロット**（100〜300 画像）→ **混同行列**（TP/FP/FN）
* **しきい値のチューニング**：赤の再現率を最優先、橙/黄で誤検知を吸収
* **レート試験**：50 rps 上限内でのキュー挙動／429→`retry_after` 準拠確認。([Discord][4])
* **スレッド/アーカイブ**：一覧・join・返信の往復試験。([discord.js][8])

---

## 11. 導入手順

1. **Bot アプリ作成**（Developer Portal）→ Bot 追加 → **Message Content**（任意、将来用）を ON。([Discord 開発者サポート][3])
2. サーバーへ招待（Scopes: `bot`, `applications.commands` / Permissions: `View Channels`, `Read Message History`, `Manage Messages`）
3. 環境構築：Python 3.11+, `pip install discord.py imagehash pillow` + （WD14/NudeNet 用依存）
4. `.env` 設定（BOT_TOKEN, GUILD_ID, DUE_HOURS=72）
5. `/scan start` → 初回フルスキャン → `/report` でしきい値調整
6. 以降は **日次差分 + リクエスト駆動**（/notify, /escalate）

---

## 12. エッジケース／既知の制約

* **CDN URL 期限**：古い添付の直リンクが失効していることあり。**メッセージ再取得で最新 URL** を得る（不可なら取得不能としてログ）。([Reddit][12])
* **アニメ年齢推定**：機械判定は**確信に至らない**ため、**赤でも自動削除は行わない**（常に人手確認）。
* **外部送信**：初期構成は**ローカル解析のみ**。クラウド API を使う場合は**プライバシーポリシー**記載と限定運用。

---

## 13. データモデル（例：SQLite）

```sql
CREATE TABLE hits (
  message_id    TEXT PRIMARY KEY,
  channel_id    TEXT NOT NULL,
  guild_id      TEXT NOT NULL,
  author_id     TEXT NOT NULL,
  ts            TEXT NOT NULL,
  jump_link     TEXT NOT NULL,
  is_nsfw_chan  INTEGER NOT NULL,
  wd14_tags     TEXT,           -- JSON
  wd14_rating   TEXT,           -- JSON
  nudenet       TEXT,           -- JSON
  rule          TEXT CHECK(rule IN ('red','orange','yellow')),
  due_at        TEXT,
  status        TEXT CHECK(status IN ('notified','reminded','author_deleted','mod_deleted','dismissed'))
);

CREATE TABLE hash_cache (
  attachment_id TEXT PRIMARY KEY,
  phash         TEXT NOT NULL,
  first_seen    TEXT NOT NULL
);

CREATE TABLE cursors (
  channel_id    TEXT PRIMARY KEY,
  last_message_id TEXT
);
```

---

## 14. 依存関係・外部コンポーネント

* **discord.py**（Bot / reply / intents / threads 操作）
* **ImageHash**（pHash 重複検出）([GitHub][9])
* **WD14 タガー（Hugging Face）**：`SmilingWolf/wd-v1-4-*`（ViT/SwinV2 等）([Hugging Face][10])
* **NudeNet**（OSS 露出検出）([GitHub][11])

---

## 15. 運用 SOP（抜粋）

* **日次**：差分スキャン → 赤/橙を優先確認 → `/notify` 実行
* **48–72h**：未対応に `/remind`
* **72–120h**：未対応を `/escalate`（削除）
* **週次**：しきい値・NG 語彙リストを見直し、FP/FN をレビュー
* **障害時**：429 多発→キューの待機時間を自動延長（`retry_after` 準拠）([Discord 開発者サポート][14])

---

## 16. セキュリティ／プライバシー

* **最小保存**（URL・pHash・根拠タグ・ステータス中心）。画像本体は保存しない方針（必要時のみ短期キャッシュ）。
* **Bot トークン**は KMS/Secrets 管理。
* **セルフボット禁止**・**ユーザートークン不使用**を README と運用規約に明記。([Discordサポート][1])

---

## 17. 実装スニペット（重要部だけ）

```python
# jump link
def jump_link(gid, cid, mid) -> str:
    return f"https://discord.com/channels/{gid}/{cid}/{mid}"  # 公式形式（messageLink の仕様に一致）

# 削除依頼（元投稿に返信・投稿者のみメンション）
async def notify_delete_request(message, due_hours=72):
    from datetime import datetime, timedelta, timezone
    JST = timezone(timedelta(hours=9))

    allowed = discord.AllowedMentions(everyone=False, roles=False,
                                      users=[message.author], replied_user=False)  # allowed_mentions
    due_at = datetime.now(JST) + timedelta(hours=due_hours)
    content = (
        f"{message.author.mention} この投稿はルール抵触の可能性があります。**ご本人での削除**をお願いします。\n"
        f"対象: {jump_link(message.guild.id, message.channel.id, message.id)}\n"
        f"期限: {due_at:%Y-%m-%d %H:%M} JST\n"
        "※期限までに対応がない場合、モデレーターが削除します。"
    )
    await message.reply(content=content, allowed_mentions=allowed)  # message_reference 返信
    return due_at
```

* `message.reply(...)` は内部で **message_reference** を付与（返信）。([Discord][6])
* `AllowedMentions(..., replied_user=False)` で**返信時の自動メンション抑止**。([discord.js][15])

---

## 18. ロードマップ

* **P0（〜1日）**：スキャナ最小実装（履歴→添付抽出→pHash→CSV 出力）
* **P1（〜3日）**：WD14/NudeNet 組み込み・ルールエンジン・/scan /report
* **P2（〜2日）**：/notify /remind /escalate・期限管理
* **P3（運用）**：しきい値チューニング・AutoMod テキスト側の規則拡張（任意）

---

## 19. 付録：参考リンク（抜粋）

* **Rate Limits**（グローバル 50 rps / 429 対応）([Discord][4])
* **Message Content Intent（特権）**([Discord 開発者サポート][3])
* **セルフボット禁止**（サポート記事 / ガイドライン）([Discordサポート][1])
* **メッセージ・リンク形式**（`messageLink`）([discord.js][5])
* **返信・メンション抑制**（allowed_mentions / replied_user）([discord-api-types documentation][13])
* **Threads（アーカイブ列挙・参加）**([Discord][7])
* **WD14 タガー** / **NudeNet**（OSS モデル）([Hugging Face][10])
* **Discord CDN の添付リンク**（期限/再取得の議論）([Reddit][12])

---

### 次アクション（提案）

1. 既定期限（例：72h or 120h）と**依頼テンプレ文**の確定
2. パイロット対象チャンネルの指定（テスト許容件数）
3. しきい値（赤/橙/黄）初期値の採択

この計画でそのまま着工できます。必要なら、`requirements.txt` / `.env.example` / 最小 CLI（`/scan start` 相当）まで一式をすぐに用意します。

[1]: https://support.discord.com/hc/en-us/articles/115002192352-Automated-User-Accounts-Self-Bots?utm_source=chatgpt.com "Automated User Accounts (Self-Bots)"
[2]: https://discord.com/community/permissions-on-discord-discord?utm_source=chatgpt.com "Permissions on Discord"
[3]: https://support-dev.discord.com/hc/en-us/articles/4404772028055-Message-Content-Privileged-Intent-FAQ?utm_source=chatgpt.com "Message Content Privileged Intent FAQ - Developers - Discord"
[4]: https://discord.com/developers/docs/topics/rate-limits?utm_source=chatgpt.com "Rate Limits | Documentation | Discord Developer Portal"
[5]: https://discord.js.org/docs/packages/discord.js/14.19.2/messageLink%3AFunction?utm_source=chatgpt.com "messageLink (discord.js - 14.19.2)"
[6]: https://discord.com/developers/docs/resources/message?utm_source=chatgpt.com "Messages Resource | Documentation"
[7]: https://discord.com/developers/docs/topics/threads?utm_source=chatgpt.com "Threads | Documentation | Discord Developer Portal"
[8]: https://discord.js.org/docs/packages/discord-api-types/main/v10/RESTGetAPIChannelUsersThreadsArchivedResult%3AInterface?utm_source=chatgpt.com "RESTGetAPIChannelUsersThrea..."
[9]: https://github.com/JohannesBuchner/imagehash?utm_source=chatgpt.com "JohannesBuchner/imagehash: A Python Perceptual Image ..."
[10]: https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2?utm_source=chatgpt.com "SmilingWolf/wd-v1-4-vit-tagger-v2"
[11]: https://github.com/vladmandic/nudenet?utm_source=chatgpt.com "NudeNet: NSFW Object Detection for TFJS and NodeJS"
[12]: https://www.reddit.com/r/DataHoarder/comments/16zs1gt/cdndiscordapp_links_will_expire_breaking/?utm_source=chatgpt.com "cdn.discordapp links will expire, breaking thousands of ..."
[13]: https://discord-api-types.dev/api/discord-api-types-v10/enum/AllowedMentionsTypes?utm_source=chatgpt.com "AllowedMentionsTypes | API | discord-api-types documentation"
[14]: https://support-dev.discord.com/hc/en-us/articles/6223003921559-My-Bot-is-Being-Rate-Limited?utm_source=chatgpt.com "My Bot is Being Rate Limited! - Developers - Discord"
[15]: https://discord.js.org/docs/packages/discord.js/14.19.2/APIAllowedMentions%3AInterface?utm_source=chatgpt.com "APIAllowedMentions (discord.js - 14.19.2)"