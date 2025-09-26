# P1 カード改修 — 補強版 詳細実装計画（Remind/Delete撤去版）

## 1. 目的と背景

`/report` のエンベッドを**監査者が1秒で判断**できる最小情報 UI に再設計する。

* 画像は**初期から展開**（クリック不要）。Embed の仕様上、**大画像は本文下部固定／サムネは右上固定**で位置のカスタム不可。必要なら画像専用Embed追加で回避する。 ([GitHub][1])
* **リンク導線はボタンのみ**に一本化。
* 指標は**exposure / violence / minors / animals**の4軸に集約。
* **Remind／Delete ボタンは撤去**（仕様として不適切な手動オペを排し、誤操作経路を削除）。
* **ActionRow は最大5行×各行最大5要素**の上限を遵守。行は `row`（0–4）で明示制御。 ([Discord][2])

---

## 2. レイアウト / UI 方針

### 2.1 画像

* `Embed.set_image(url=...)` を**常時設定**（Compact/Expand トグル廃止）。
* 外部CDNの期限切れに備え、将来タスクとして**Bot 添付（`attachment://`）運用**も選択肢。 ([Discord.jsガイド][3])

### 2.2 テキスト

* タイトル：`[SEVERITY] rule_title`（踏襲）
* 日付：UTCで

  ```
  YYYY-MM-DDT
  HH:MM:SSZ
  ```
* **削除**：`🔗 Jump` / `🧮 Ratings` / `🔖 Tags`。

### 2.3 ボタン配置（行固定）

* **row=0**：`◀` `▶`（ページャ）
* **row=1**：`Open Message`（常時）・`Open Original`（原画像URLがある時のみ有効）

  * 外部公開URLは**期限付き（署名・有効期限付きURL）**に移行済みのため、失効時は**メッセージ再取得で新しいURL**を使う方針。 ([GitHub][4])
* **row=2**：`Notify`・`Log`（**Remind／Delete は撤去**）

---

## 3. 指標とデータ処理

### 3.1 NSFW ブロック

* **nsfw Σ**：`configs/rules.yaml` の `nsfw_general_tags`（約25種）に一致する WD14 *general* タグの**スコア合計**（0.0–1.0 目安）。 ([Reddit][5])

### 3.2 Exposure ブロック（4軸）

1. **exposure**：NudeNet `EXPOSED_*` から**最大露出スコア**（正規化）。 ([Reddit][6])
2. **violence**：暴力・流血系タグの**最大値**。
3. **minors**：未成年疑いタグの**合計値**。
4. **animals**（新設）：動物虐待・獣姦系タグの**合計値**。

   * `animal_abuse_tags` 例：`animal_abuse, animal_cruelty, animal_blood, animal_gore, bestiality, zoophilia, zoosadism`（誤綴 `beastiality` は採用しない）。

---

## 4. 実装ステップ（差分）

### 4.1 コンフィグ

* `configs/rules.yaml` に `animal_abuse_tags` を新設（上記例）。
* 必要なら ORANGE/RED への連動条件をパラメタ化。

### 4.2 RuleEngine

* WD14出力から `animal_abuse_tags` を抽出し、`metrics.animals_sum`（＋必要なら `animals_max`）を算出・格納。
* 既存の `violence_max` / `minors_sum` と同等の実装パターン。

### 4.3 カード生成（`build_record_embed`）

* **削除**：Ratings／Tags／Jump。
* **変更**：日付を ISO簡略2行（UTC）、Exposure を4軸に差替え、画像を常時 `set_image`。
* **リンク**：`Open Original` は**原画像URLがあるときのみ** enable。期限切れ時は**メッセージ再取得で新URL**。 ([BleepingComputer][7])

### 4.4 ビュー（`ReportPaginator`）

* **モード切替（Compact/Expand）ロジック削除**。
* **Remind／Delete のコンポーネント登録を削除**し、**row=2 は `Notify`・`Log` のみ**に再配置。
* ページャは **row=0**、リンク群は **row=1** に固定。 （5行×各5要素の制約内） ([Discord][2])

### 4.5 CSV（任意）

* `animals_sum` 列を末尾に追加（互換維持）。

---

## 5. ボタン仕様
> ※ **Remind／Delete は本カードから撤去**：
>
> * Remind＝運用上は自動化・個別タスク化が適切のためUIから削除。
> * Delete＝誤操作防止・監査追跡性確保のためUIから削除（必要なら運用手順で対応）。

---

## 6. テスト計画

### 6.1 表示/UI

* 画像が**初期から展開**され、トグルが存在しない。 ([GitHub][1])
* `◀ ▶` が **row=0** で隣接。
* `Open Message`（常時）／`Open Original`（URL有時のみ）が **row=1**。
* **row=2** に `Notify` と `Log` のみ（Remind／Delete が無い）。

### 6.2 指標

* Exposure 4軸が `exposure / violence / minors / animals` の順で正しく表示。
* `nsfw Σ` の定義テキストが最小表示（冗長説明なし）。

### 6.3 リンク失効・例外

* `Open Original` が失効した場合でも、**メッセージ再取得で新URL**により復旧できる。 ([BleepingComputer][7])

---

## 7. 既知制約と対応

* **Embed レイアウト固定**（大画像は下部／位置変更不可）。必要なら**画像専用Embed**を追加送信。 ([GitHub][1])
* **コンポーネント上限**（5行×5要素）に適合。今回の行設計は余裕あり。 ([Discord][2])
* **CDNリンクの期限付き化**：外部共有URLは**署名付き・期限付き**になっているため、失効時は**API経由で再取得**する。 ([GitHub][4])

---

## 8. 受け入れ基準（DoD）

* `/report` 実行時：**Ratings／Tags／Jump が出ない**。
* 画像は**常時展開**（トグルなし）。
* Exposure 4軸は所定順序で表示。
* ボタン行：**row=0 = ページャ**, **row=1 = Open Message / Open Original**, **row=2 = Notify / Log**。
* **Remind／Delete のコンポーネントが存在しない**（handler も未登録）。
* `Open Original` はURL有時のみ有効、失効時に**再取得で復旧可能**。 ([BleepingComputer][7])

---

## 9. 変更点サマリ（実装者向けチェックリスト）

* [ ] `configs/rules.yaml` に `animal_abuse_tags` を追加
* [ ] RuleEngine：`animals_sum`（＋必要なら `animals_max`）を算出・格納
* [ ] `build_record_embed`：Ratings/Tags/Jump削除、日付整形、Exposure差替え、画像常時 `set_image`
* [ ] `ReportPaginator`：**Remind/Delete のボタン・ハンドラ削除**、行を `row=0/1/2` に固定
* [ ] （任意）CSV末尾に `animals_sum` 追加

---

### 参照（主要根拠）

* **Action Row 上限・行制御**：Discord Developer Portal（Components）([Discord][2])
* **Embed画像位置（下部固定）**：discord-api-docs の議論（仕様背景）([GitHub][1])
* **添付URLの期限付き化／再取得**：APIドキュメント更新（コミット言及）＋報道／解説記事 ([GitHub][4])
* **`attachment://` を埋め込みに参照**（将来タスクの根拠）([Discord.jsガイド][3])
* **`jump_url`（メッセージ直リンク）**：discord.py API リファレンス ([discord.py][8])

---

[1]: https://github.com/discord/discord-api-docs/discussions/4258?utm_source=chatgpt.com "Ability to specify positioning for embed image #4258"
[2]: https://discord.com/developers/docs/components/reference?utm_source=chatgpt.com "Component Reference | Documentation"
[3]: https://discordjs.guide/popular-topics/embeds.html?utm_source=chatgpt.com "Embeds"
[4]: https://github.com/discordeno/discordeno/issues/3430?utm_source=chatgpt.com "[api-docs] Add Attachment CDN params (#6650) #3430"
[5]: https://www.reddit.com/r/discordapp/comments/17nsrhc/psa_discord_cdn_links_will_become_temporary_by/?utm_source=chatgpt.com "Discord CDN links will become temporary by end of year!!"
[6]: https://www.reddit.com/r/learnpython/comments/mdpyau/how_could_i_go_about_distinguishing_between/?utm_source=chatgpt.com "How could I go about distinguishing between replies and ..."
[7]: https://www.bleepingcomputer.com/news/security/discord-will-switch-to-temporary-file-links-to-block-malware-delivery/?utm_source=chatgpt.com "Discord will switch to temporary file links to block malware ..."
[8]: https://discordpy.readthedocs.io/en/latest/interactions/api.html?utm_source=chatgpt.com "Interactions API Reference - discord.py"
