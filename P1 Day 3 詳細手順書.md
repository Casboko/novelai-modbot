# Day 3 詳細実装計画書 — **ルールエンジン + `/scan` `/report`（暴力・ゴア/動物虐待も含む）**

目的：Day1/Day2 の出力（`p1_wd14.jsonl` + `p2_analysis.jsonl`）を入力に、**Discordの最新ガイドライン**に準拠した**画像モデレーション判定（赤 / 橙 / 黄）**を付与し、モデレーション運用に使う**スラッシュコマンド** `/scan`（判定実行）と `/report`（CSV出力）を実装する。**暴力・ゴア・動物虐待**は**年齢制限の有無に関係なく**拾い上げ対象（禁止領域）として組み込む。([Discord][1])

---

## 0) ポリシー基準（設計の土台）

* **暴力・グロ・動物虐待の共有は不可**（サーバー内のどこであっても）。Discordガイドライン #12 は、**暴力・ゴア・動物虐待を描写するコンテンツのアップロード/共有を禁止**としており、ポリシー解説でも **「実際の暴力・ゴア・動物虐待のコンテンツは許可しない」** を明記。よって**年齢制限(18+)での免除は無い**設計とする。([Discord][1])
* **成人向け（性的）コンテンツ**は**年齢制限チャンネルに限定**。アバター/サーバーアイコン/バナー/絵文字/ステッカーなど**年齢制限できない場所**では不可。([Discord][2])
* **年齢制限チャンネルの設定**は公式手順どおり（ただし上記の「どこでもNG」な領域は別）。([Discord Support][3])
* Slashコマンド（Application Commands）で `/scan` `/report` を提供する。([Discord][4])
* レート制限は**429レスポンスとバケット**を尊重（固定RPSを前提にせず、**公式の推奨運用**に従う）。([support-dev.discord.com][5])

---

## 1) 成果物（DoD）

* `app/rule_engine.py`：**評価器 `evaluate(analysis, is_nsfw_channel) -> {severity, reasons[]}`**
* `configs/rules.yaml`：**しきい値・タグ集合・外部APIオプション**
* スラッシュコマンド：`/scan`（対象スコープに判定付与）・`/report`（CSV/期間/Severityフィルタ）
* 出力：`out/p3_findings.jsonl`（判定付きレコード）＋ `out/p3_report.csv`
* 小規模E2E（数十〜数百枚）で**RED/ORANGE/YELLOW の割付が想定通り**、CSVがダウンロードできる

---

## 2) 入出力

**入力**：`p2_analysis.jsonl`（Day2の統合出力：WD14＋NudeNet＋xsignals）
**出力**：

* `p3_findings.jsonl` … `p2_analysis`に `severity` と `reasons[]` を付与
* `p3_report.csv` … レビュ用集計（主要列：`severity, message_link, author_id, is_nsfw_channel, wd14.rating.*, top_tags, nudenet_tops, violence_flags, reasons`）

---

## 3) ルール設計（RED / ORANGE / YELLOW）

### 3.1 RED（**即時・高優先レビュー**／“どこでもNG”領域）

* **暴力・ゴア・動物虐待**の強シグナル

  * WDタグ：`gore`, `blood`, `guro`, `severed_*`, `decapitation`, `guts`, `wound`, など**モデルの `selected_tags.csv` と一致する集合**のスコアがしきい値超（`gore ≥ 0.25` 等、後述）。
  * **オプション**：外部APIで高確信（例：**Vision SafeSearch** `violence ∈ {LIKELY, VERY_LIKELY}`、**Rekognition** `Graphic Violence or Gore`相当のラベル≥閾値、**Hive** の **Violence/Gore**クラス≥閾値）。([Google Cloud][6])
  * いずれか成立 → **RED**。**年齢制限の有無は問わない**（ガイドラインで禁止）。([Discord][1])
* **未成年の性的描写（実/架空/擬人化/AI生成含む）疑い**

  * WD14の**年少想起タグ**（`loli, shota, child, kid, toddler, teen, young` など）の合算が所定閾値超。**NudeNetの露出検出**は補助（年齢は推定しないため）。
  * 成立 → **RED**（どこでも禁止）。([Discord][1])

### 3.2 ORANGE（**配置違反の疑い**／18+外の成人向け）

* **成人向けコンテンツ**（WD14 `questionable`/`explicit` 高、またはNudeNetの `EXPOSED_*` 強）**かつ** チャンネルが **非18+**（`is_nsfw_channel=false`）

  * → **ORANGE**（年齢制限チャンネル外配置の疑い）。([Discord][2])

### 3.3 YELLOW（**参考レビュー**）

* **弱い露出**＋若年風特徴タグの複合／暴力性が**ゲーム/フィクション文脈の可能性**（テキスト無しの静止画では誤検知多→要人手確認）

  * → **YELLOW**（後段レビューへ）

> **方針**：判定は**拾い上げ（triage）**に徹し、**自動削除は実装しない**。REDでも**必ず人手レビュー**を経る。

---

## 4) 設定ファイル例（`configs/rules.yaml`）

```yaml
# どのモデル/ラベル名が来てもカスタマイズできるよう、集合を外だし
models:
  wd14_repo: "SmilingWolf/wd-eva02-large-tagger-v3"   # 記録用。v3はONNX/バッチ対応。:contentReference[oaicite:10]{index=10}

thresholds:
  # WD14 Ratings
  wd14_questionable: 0.35
  wd14_explicit:     0.20

  # WD14 minors（スコア合算）
  wd14_minors_sum:   0.40

  # Gore/Violence 初期値（タグ個別・合算の両面で）
  wd14_gore_tag_any: 0.25
  wd14_gore_sum:     0.35

  # NudeNet 強露出（補助信号）
  nudenet_exposed_strong: 0.60

  # 外部API（任意でON）
  gcv_violence_min: "LIKELY"        # SafeSearch violence ≥ LIKELY でRED扱い。:contentReference[oaicite:11]{index=11}
  rekognition_min_conf: 80          # DetectModerationLabels の confidence ≥ 80。:contentReference[oaicite:12]{index=12}
  hive_violence_min: 0.75           # Hive Violence/Gore ≥ 0.75。:contentReference[oaicite:13]{index=13}

sets:
  minors_like:
    - loli
    - shota
    - child
    - kid
    - toddler
    - teen
    - young
  nsfw_general:  # 配置違反の判別に使う代表タグ例（必要に応じ拡張）
    - bikini
    - lingerie
    - nude
    - panties
    - penis
    - breasts
  gore_like:
    - gore
    - blood
    - bloody
    - guro
    - decapitation
    - severed
    - guts
    - wound
features:
  use_hive: false
  use_gcv:  false
  use_rekognition: false
```

> **注意**：`gore_like`/`nsfw_general` は **WD14の `selected_tags.csv`** に存在するラベル名と一致するよう**運用でチューニング**する（モデルカード参照）。([Hugging Face][7])

---

## 5) ルールエンジン実装（擬似コード）

```python
def evaluate(ana, is_nsfw, cfg):
    reasons = []

    # ---- minors（どこでもNG）
    minors_score = sum(ana.wd14.general.get(t, 0.0) for t in cfg.sets.minors_like)
    if minors_score >= cfg.thresholds.wd14_minors_sum:
        return "red", reasons + [f"minors_like_sum={minors_score:.2f}"]

    # ---- gore（どこでもNG）
    gore_hits = {t: ana.wd14.general.get(t, 0.0) for t in cfg.sets.gore_like}
    gore_any = max(gore_hits.values() or [0.0])
    gore_sum = sum(gore_hits.values())
    if gore_any >= cfg.thresholds.wd14_gore_tag_any or gore_sum >= cfg.thresholds.wd14_gore_sum:
        return "red", reasons + [f"gore_any={gore_any:.2f}", f"gore_sum={gore_sum:.2f}"]

    # ---- 外部API（任意ON）
    if cfg.features.use_gcv and ana.gcv.violence in at_least(cfg.thresholds.gcv_violence_min):
        return "red", reasons + [f"gcv.violence={ana.gcv.violence}"]   # SafeSearch: violence ライクリ度。:contentReference[oaicite:15]{index=15}
    if cfg.features.use_rekognition and any(lbl in GORE_LABELS and conf>=cfg.thresholds.rekognition_min_conf for lbl,conf in ana.rekognition.labels):
        return "red", reasons + ["rekognition.gore"]                   # DetectModerationLabels。:contentReference[oaicite:16]{index=16}
    if cfg.features.use_hive and ana.hive.violence >= cfg.thresholds.hive_violence_min:
        return "red", reasons + [f"hive.violence={ana.hive.violence:.2f}"]  # Hive Violence/Gore。:contentReference[oaicite:17]{index=17}

    # ---- 配置違反（成人向け×非NSFW）
    rating = max(ana.wd14.rating.get("questionable",0), ana.wd14.rating.get("explicit",0))
    exposed = ana.xsignals.exposure_score
    if not is_nsfw and (rating >= cfg.thresholds.wd14_questionable or exposed >= cfg.thresholds.nudenet_exposed_strong):
        return "orange", reasons + [f"placement: rating={rating:.2f}, exposure={exposed:.2f}"]

    # ---- 参考レビュー（グレー）
    # 例：弱露出 + 若年風特徴
    if (ana.wd14.general.get("flat chest",0)+ana.wd14.general.get("petite",0))>0.6 and exposed>0.3:
        return "yellow", reasons + ["weak_exposure+youngish_features"]

    return "clean", []
```

---

## 6) スラッシュコマンド仕様

### `/scan` — 判定実行（P2出力に対して RED/ORANGE/YELLOW 付与）

* **引数**：`scope:<all|channel|since:ISO8601>`、`save:true|false`、`post_summary:true|false`
* **動作**：

  1. `p2_analysis.jsonl` を読み、指定スコープを抽出
  2. `rules.yaml` をロードして `evaluate()` 実行
  3. 結果を `p3_findings.jsonl` に追記（`save=false` なら一時ファイル）
  4. `post_summary=true` なら **#mod-bot-log** に件数サマリ投稿（RED/ORANGE/YELLOW 件数・上位タグ）
* **備考**：Slash は Application Commands（チャット入力型）として登録する。([Discord][4])

### `/report` — CSV出力（レビュ用）

* **引数**：`severity:<red|orange|yellow|all>`、`since:ISO8601?`、`limit:int?`
* **動作**：`p3_findings.jsonl` を読み、フィルタ後に **CSVファイルを添付**（`message_link` を必ず列に含む）
* **追加**：**暴力・ゴア列**（`violence_flags`）と**理由列**（`reasons`）を必須化

---

## 7) データモデル（`p3_findings.jsonl` レコード例）

```json
{
  "guild_id":"...", "channel_id":"...", "message_id":"...", "message_link":"...",
  "author_id":"...", "created_at":"...",
  "is_nsfw_channel": false,
  "wd14": { "rating": {...}, "general": {"blood":0.41, "bikini":0.77}, "character": {...} },
  "nudity_detections":[{"class":"EXPOSED_GENITALIA_F","score":0.88,"box":[...]}],
  "xsignals": { "exposure_score": 0.88, "placement_risk_pre": 0.72 },
  "severity": "red",
  "reasons": ["gore_any=0.41","gore_sum=0.41"]
}
```

---

## 8) 外部モデレーションAPI（任意ON）

* **Google Cloud Vision SafeSearch**：`adult/spoof/medical/violence/racy` の**5軸**で Likelihood を返す。`violence≥LIKELY` を RED 補助条件に。([Google Cloud][6])
* **AWS Rekognition**：`DetectModerationLabels` で **Graphic Violence or Gore** などのラベルと確信度。しきい値（例：conf≥80）で RED 補助。([AWS ドキュメント][8])
* **Hive**：**Violence/Gore** クラスのスコアを参照（0–1）。([docs.thehive.ai][9])

> いずれも **プライバシーポリシーに外部推論を明記**し、**保存なし・pHashのみログ**で運用する。

---

## 9) 実装ステップ（推奨順）

1. `configs/rules.yaml` 雛形作成（上記初期値）
2. `rule_engine.py` を実装（WD14/NudeNet/外部APIを吸収し `evaluate()` で判定）
3. `/scan` 実装：`p2_analysis.jsonl` を読み、判定付与→ `p3_findings.jsonl` 出力＋サマリ投稿
4. `/report` 実装：`p3_findings.jsonl` から CSV 生成（期間/Severityフィルタ、ジャンプリンク必須）
5. 小規模E2E（RED/ORANGE/YELLOW が意図通り出るかを検証）

---

## 10) テスト項目

* **RED（暴力/ゴア）**：`blood/gore` 強画像 → RED になるか（WD14タグ／外部APIが閾値超）([Discord][10])
* **RED（未成年性）**：`loli/shota/child/teen/young` 合算高 → RED
* **ORANGE**：成人向け（`questionable/explicit` or `EXPOSED_*`）× 非NSFW → ORANGE（**NSFWならORANGEにしない**）([Discord][2])
* **YELLOW**：弱露出＋若年風特徴 → YELLOW
* **Slash**：`/scan` がスコープ指定で走る、`/report` がCSVを返す（Application Commands として登録）([Discord][4])
* **レート制限**：大量 `/scan` 実行時も**429/バケット**を尊重して復帰（実運用想定）([support-dev.discord.com][5])

---

## 11) 運用メモ

* **タグ名の差異**：WD14（EVA02 v3）のタグ名称は**モデルカード/`selected_tags.csv`**準拠。運用で false positive/negative を見ながら `rules.yaml` の集合を**都度調整**。([Hugging Face][7])
* **ゲーム的暴力**：**赤判定の基準は「ショック目的のグロ・実際の暴力等」**に寄せる。WD14タグのみで過検知しがちな場合は、**Hive/GCV/Rekognition**の**補助ON**で緩和。([Google Cloud][6])
* **表でもOK**：`/report` で**REDだけ抽出**→**投稿者本人への削除依頼**は Day4 以降の `/notify` 実装で（本Dayは triage まで）。

---

## 12) 付録：仕様根拠

* **Community Guidelines**（#12：暴力・ゴア・動物虐待の共有禁止 / 未成年の性的内容禁止）([Discord][1])
* **Violence & Graphic Content Explainer**（「実際の暴力・ゴア・動物虐待は許可しない」明記）([Discord][10])
* **Age-Restricted**（成人向けは18+に隔離。年齢制限不可の場所では不可）([Discord][2])
* **Application Commands（Slash）**（Discord公式ドキュメント）([Discord][4])
* **Google Vision SafeSearch**（`adult/spoof/medical/violence/racy` の5軸）([Google Cloud][6])
* **AWS Rekognition**（DetectModerationLabels）/ **Hive**（Violence/Gore クラス）([AWS ドキュメント][8])

---

### 次アクション（このまま実装へ）

* `rules.yaml` をこの初期値で同梱 → `rule_engine.py` → `/scan` `/report` の順で実装
* 小規模チャンネルで E2E → RED/ORANGE/YELLOW バランスと**理由の可読性**をチェック
* 問題なければ **Day4（通知ワークフロー `/notify` `/remind` `/escalate`）** に接続します。

[1]: https://discord.com/guidelines "Community Guidelines | Discord"
[2]: https://discord.com/safety/360043653552-age-restricted-content-on-discord "Age-Restricted Content Policy | Discord Safety"
[3]: https://support.discord.com/hc/en-us/articles/115000084051-Age-Restricted-Channels-and-Content "Age-Restricted Channels and Content – Discord"
[4]: https://discord.com/developers/docs/interactions/application-commands "Discord Developer Portal"
[5]: https://support-dev.discord.com/hc/en-us/articles/6223003921559-My-Bot-is-Being-Rate-Limited?utm_source=chatgpt.com "My Bot is Being Rate Limited! - Developers - Discord"
[6]: https://cloud.google.com/vision/docs/detecting-safe-search?utm_source=chatgpt.com "Detect explicit content (SafeSearch) | Cloud Vision API"
[7]: https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3?utm_source=chatgpt.com "SmilingWolf/wd-eva02-large-tagger-v3"
[8]: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_DetectModerationLabels.html?utm_source=chatgpt.com "DetectModerationLabels - Amazon Rekognition"
[9]: https://docs.thehive.ai/docs/class-descriptions-violence-gore?utm_source=chatgpt.com "Visual Moderation - Violence, Weapons & Gore Classes"
[10]: https://discord.com/safety/violence-graphic-content-policy-explainer "Violence and Graphic Content Policy Explainer | Discord"
