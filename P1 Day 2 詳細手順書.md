# Day 2 詳細実装計画書 — **NudeNet 統合 & クロスシグナル整備**（WD14×NudeNet）

目的：Day1 で得た **WD14（EVA02 v3）** の推定結果に、**NudeNet の露出部位検出**（部位ラベル＋確信度）を加え、Day3 のルールエンジンで使える **共通スキーマ（Analysis）** と **クロスシグナル**（併合特徴量）を整える。
前提：`p0_scan.csv`（画像URL群）と `p1_wd14.jsonl`（WD14タグ結果）が出力済み。

---

## 0) 仕様の要点（根拠）

* **NudeNet** は Python パッケージで、`NudeDetector().detect()` / `detect_batch()` を提供（パス／OpenCV画像／画像バイト列などを受け取る）。返り値は**検出配列（クラス名・スコア・矩形）**。([PyPI][1])
* **検出クラス**は多数（例：`EXPOSED_BREAST_F/M`, `EXPOSED_GENITALIA_F/M`, `EXPOSED_BUTTOCKS`, `EXPOSED_ANUS`, `EXPOSED_ARMPITS` など）。バージョン 2.0.1 の**デフォルトクラス一覧**が公開されている。([PyPI][2])
* **WD14（EVA02 v3）** は Ratings（`general/sensitive/questionable/explicit`）＋ General/Character タグを返すモデルカードが公開されている（Day1 実装で参照）。([Hugging Face][3])

> 注意：NudeNet は**露出部位の検出器**で年齢判定器ではない。**未成年疑い**は引き続き **WD14の年少想起タグ群**（例：`loli/shota/child/teen/young` 等）で扱う設計とする（Day3 でルール化）。

---

## 1) 成果物（DoD）

* `app/analyzer_nudenet.py`：**バッチ推論**＋**スキーマ正規化**（`nudity_detections[]`）
* `out/p2_analysis.jsonl`：**WD14＋NudeNet 統合**（1画像1行）
* `out/p2_metrics.json`：処理枚数／平均レイテンシ／失敗件数
* **クロスシグナル計算**（例：`exposure_score`, `placement_risk_pre`）の実装（rules 前段）
* 同一 pHash の**再推論回避（キャッシュ）**

---

## 2) 依存関係と環境

```bash
pip install nudenet  # 公式PyPI
```

* `detect` / `detect_batch` の受け取り型／戻り値の形は PyPI に記載。([PyPI][1])
* バージョンによってはクラス集合が増減するため、**実行時にモデルのラベル集合をダンプ**しておく（検証容易化）。標準クラスの例は 2.0.1 の PyPI ページに明記。([PyPI][2])

> 代替：もし `nudenet` の入手に問題が出た場合は、`NudeNetv2`（v2 ブランチのフォーク）を一時利用可（**将来は戻す**）。([PyPI][4])

---

## 3) データフローとディレクトリ

```
modbot/
  app/
    analyzer_nudenet.py     # ★ NudeNet 推論 & 正規化
    analysis_merge.py       # WD14とNudeNetのマージ & クロス特徴算出
    schema.py               # 共通スキーマ定義
    cache_nudenet.sqlite    # pHashベースの結果キャッシュ
  out/
    p1_wd14.jsonl           # Day1 出力（既存）
    p2_analysis.jsonl       # ★ 統合出力（WD14 + NudeNet）
    p2_metrics.json
```

---

## 4) スキーマ（共通・将来互換）

`p2_analysis.jsonl`（1行=1画像）

```json
{
  "guild_id":"...", "channel_id":"...", "message_id":"...", "message_link":"...",
  "phash":"...", "is_nsfw_channel": false,
  "source":"attachment|embed",
  "wd14": {
    "rating":{"general":0.1,"sensitive":0.1,"questionable":0.6,"explicit":0.2},
    "general":{"blood":0.41,"gore":0.12,"bikini":0.77},
    "character":{"some name":0.91}
  },
  "nudity_detections":[
    {"class":"EXPOSED_BREAST_F","score":0.92,"box":[x0,y0,x1,y1]},
    {"class":"EXPOSED_GENITALIA_F","score":0.88,"box":[...]}
  ],
  "xsignals": {
    "exposure_score": 0.88,
    "placement_risk_pre": 0.72
  },
  "meta": { "nudenet_version":"2.0.1", "wd_model":"SmilingWolf/wd-eva02-large-tagger-v3" }
}
```

* `nudity_detections[].class` は NudeNet の**公開クラス名**をそのまま保存（後で rules.yaml でマッピング）。([PyPI][2])

---

## 5) NudeNet 推論（実装詳細）

### 5.1 API の使い方（根拠つき）

* 初期化：`from nudenet import NudeDetector; detector = NudeDetector()`
* 単発：`detector.detect(image)`／バッチ：`detector.detect_batch([image1, image2, ...])`
* 引数は**ファイルパス／OpenCV画像／画像バイト／バッファ**いずれも可。返り値は**画像ごとの検出配列**。([PyPI][1])

### 5.2 主要クラスの扱い（初期セット）

* **強シグナル（露出）**：
  `EXPOSED_GENITALIA_F/M`, `EXPOSED_BREAST_F/M`, `EXPOSED_BUTTOCKS`, `EXPOSED_ANUS`（スコア閾値：例 0.6）
* **弱シグナル（挑発/覆い）**：
  `COVERED_BREAST_F`, `COVERED_GENITALIA_F`, `COVERED_BUTTOCKS`, `EXPOSED_BELLY` など（閾値：例 0.5）
* **補助**：`FACE_F/M`, `EXPOSED_ARMPITS`, `EXPOSED_FEET` など（閾値：例 0.7; 直接の赤判定には使わない）

> クラス名・説明は PyPI に列挙（Default／Base いずれも）。実際は `detector` が返す *class* 文字列をそのまま記録し、**rules 側で集合にマップ**する。([PyPI][2])

### 5.3 実装ポイント

* **バッチ化**：`detect_batch()` を既定。大きすぎる画像は**長辺 1024〜1536**程度で一時リサイズ（速度・メモリと検出精度のバランス）。
* **キャッシュ**：キー=`(phash, nudenet_model_id)`。**同一画像は再推論しない**。
* **失敗処理**：画像取得不可（CDN失効等）は `note=fetch_failed`。URL 再取得は P0 側の再フェッチで対処。

---

## 6) クロスシグナル（WD14 × NudeNet の併合特徴量）

Day3 のルールが設計しやすいよう、**数値化された前処理スコア**をここで計算しておく。

### 6.1 例：`exposure_score`（成人向けの可能性・露出強度）

```
exposure_score =
  max( score(EXPOSED_GENITALIA_*), score(EXPOSED_BREAST_*), score(EXPOSED_BUTTOCKS), score(EXPOSED_ANUS) ) * 1.0
  ∨ 0.6 * max(score(COVERED_GENITALIA_*, COVERED_BREAST_*, COVERED_BUTTOCKS))
```

* **根拠**：NudeNet のクラスは「露出/覆い」を区別して返すため、露出系クラスを強く評価。クラス名は公開一覧に準拠。([PyPI][2])

### 6.2 例：`placement_risk_pre`（NSFWチャンネル外でのリスク“予測”）

```
placement_risk_pre =
  max( wd14.rating.questionable, wd14.rating.explicit ) * 0.5
+ topK_mean( wd14.general ∩ {nsfw傾向タグ群} ) * 0.3
+ exposure_score * 0.7
```

* Day3 で `is_nsfw_channel == false` と合わせて \*\*配置違反疑い（橙）\*\*へ昇格しやすくする。WD14 は NSFW 傾向タグ（例：`bikini`, `lingerie`, など）と **Ratings** を持つ。([Hugging Face][3])

> ※最終の「赤/橙/黄」判定は Day3。ここでは\*\*（数値の）特徴量\*\*だけ作る。

---

## 7) 設定ファイル

`configs/nudenet.yaml`（例）

```yaml
version: "2.0.1"
thresholds:
  strong_exposed: 0.60   # EXPOSED_GENITALIA_*, EXPOSED_BREAST_*, EXPOSED_BUTTOCKS, EXPOSED_ANUS
  weak_exposed:   0.50   # COVERED_* 系
keep_topk: 10            # 画像あたり最大10件保持
batch_size: 8
```

`configs/xsignals.yaml`（例）

```yaml
exposure_score:
  strong_weight: 1.0
  weak_weight:   0.6
placement_risk_pre:
  rating_weight: 0.5
  general_weight: 0.3
  exposure_weight: 0.7
```

---

## 8) 実装スケッチ

```python
# analyzer_nudenet.py
from nudenet import NudeDetector  # PyPI
# detector.detect_batch([...]) -> list[list[{label/score/box}]]  を前提に正規化。:contentReference[oaicite:11]{index=11}

def run_nudenet_batch(images):  # images: List[bytes or path]
    det = NudeDetector()
    raw = det.detect_batch(images)  # [[{box:..., score:..., label:...}, ...], ...]
    out = []
    for dets in raw:
        arr = [{"class":d.get("label") or d.get("class"),
                "score":float(d["score"]),
                "box":d.get("box")} for d in dets]
        out.append(arr)
    return out
```

> 検出ラベル集合は PyPI の公開一覧にあるため、**`class` 文字列をそのまま保存**しておけば後工程で柔軟に集合化できる。([PyPI][2])

---

## 9) パイプライン統合

1. `p1_wd14.jsonl` を読み込み（WD14 部はそのまま）
2. まだ NudeNet 未処理の行について、画像をロード → `detect_batch` 実行
3. `nudity_detections[]` を追加し、`xsignals.*` を計算
4. `p2_analysis.jsonl` と `p2_metrics.json` を出力（処理/失敗カウント・平均ms）

> WD14 側の根拠（モデルカード／更新日付）は HF モデルカードを出力 `meta.wd_model` に記録する。([Hugging Face][3])

---

## 10) テスト計画

* **単体**

  * `detect_batch` の I/O 形が **PyPI 記載どおり**か（パス／バイトで試す）。([PyPI][1])
  * クラス名の例が **PyPI 掲載の標準集合**に含まれるか（`EXPOSED_*`, `COVERED_*` など）。([PyPI][2])
* **E2E（小規模）**

  * 10〜20枚で `p2_analysis.jsonl` が生成され、`nudity_detections[]` に妥当なクラス・スコアが入っている
  * `xsignals.exposure_score` が `EXPOSED_*` 検出で上がること
* **性能**

  * バッチサイズを 4/8/16 で比較、平均レイテンシを `p2_metrics.json` に記録

---

## 11) 既知の制約・運用注意

* **アニメ絵**では NudeNet の露出検出が**過検知/過小検知**する場合がある（訓練分布差）。よって **“自動削除”に使わず、拾い上げシグナル**として扱う（Day3 の設計方針通り）。
* **CDN URL 失効**時は `fetch_failed` としてログし、P0 の再フェッチで URL 更新（既定運用）。
* **キャッシュ**：pHash 衝突は理論上ありうるため、\*\*URL＋寸法（w×h）\*\*もキャッシュキー補助に使うことを推奨。

---

## 12) 参考（一次情報）

* **NudeNet（PyPI）**：API（`detect`/`detect_batch` 引数・戻り値の説明）([PyPI][1])
* **NudeNet 検出クラス一覧（v2.0.1）**：`EXPOSED_*`/`COVERED_*`/`FACE_*` などの定義と説明。([PyPI][2])
* **WD14（EVA02 v3）モデルカード**：Ratings/Tags/学習データの要点（Day1の仕様根拠）。([Hugging Face][3])

---

## 13) 次アクション（実装順）

1. `pip install nudenet` → `analyzer_nudenet.py` を実装（`detect_batch` 正規化）([PyPI][1])
2. `analysis_merge.py` で WD14＋NudeNet のマージと `xsignals` 計算
3. 小規模 E2E 実行 → `p2_analysis.jsonl` / `p2_metrics.json` 確認
4. ここまでの成果を Day3（ルールエンジン＋/scan＋/report）に受け渡し

この計画に沿えば、**Day3** でそのまま **赤/橙/黄** へルール化できます（グロ・ゴア・動物虐待の扱いも Day3 に実装）。必要なら **雛形コード**を切り出します。

[1]: https://pypi.org/project/nudenet/?utm_source=chatgpt.com "nudenet - PyPI"
[2]: https://pypi.org/project/nudenet/2.0.1/?utm_source=chatgpt.com "nudenet 2.0.1 - PyPI"
[3]: https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3?utm_source=chatgpt.com "SmilingWolf/wd-eva02-large-tagger-v3"
[4]: https://pypi.org/project/NudeNetv2/?utm_source=chatgpt.com "NudeNetv2"
