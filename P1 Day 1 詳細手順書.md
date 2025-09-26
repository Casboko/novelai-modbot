# P1 Day 1 詳細手順書 — **WD14 統合（タグ推定レイヤ / キャッシュ基盤）**

目的：P0で得た `p0_scan.csv`（画像URL/メタ）を入力に、**WD14** で
**rating（general / sensitive / questionable / explicit）＋一般タグ＋キャラタグ**を推定する最小パイプラインを構築。ONNX Runtime（CPU）を既定とし、**OpenVINO EP** での加速は任意。モデル/閾値/前処理は **WD 1.4 SwinV2 Tagger v2.1** の公開実装・カードに準拠。([Hugging Face][1])

---

## 0) 仕様の要点（根拠）

* **モデル**：SmilingWolf *wd-v1-4-swinv2-tagger-v2*（ONNX）。**ORT ≥ 1.17.0** 必須。**バッチ次元は可変（1固定ではない）**。**P=R しきい値 ≈ 0.3771**。([Hugging Face][1])
* **ラベル定義**：`selected_tags.csv` の `category` 列で **rating=9 / general=0 / character=4** を識別。スペース実装は **general=0.35 / character=0.85** を既定閾値に採用。([Hugging Face][2])
* **前処理**（スペース実装に準拠）：**白地で正方形パディング → 入力解像度へリサイズ → RGB→BGR → NHWC float32、バッチ次元追加**。([Hugging Face][2])
* **ダウンロード**：`huggingface_hub.hf_hub_download()` で `model.onnx` と `selected_tags.csv` を取得（ローカルキャッシュ対応）。([Hugging Face][3])
* **性能チューニング**：

  * ORT の **intra/inter op threads** を明示設定。([ONNX Runtime][4])
  * **OpenVINO EP** を使う場合は `pip install onnxruntime-openvino`、Linuxは wheel に OpenVINO 付属。高位最適化は **OVEP利用時は無効化推奨**。([ONNX Runtime][5])

---

## 1) ディレクトリ & ファイル

```
modbot/
  app/
    analyzer_wd14.py        # ★ 本日作成
    labelspace.py           # CSV→インデックス（rating/general/character）
    batch_loader.py         # 画像URL→bytes（P0のfetchユーティリティを再利用）
    cache_wd14.sqlite       # pHashベースの結果キャッシュ（SQLite or JSONLでも可）
    cli_wd14.py             # CSV入力→WD14推定→JSONL出力
  models/
    wd14/                   # hf_hub_download で配置（model.onnx, selected_tags.csv）
  out/
    p1_wd14.jsonl           # ★ 1画像1行の推定結果
    p1_wd14_metrics.json    # 枚数/スループット/失敗件数
```

---

## 2) 依存の確定

```bash
pip install "onnxruntime>=1.17.0" huggingface_hub pandas pillow numpy
# 任意: Intel CPU最適化（OpenVINO EP）
pip install onnxruntime-openvino
```

* ORT≥1.17.0 は v2.1 ONNX の要件。OpenVINO EP の導入は `onnxruntime-openvino` を追加。([Hugging Face][1])

---

## 3) モデル & ラベルの取得

```python
# labelspace.py
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np

REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

def download_wd14(repo_id: str = REPO):
    csv = hf_hub_download(repo_id, LABEL_FILENAME)          # HFローカルキャッシュ
    onnx = hf_hub_download(repo_id, MODEL_FILENAME)
    return csv, onnx

def load_labelspace(csv_path: str):
    df = pd.read_csv(csv_path)
    # category: 9=rating, 0=general, 4=character
    idx_rating = list(np.where(df["category"] == 9)[0])
    idx_general = list(np.where(df["category"] == 0)[0])
    idx_character = list(np.where(df["category"] == 4)[0])
    names = (df["name"]
             .map(lambda x: x.replace("_"," ") if x not in ["0_0","(o)_(o)"] else x)
             .tolist())
    return names, idx_rating, idx_general, idx_character
```

* `hf_hub_download()` は**単一ファイルを取得し、バージョン管理されたローカルキャッシュへ格納**。([Hugging Face][3])
* ラベル区分の index 取得（`category` 列）はスペース実装と同等。([Hugging Face][2])

---

## 4) 推論セッション（CPU/OVEP 切替）

```python
# analyzer_wd14.py
import onnxruntime as ort
from PIL import Image
import numpy as np

class WD14Session:
    def __init__(self, model_path: str, provider: str = "cpu", threads: int = 0):
        so = ort.SessionOptions()
        # 必要に応じグラフ最適化/スレッド設定
        if threads > 0:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = max(1, threads // 2)
        providers = ["CPUExecutionProvider"]
        if provider.lower() == "openvino":
            # OVEP: ORT高位最適化の無効化が推奨されるケースあり
            # （OV側に最適化を委ねる） 
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        # 入力テンソル shape からターゲット解像度を取得（バッチ次元は可変）
        _, H, W, _ = self.sess.get_inputs()[0].shape
        self.size = int(H)

    def preprocess(self, pil_rgba: Image.Image) -> np.ndarray:
        # 白地で正方形パディング → リサイズ → RGB→BGR → NHWC float32、バッチ次元
        canvas = Image.new("RGBA", pil_rgba.size, (255, 255, 255))
        canvas.alpha_composite(pil_rgba)
        img = canvas.convert("RGB")
        max_dim = max(img.size)
        pad = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        pad.paste(img, ((max_dim - img.size[0]) // 2, (max_dim - img.size[1]) // 2))
        if max_dim != self.size:
            pad = pad.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(pad, dtype=np.float32)[:, :, ::-1]  # BGR
        return np.expand_dims(arr, axis=0)

    def infer_batch(self, pil_list):
        x = np.concatenate([self.preprocess(p) for p in pil_list], axis=0)
        input_name = self.sess.get_inputs()[0].name
        out_name = self.sess.get_outputs()[0].name
        preds = self.sess.run([out_name], {input_name: x})[0]  # shape: [B, n_tags]
        return preds
```

* **前処理**・**BGR**・**NHWC**・**入力解像度はモデル側 shape から読む**という流儀は HF スペース実装に準拠。([Hugging Face][2])
* ORT の **スレッド**は `intra_op_num_threads` 等で調整可能。([ONNX Runtime][4])
* **OpenVINO EP** を使う場合の導入と注意点（Linux wheel / 最適化の委譲）は公式ガイドに沿う。([ONNX Runtime][5])

---

## 5) 推定値 → ラベル空間へマッピング

```python
# analyzer_wd14.py（続き）
from dataclasses import dataclass

@dataclass
class WD14Config:
    thresh_general: float = 0.35   # スペース既定
    thresh_character: float = 0.85 # スペース既定
    use_mcut_general: bool = False
    use_mcut_character: bool = False

def mcut_threshold(probs: np.ndarray) -> float:
    # HFスペースと同等のMCut実装
    s = np.sort(probs)[::-1]
    if len(s) < 2: 
        return 1.0
    difs = s[:-1] - s[1:]
    t = np.argmax(difs)
    return float((s[t] + s[t+1]) / 2)

def decode_logits(logits_row, names, idx_rating, idx_general, idx_character, cfg: WD14Config):
    labels = list(zip(names, logits_row.astype(float)))
    rating = {names[i]: labels[i][1] for i in idx_rating}
    general = [labels[i] for i in idx_general]
    character = [labels[i] for i in idx_character]
    gt = mcut_threshold(np.array([p for _, p in general])) if cfg.use_mcut_general else cfg.thresh_general
    ct = mcut_threshold(np.array([p for _, p in character])) if cfg.use_mcut_character else max(0.15, cfg.thresh_character)
    general_sel = {k:v for k,v in general if v > gt}
    character_sel = {k:v for k,v in character if v > ct}
    return rating, general_sel, character_sel
```

* 既定閾値と **MCut** の選択ロジックはスペース実装の動作に合わせる（一般0.35/キャラ0.85、MCutの式）。([Hugging Face][2])
* 参考としてモデルカードの **P=R \~0.3771**（データセットv2）も閾値設計の目安に残す。([Hugging Face][1])

---

## 6) キャッシュ設計（pHash 起点）

* キー：`(phash_hex, model_repo_id, model_sha)`（**同一画像/同一モデル**の再推論を回避）。
* 値：`{rating:{}, general:{}, character:{}, n_tags, elapsed_ms}`。
* **pHash** 自体は P0 で算出済み（未算出の一時画像は `imagehash.phash()` を適用）。([PyPI][6])

---

## 7) CLI：CSV → WD14 → JSONL

```python
# cli_wd14.py
"""
Usage:
  python -m app.cli_wd14 \
    --in out/p0_scan.csv --out out/p1_wd14.jsonl \
    --provider cpu|openvino --threads 8 \
    --model SmilingWolf/wd-v1-4-swinv2-tagger-v2 \
    [--general 0.35] [--character 0.85] [--mcut-general] [--mcut-character]
"""
```

処理流れ：

1. `labelspace.download_wd14()` で `model.onnx` と `selected_tags.csv` を取得。([Hugging Face][3])
2. `WD14Session` を初期化（`--provider` と `--threads` 反映）。CPU 既定。OpenVINO は任意。([ONNX Runtime][5])
3. `p0_scan.csv` を読み、**未キャッシュのみ**画像をロード→**バッチ推論**→デコード→`p1_wd14.jsonl` に1行/1画像で保存。
4. `p1_wd14_metrics.json` に件数・平均レイテンシ等を記録。
5. 失敗（URL失効等）は `reason=fetch_failed` としスキップ（後日 P0 ルーチンで URL 更新）。

---

## 8) 出力スキーマ（`p1_wd14.jsonl`）

```json
{
  "guild_id":"...", "channel_id":"...", "message_id":"...", "message_link":"...",
  "phash":"...", "is_nsfw_channel":false, "source":"attachment|embed",
  "wd14":{
    "rating":{"general":0.12,"sensitive":0.03,"questionable":0.55,"explicit":0.30},
    "general":{"tag_a":0.82,"tag_b":0.61, "...":"..."},
    "character":{"some name":0.91}
  },
  "meta":{"model":"SmilingWolf/wd-v1-4-swinv2-tagger-v2","size":448,"elapsed_ms":23}
}
```

* **rating 4値＋一般/キャラの確信度**を保持（しきい値超のタグのみ）。
* `size` は ONNX 入力 shape から決定（v2.1 は**可変バッチ**対応）。([Hugging Face][1])

---

## 9) テスト計画（Day 1）

* **単体**：

  * `download_wd14()` が **model.onnx / selected\_tags.csv** を取得できる。([Hugging Face][3])
  * `WD14Session.preprocess()` が **パディング→リサイズ→BGR→NHWC** を満たす。([Hugging Face][2])
  * `infer_batch()` で **\[B, n\_tags]** の配列が返る（B>1）。**可変バッチ**が通ること。([Hugging Face][1])
  * `decode_logits()` が **rating/general/character** を正しく分離（`category` 番号に依存）。([Hugging Face][2])
* **小規模E2E**（10–20枚）：`p0_scan.csv` の一部を対象に、`p1_wd14.jsonl` が生成され、代表タグに明らかな破綻がないこと（目視）。
* **性能**：threads=0/物理コア数での比較、OpenVINO EP 有効時のスループット差を計測。([ONNX Runtime][4])

---

## 10) 失敗系と対処

* **URL失効/アクセス拒否**：`fetch_failed` として記録し、P0の再フェッチ機構で URL 更新を試みる（後続Dayでハンドリング）。
* **ORT不一致**：`onnxruntime < 1.17` だとモデル読み込みに失敗しうる → バージョンを満たす。([Hugging Face][1])
* **OVEPでの挙動差**：OVEP利用時は ORT の高位最適化をオフにしてOV側の最適化に委譲するのが推奨（必要に応じ `graph_optimization_level` を落とす）。([ONNX Runtime][5])

---

## 11) パラメータ初期値（提案）

* `--general 0.35` / `--character 0.85`（HFスペース既定）; **参考**：モデルカード P=R ≈ 0.3771。([Hugging Face][2])
* `--threads` は物理コア数または SMT×0.75 程度から開始（要計測）。**intra/inter** は状況に応じ調整。([ONNX Runtime][4])
* `--provider openvino` は CPU で伸びがあれば採用（Linux wheelは OpenVINO 同梱。Windowsは setupvars 必要）。([ONNX Runtime][5])

---

## 12) 受け入れ基準（Day 1 DoD）

* `p1_wd14.jsonl` が **P0の全対象**に対して生成され、各行に **rating＋タグ** が入っている。
* **同一画像**（同 pHash）は**キャッシュで再推論しない**。
* **可変バッチ**で安定して走る（B>1 でトラブルなし）。([Hugging Face][1])
* `p1_wd14_metrics.json` に枚数/平均ms/失敗件数が出力される。

---

### 備考（次日に向けたインターフェース約束）

* **Day 2** の NudeNet 追加時に、`analysis.jsonl` へ `nudity_detections[]` をマージできるよう、`p1_wd14.jsonl` の上位キーは **将来互換**（`wd14:{...}` をネームスペース化）とする。
* **rules.yaml**（Day 3 で使用）に、`wd14.rating` / `wd14.general` / `wd14.character` の閾値入口を用意しておく。

---

必要なら、この手順書どおりに動く **雛形コード**（`analyzer_wd14.py / cli_wd14.py`）をまとめて渡します。今回は**モデルカード/スペース実装**に忠実な設計なので、しきい値チューニングに集中しやすいはず。

[1]: https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2 "SmilingWolf/wd-v1-4-swinv2-tagger-v2 · Hugging Face"
[2]: https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py "app.py · SmilingWolf/wd-tagger at main"
[3]: https://huggingface.co/docs/huggingface_hub/main/en/guides/download?utm_source=chatgpt.com "Download files from the Hub"
[4]: https://onnxruntime.ai/docs/performance/tune-performance/threading.html?utm_source=chatgpt.com "Thread management"
[5]: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html "Intel - OpenVINO™ | onnxruntime"
[6]: https://pypi.org/project/ImageHash/?utm_source=chatgpt.com "ImageHash"

---

結論：\*\*ほぼそのまま流用できます。\*\*必要な差分は「リポID／キャッシュ鍵／既定しきい値の確認」くらいです。根拠は以下の通りです。

* **v3（EVA02）ONNXはv2系のコードと互換**、かつ**バッチ次元が可変**（=一括推論OK）。実行には **onnxruntime ≥ 1.17.0** が前提。([Hugging Face][1])
* **ラベル仕様は同じ**（`selected_tags.csv` を使い、`category` が *rating=9 / general=0 / character=4*）。Spaceの参照実装も同じ分離ロジック・同じ既定しきい値（**general=0.35 / character=0.85**）。([Hugging Face][2])
* **前処理も同じ**（白地で正方形パディング→入力解像度へリサイズ→RGB→BGR→NHWC）。入出力解像度は**ONNXの入力shapeから取得**する実装が推奨。([Hugging Face][2])

---

## 変更点（差分パッチとして）

1. **モデル識別子の置換**

   * `SmilingWolf/wd-v1-4-swinv2-tagger-v2` → **`SmilingWolf/wd-eva02-large-tagger-v3`**
   * ダウンロード対象は同名の `model.onnx` と `selected_tags.csv`（関数名はそのまま流用可）。([Hugging Face][2])

2. **キャッシュ鍵の拡張**

   * `cache_key = (phash, model_repo_id, model_revision)` に。**v2→v3で学習データ/重みが違う**ため、リポID（できればHFのリビジョン）も鍵に含めて再推論の混入を防止。([Hugging Face][1])

3. **バッチ推論の有効化**

   * v3は**バッチ次元が固定1でない**ので、Day1設計の `infer_batch()` を**B>1前提**にしてスループットを上げる（既にその設計なら変更不要）。([Hugging Face][1])

4. **前処理と閾値は既定のままでOK**

   * 前処理：Space実装と同様で問題なし（パディング→リサイズ→BGR→NHWC）。
   * 既定しきい値：**general=0.35 / character=0.85** を初期値に。必要ならMCut（Maximum Cut）を有効化できる設計のまま。([Hugging Face][2])

5. **メトリクス・モデル情報の記録**

   * 出力JSONに `meta.model = "SmilingWolf/wd-eva02-large-tagger-v3"` と **入力解像度（ONNX shapeから取得）**を必ず残す。Spaceも**shapeから高さを読む実装**です。([Hugging Face][2])

---

## そのまま使い回せる項目

* **ラベル分割**（`category==9/0/4`）、**MCut実装**、**タグ名の`_`→空白置換（顔文字除外）**、**既定しきい値**、**JSONスキーマ**（`wd14:{rating,general,character}` 名前空間は `wd:` や `tagger:` など汎用名にしておくのも手）。([Hugging Face][2])

---

## 追加アドバイス（P1以降を見据えて）

* **v3は学習セット・指標（Macro-F1）を更新**しており、タグの出やすさに差が出ることがあります。P1 Day3のルールチューニングで**赤/橙/黄のしきい値**を再校正してください。([Hugging Face][1])
* 既知の**個別タグの誤検知報告**（例：`tokyo (city)`）もあるため、**NG/参考タグのホワイト/ブラックリスト**を rules 側で持てるように。([Hugging Face][3])

---

### まとめ

* **計画書は基本そのまま使える**／**置換はリポID＋キャッシュ鍵**。
* **処理・前処理・閾値・CSV/JSONスキーマは共通**でOK。
* **onnxruntime ≥ 1.17.0** と **バッチ推論**活用だけは確実に。([Hugging Face][1])

この方針で Day1 実装を EVA02 v3 に差し替えます。必要なら、既存の Day1 スクリプトに対する **最小diff（コード）** もすぐ出します。

[1]: https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/commit/fa2b83fd2d68b3d994e22fed5859d7cc99e1cdb0?utm_source=chatgpt.com "Add model files · SmilingWolf/wd-eva02-large-tagger-v3 at ..."
[2]: https://huggingface.co/spaces/SmilingWolf/wd-tagger/raw/main/app.py "huggingface.co"
[3]: https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/discussions/1?utm_source=chatgpt.com "SmilingWolf/wd-eva02-large-tagger-v3 · `tokyo \\(city\\)` is ..."
