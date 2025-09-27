# P1 モデレーション判定リワーク 詳細実装計画書

## 目的と背景
- 既存の単純しきい値判定を、チャンネル属性・主語・文脈・露出を組み合わせた優先順位ルールへ刷新する。
- UI や既存 CSV を崩さずに、以下の最終方針を達成することがゴール。
  - 非 NSFW チャンネル × 性的描写 = RED
  - 未成年 × 軽微露出 = ORANGE（露骨=RED）
  - 未成年のみ（文脈なし）= GREEN
  - 成人 × 性的 × 薬物 = ORANGE（非 NSFW は規約 RED）
  - 欠損 × 出血 = RED
  - Violence Threat は今回のルールから除外
  - YELLOW はロジックから出さない（UI 上は残す）

## 変更スコープ
- `app/rule_engine.py`
  - モジュール定数として静的サブ集合・閾値デフォルト・NudeNet パターン定義を追加。
  - `RuleEngine.__init__` で `configs/nudenet.yaml` の任意設定を読み込み（存在時のみ）。
  - `evaluate` 内で新規指標を算出し、優先度付き判定ロジックを実装。
  - `metrics` に新指標を追加しつつ既存キーを維持。
  - WD14 欠落時の `wd14_missing` 理由を付与。
- `configs/rules.yaml`
  - `rule_titles` に新規ルール ID とタイトルを追記。
  - 必要に応じて `thresholds` に新キー（sexual_explicit_sum_med/high など）を追加し、未定義時はコード側デフォルトを利用。
- `docs/`（本ファイルのみ）
  - 作業方針とテスト観点の記録。

## 新規ルール ID / タイトル
`configs/rules.yaml` に以下を追記する。

| ルール ID | タイトル | 適用条件（概要） |
|-----------|----------|------------------|
| `RED-NSFW-101` | 非NSFWチャンネルの性的表現 | 非 NSFW チャンネルで Sexual MED/HIGH or Modifiers |
| `RED-MINOR-SEX-201` | 未成年×性的 | 未成年シグナル + Sexual HIGH/Modifiers |
| `RED-MINOR-GORE-202` | 未成年×流血/損壊 | 未成年シグナル + Gore ANY |
| `RED-ANIMAL-SEX-301` | 動物×性的/獣姦 | 動物主語 + bestiality/zoophilia or Sexual HIGH + Modifiers |
| `RED-ANIMAL-GORE-302` | 動物×虐待/流血 | 動物主語 + 虐待文脈 or Gore ANY |
| `RED-DISMEMBER-BLOOD-401` | 欠損×出血 | Dismember HIT + Gore ANY |
| `ORANGE-ADULT-SEX-DRUG-501` | 成人×性的×薬物 | Sexual MED/HIGH/Modifiers + Drug シグナル（未成年閾値未満、NSFW 限定） |
| `ORANGE-MINOR-MILD-601` | 未成年×軽微露出 | 未成年シグナル + Mild Exposure（Sexual HIGH を除く） |

既存の `ORANGE-101` 等は後方互換のため維持するが、上記ルールが先に発火するよう評価順を調整する。

## 指標と閾値の算出
### 基本指標
- `minor_peak` / `animal_peak`: `minor_tags` および `animal_abuse_tags` から最大値。動物タグは虐待文脈語を除外して主語スコアとする。
- `sexual_explicit_sum` / `sexual_modifier_sum`: `nsfw_general_tags` を静的サブ集合に分岐して合計。
- `exposure_peak`: 既存 `xsignals.exposure_score` と NudeNet `EXPOSED_*` の最大値。
- `mild_exposure_peak`: NudeNet クラス名が `EXPOSED` を含みつつ `ARMPIT` / `BELLY` / `MIDRIFF` / `STOMACH` / `ABDOMEN` を部分一致したピーク。該当なしの場合は `questionable >= 0.35` で代替。
- `gore_peak` / `gore_sum`: 静的 `GORE` 集合から最大・合計。
- `dismember_peak`: 静的 `DISMEMBER` 集合の最大。
- `drug_any`: WD14 一般タグの小文字化を前提に、`DRUG_KEYWORDS` 部分一致ヒットの最大スコア。閾値 (`drug_any_min`) 以上で真とする。

### 閾値デフォルト
- `sexual_explicit_sum_med = 0.10`
- `sexual_explicit_sum_high = 0.15`
- `sexual_modifier_sum_min = 0.10`
- `animal_subject_min = 0.35`
- `minor_peak_min = 0.20`
- `gore_peak_min = 0.30`
- `gore_sum_min = 0.40`
- `dismember_peak_min = 0.20`
- `mild_exposure_peak_min = 0.30`
- `drug_any_min = 0.15`

設定ファイルに値が無い場合は上記デフォルトを使用する。

## 判定フロー
1. **規約 RED**: `not is_nsfw_channel` かつ `sexual_med/high or sexual_with_modifiers` → `RED-NSFW-101`
2. **未成年 × 性的**: `minor_peak >= minor_peak_min` かつ `(sexual_high or sexual_with_modifiers)` → `RED-MINOR-SEX-201`
3. **未成年 × ゴア**: `minor_peak >= minor_peak_min` かつ `gore_any` → `RED-MINOR-GORE-202`
4. **動物 × 性的**: `animal_peak >= animal_subject_min` かつ `(bestiality/zoophilia hit または sexual_high && sexual_modifier_sum >= sexual_modifier_sum_min)` → `RED-ANIMAL-SEX-301`
5. **動物 × 虐待/流血**: `animal_peak >= animal_subject_min` かつ `(animal_abuse_context hit または gore_any)` → `RED-ANIMAL-GORE-302`
6. **欠損 × 出血**: `dismember_peak >= dismember_peak_min` かつ `gore_any` → `RED-DISMEMBER-BLOOD-401`
7. **成人 × 性的 × 薬物**: `minor_peak < minor_peak_min` かつ `(sexual_med/high or sexual_with_modifiers)` かつ `drug_any` → `ORANGE-ADULT-SEX-DRUG-501`
8. **未成年 × 軽微露出**: `minor_peak >= minor_peak_min` かつ `mild_exposure` かつ `not sexual_high` → `ORANGE-MINOR-MILD-601`
9. 上記に該当しない場合は `GREEN`。YELLOW 分岐は外し、旧ロジックの閾値はメトリクス抑止用としてのみ残す。

`reasons` にはトリガーとなった主要指標とチャネル種別を記録する（例: `minor_peak=0.28`, `exposure_peak=0.64`, `mods=leash,collar`, `channel=non-nsfw`）。

## NudeNet パターン設定フック
- `configs/nudenet.yaml` に任意キー `mild_exposure_label_patterns` / `mild_exposure_threshold` を追加できるようにする。
- 存在しない場合はコード内デフォルト (`{"ARMPIT", "BELLY", "MIDRIFF", "STOMACH", "ABDOMEN"}`, 閾値 `0.30`) を使用。

## 非機能要件
- 既存の `/scan` `/report` コマンド、Discord 埋め込み、CSV 出力への変更は不要。
- `metrics` ブロックに新指標を追加しつつ、既存項目 (`nsfw_margin`, `nsfw_ratio`, `nsfw_general_sum`, `exposure_peak` など) を保持し互換性を維持。
- `violence_threat` 系タグは今回参照しないことを明確化するコメントを `rule_engine.py` に残す。
- 将来の環境フラグ (`ENFORCE_NONNSFW_SEXUAL_RED`, `RULESET_MINOR_MILD_ORANGE`) はコメントで設計メモのみ残す。

## テスト計画
1. **非NSFW × 成人性的** → `RED-NSFW-101`
2. **NSFW × 成人性的 × 薬物** → `ORANGE-ADULT-SEX-DRUG-501`
3. **未成年 × 軽微露出** → `ORANGE-MINOR-MILD-601`
4. **未成年のみ（露出/文脈なし）** → `GREEN`
5. **欠損のみ** → `GREEN` / **欠損 × 出血** → `RED-DISMEMBER-BLOOD-401`
6. **dog + leash/collar（露出/explicitなし）** → `GREEN`
7. **child + leash/collar + NudeNet≥0.30 or WD14 Q/E** → `RED-MINOR-SEX-201`
8. **暴力脅威（stab/gunshot）単体** → `GREEN`（今回不使用であることを確認）
9. 代表 100 件で Before/After の色分布を比較し、所期の変化と副作用がないか確認。

## ロールアウト・検証
- 手動で `/scan` → `/report` を実行し、
  - severity カウント（yellow が 0〜極小であること）
  - CSV 出力の列整合
  - 理由文の可読性（主要指標が記録されていること）
  を確認。
- 誤判定レビューのため、`wd14_missing` や NudeNet パターンが空の場合の理由ログを重点確認。

## リスクと緩和策
- **WD14 タグ揺れ**: 文字列正規化と静的集合を厳格化し、部分一致は薬物検出など必要最低限に限定。
- **NudeNet ラベル差分**: 部分一致 + 設定フックで吸収。新しいラベルが登場した場合は `nudenet.yaml` で追加。
- **既存 UI の yellow 想定**: severity 列挙はそのまま維持しつつ、生成しないことで後方互換を確保。
- **動物タグの種/文脈混在**: コード内で subject/context を分離し、重複タグは最優先集合でのみ使用する方針をコメント化。

## 作業手順
1. ブランチ切り出し（例: `feat/rule-engine-rework`）。
2. `configs/rules.yaml` にルール ID と閾値キー（必要なら）を追加。
3. `app/rule_engine.py` に静的集合・閾値・評価ロジックを実装。
4. NudeNet パターン設定フックを組み込み（任意キー読み込み）。
5. ローカルで `/scan` → `/report` を実行し、期待挙動とログを確認。
6. テスト項目に沿って代表ケースで実測。
7. 影響範囲をまとめ、コミット前に差分とログを添えてレビューに備える。

---
本計画に基づき実装を進め、完了後に検証結果を共有する。
