# Rules v2 ローダ仕様サマリ

## トップレベル構造

- 必須キー: `version` / `rule_titles` / `groups` / `features` / `rules`
- 任意キー: `dsl_mode`
- その他のキーは無視されるが、`R2-K001` の WARNING として記録される。
  - strict モードでは即座に `status=error` で終了。

## モードの優先順位

1. CLI `--mode` / `--dsl-mode`
2. ENV `MODBOT_DSL_MODE`
3. YAML の `dsl_mode`
4. 既定値 `warn`

`LoadResult.mode` は優先順位適用後の値。
環境変数による上書きは呼び出し側で `override_mode` として渡される前提です。

## バリデーション結果（LoadResult）

| status   | 意味                                       | CLI 終了コード |
|----------|--------------------------------------------|----------------|
| `ok`     | 問題なし                                   | 0              |
| `invalid`| 読み込みは可能だが削除/警告を含む           | 2 *(※)*        |
| `error`  | 致命的で読み込み不可（config は `None`）   | 2              |

※ `--allow-disabled` 指定時、`disabled_rules`/`disabled_features` のみが原因なら `exit=0` を許容。

### LoadResult フィールド

- `mode`: 最終的に採用されたモード
- `config`: 正常系では `LoadedConfig`
- `issues`: `ValidationIssue` の配列（code/where/msg/hint）
- `counts`: 集計値
  - `rules` / `features` / `groups`
  - `errors` / `warnings`
  - `disabled_rules` / `disabled_features`
  - `placeholder_fixes`
  - `unknown_keys`
  - `collisions`
- `summary`: `rules_ok` / `rules_disabled` / `features_disabled`

## 主要な検査内容

| コード      | 内容                                                |
|-------------|-----------------------------------------------------|
| `R2-V001`    | `version` 不一致、`rules` 非リスト、必須項目欠落 等 |
| `R2-K001`    | 未知トップレベルキー検知                           |
| `R2-T000`    | `rule_titles` 欠落／型不正                          |
| `R2-T001`    | `rule_titles` に不足 ID                             |
| `R2-D001`    | `rules[].id` 重複                                   |
| `R2-E001`    | DSL 式の構文エラー／未知関数                        |
| `R2-E002`    | DSL 式の未知識別子                                 |
| `R2-G001`    | 未知グループ・タグ正規化時の問題                    |
| `R2-P001`    | `reasons[]` のプレースホルダ解析失敗                 |

### グループ正規化

- Unicode NFKC → trim → lower → 空白/ハイフンを `_` へ統一
- 重複合流を `counts.collisions` へ計上し、DEBUG ログでサンプルを出力

### 式の静的解析

- 利用可能識別子: `rating.*`, `channel.is_nsfw`, `message.is_spoiler`, `attachment_count` など `dsl_runtime.list_builtin_identifiers()` が返す集合 + 成功した `features` 名
- 許可関数: `score`, `sum`, `max`, `min`, `any`, `count`, `clamp`, `topk_sum`
- グループ関数の第 1 引数がリテラル文字列のとき、`groups` に存在する必要あり
- `metrics.*` 参照は許可（動的値として扱う）

追加メトリクス:

- `exposure_area` / `exposure_count` — `analysis_merge` が算出する `nudity_area_ratio` / `nudity_box_count` のエイリアス。DSL から直接参照できます。

### `rules[].reasons`

- `{EXPR[:FMT]}` を DSL と同じパーサで検証
- 失敗時は該当要素を削除し、`counts.placeholder_fixes` を加算
- strict の場合は `status=error`

## CLI `python -m app.cli_rules validate`

オプション:

- `--rules <path>`: デフォルト `configs/rules.yaml`
- `--mode warn|strict`: CLI でモードを上書き
- `--print-config`: 集計サマリを表示
- `--json`: JSON 出力（`status/mode/counts/issues/summary`）
- `--treat-warnings-as-errors`: WARNING が 1 件でもあれば exit=2
- `--allow-disabled`: 無効化のみで `invalid` の場合に exit=0

## 既定の DSL 雛形

```yaml
version: 2
rule_titles:
  EXAMPLE: "Example rule"

groups:
  nsfw_general: ["bikini", "see_through"]

features:
  nsfw_margin: "max(rating.explicit, rating.questionable) - max(rating.general, rating.sensitive)"

rules:
  - id: EXAMPLE
    severity: orange
    when: "(!channel.is_nsfw) && (nsfw_margin >= 0.10)"
    reasons: ["margin={nsfw_margin:.2f}"]
```

## エラーコードの補足

- `R2-IO`: ファイル読み込み失敗
- `R2-YAML`: YAML パース失敗

これらは CLI でも `status=error` として扱われる。
