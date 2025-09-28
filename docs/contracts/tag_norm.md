# Tag Normalization Contract

この文書はタグ正規化の単一起点 (Single Source of Truth; SOT) を定義します。以降の実装は本契約に従い、タグ名の保存・評価・比較で同一の正規化ルールを適用してください。

## 正規化ステップ (SOT)

1. 入力値を `str(value)` で文字列化する。
2. Unicode NFKC 正規化を行う。
3. 両端の空白を取り除く。
4. すべて小文字に変換する。
5. 連続する空白を 1 つの空白に縮約する。
6. 空白とハイフンをアンダースコア (`_`) に置き換える。
7. 連続するアンダースコアを 1 つに縮約する。

> 例: `"See Through" -> "see_through"`, `"blood-loss" -> "blood_loss"`。

この結果得られる文字列が標準タグ名です。空文字列になった場合は無効なタグとして扱います。

## スコープ

- 正規化対象は **タグ名** およびそれに準ずる識別子のみです。
- モデル名、ファイル名などの別ドメイン文字列に SOT を適用してはいけません。
- JSONL や SQLite キャッシュの内部表現も本 SOT に合わせて保存します。

## API

正規化 API は `app.engine.tag_norm` モジュールが唯一の実装です。

- `normalize_tag(value: str) -> str`
- `normalize_pair(item: object) -> tuple[str, float] | None`
- `format_tag_for_display(value: str) -> str`

上記 API 以外に新たな正規化ヘルパーを追加しないでください。既存コードはこのモジュールを import して使用します。

## 表示用フォーマッタ

人間に提示する際は `format_tag_for_display` を使用し、`see_through` などのタグを `see through` のように整形できます。表示専用であり、整形結果を保存・比較・キャッシュに書き戻すことは禁止します。

## 衝突と診断

複数の入力が同一の正規化結果に収束した場合を **衝突 (collision)** と呼びます。

- YAML ローダは衝突を検知し、`DslPolicy.logger` の DEBUG/WARN に記録します。
- ルール評価時は任意で `metrics["tag_norm"]["collisions"]` に件数を追加できます。
- 衝突は監視目的で記録するのみで、タグ辞書の更新により解消します。

## 互換ポリシー

- 本 SOT による入出力は互換性の基準です。将来の変更は後方互換になるよう慎重に行ってください。
- 旧来の `_normalize_tag` などのヘルパーは順次撤去し、本モジュールへの委譲で置換します。

