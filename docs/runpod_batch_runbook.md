# Runpod バッチ実行 Runbook（p1/p2/p3）
最終更新: 2025-09-28

この Runbook は、20,000 枚規模の一括処理を Runpod の GPU Pod で安全かつ再現性高く実行するための手順です。
新規の運用担当でもこのドキュメントだけで再現実行できることを目標にしています。

---

## 0. 目的と全体像

- **対象フェーズ**: p1（WD14 タグ付け / GPU）、p2（統合 + NudeNet ゲート）、p3（判定 / DSL 併用）
- **ストレージ方針**: Network Volume（`/workspace`）にモデル・キャッシュ・出力を永続化
  - Network Volume は Pod のライフサイクルと独立に永続し、複数 Pod で共有できます。デフォルトマウントは `/workspace` です。[^runpod-nv]
  - Network Volume は Runpod S3互換 API で直接読み書きできます（Pod `/workspace/path` ⇄ `s3://<VOLUME_ID>/path` が 1:1 対応）。[^runpod-s3]
- **I/O レート制御**: Discord は **グローバル 50 rps** のレート上限があります。必ず QPS を抑制し、429 レスポンス時は `Retry-After` を尊重してください。[^discord-rate]
- **NSFW辞書の一次ソース**: `configs/rules.yaml` の `groups.nsfw_general` を編集し、反映が必要な場合は必ず **p1（必要なら）→p2→p3** の順で再実行してください。`rules.yaml` が見つからないときのみ `configs/xsignals.yaml` へフォールバックします。
- **GPU 推論**: ONNX Runtime の Execution Provider は `['CUDAExecutionProvider', 'CPUExecutionProvider']` を最低ラインに、環境が対応していれば `TensorrtExecutionProvider` を優先してください。[^onnx-ep]

---

## 1. Pod 起動パラメータ

### 1.1 推奨構成

| 項目 | 推奨値 |
| --- | --- |
| GPU | A100 40/80GB, H100, L40S（用途と予算で選択） |
| OS イメージ | 自前のコンテナ or Runpod PyTorch 系（Python 3.10 以上推奨） |
| Network Volume | **必須**。マウント先 `/workspace`（デフォルト設定） |
| Container Disk | 10–20 GB（OS/ツール用。成果物は NV 側に置く） |
| 主要パッケージ | `onnxruntime-gpu`, `awscli`, `git`, `python-dotenv` など |
| 任意ツール | `runpodctl` を利用する場合は Pod 起動後にセットアップ |

> Network Volume の料金目安は $0.07/GB/月。詳細は公式ドキュメントを参照してください。[^runpod-nv]

### 1.2 Pod 起動後の初回セットアップ

```bash
# 依存ライブラリ（GPU 向け）
pip install -U pip
pip install -r requirements-gpu.txt

# ONNX Runtime のプロバイダ確認
python - <<'PY'
import onnxruntime as ort
print("providers:", ort.get_available_providers())
PY
# 期待値: ['CUDAExecutionProvider', 'CPUExecutionProvider']
# (TensorRT 環境なら 'TensorrtExecutionProvider' も含まれる)
```

---

## 2. ストレージ & S3 互換 API

1. **Network Volume をマウント**（Pod 作成時に指定）。 `/workspace` 直下に `models`, `cache`, `out` などを作成して使い回します。[^runpod-nv]
2. **S3 API キーを作成**（Runpod Console → Storage → S3 Access）。
3. `awscli` にエンドポイントとリージョンを設定。DC ごとに endpoint URL が異なるので注意してください。代表例:

| データセンター | エンドポイント | region に指定する値 |
| --- | --- | --- |
| US-KS-2 | `https://s3api-us-ks-2.runpod.io/` | `US-KS-2` |
| EU-DE-1 | `https://s3api-eu-de-1.runpod.io/` | `EU-DE-1` |
| AP-SG-1 | `https://s3api-ap-sg-1.runpod.io/` | `AP-SG-1` |

```bash
# 例: NV の /workspace/out をローカルに同期
export RUNPOD_VOLUME_ID=<YOUR_VOLUME_ID>
export RUNPOD_S3_ENDPOINT=https://s3api-us-ks-2.runpod.io/   # DC に合わせて変更
aws s3 sync s3://$RUNPOD_VOLUME_ID/out ./out \
  --endpoint-url $RUNPOD_S3_ENDPOINT --region US-KS-2
```

---

## 3. .env / 実行パラメータ

`.env` には推奨プロファイルとして以下を置き、**実行時は CLI 引数で上書き**してください。

```dotenv
MODELS_DIR=/workspace/models
CACHE_DIR=/workspace/cache
OUT_DIR=/workspace/out
MODBOT_PROFILE=current
MODBOT_TZ=UTC

WD14_PROVIDER=cuda
WD14_BATCH=48
NUDENET_BATCH=16

# 推奨プロファイル（任意）
ANALYSIS_QPS=4.0
ANALYSIS_CONCURRENCY=24
# MODBOT_DSL_MODE=strict  # 環境変数で strict を指定したい場合のみ
```

実行時は、環境変数を渡すか CLI 引数で明示します。

```bash
python -m app.cli_wd14 ... --qps ${ANALYSIS_QPS:-4.0} --concurrency ${ANALYSIS_CONCURRENCY:-24}
python -m app.cli_scan ...

strict で実行したい場合は `MODBOT_DSL_MODE=strict` を環境変数として渡すか、CLI に `--dsl-mode strict` を明示してください。
```

---

## 4. シャーディング実行フロー

> `.tmp → rename` 方式で中断耐性を持たせています。429 が出た場合は必ず QPS → 並列の順に下げてください。[^discord-rate]

### 4.1 分割（p0 → shard 生成）

```bash
python scripts/split_index.py \
  --profile current \
  --date 2025-10-01 \
  --shards 10
```

### 4.2 p1（WD14 推論 / GPU）

```bash
python scripts/run_p1_sharded.py \
  --profile current \
  --date 2025-10-01 \
  --shard-glob "out/profiles/current/p0/shards/shard_*.csv" \
  --provider cuda \
  --batch-size ${WD14_BATCH:-48} \
  --concurrency ${ANALYSIS_CONCURRENCY:-24} \
  --qps ${ANALYSIS_QPS:-4.0} \
  --parallel 2 \
  --resume \
  --status-file out/profiles/current/status/p1_manifest_current.json
```

### 4.3 p2（統合 / NudeNet ゲート）

```bash
python scripts/run_p2_sharded.py \
  --profile current \
  --date 2025-10-01 \
  --shard-glob "out/profiles/current/p0/shards/shard_*.csv" \
  --qps ${ANALYSIS_QPS:-4.0} \
  --concurrency 16 \
  --parallel 2 \
  --resume \
  --status-file out/profiles/current/status/p2_manifest_current.json \
  --extra-args "--nudenet-mode auto"
```

- ランナーは `--rules-config configs/rules.yaml` を自動付与します。別パスを使う場合は `--extra-args` に明示して上書きしてください。

### 4.4 p3（判定 / DSL 併用）

```bash
# 本番
python -m app.cli_scan \
  --profile current \
  --date 2025-10-01 \
  --rules configs/rules.yaml \
  --metrics out/profiles/current/metrics/p3/p3_metrics_2025-10-01.json \
  --since 1970-01-01 --until 2099-12-31

レガシー構成（`version: 1`）を一時的に続行する場合は `--allow-legacy --fallback green` を付与してください。結果は強制的に `severity=green` となり、CSV 契約は保たれます（書き込みを抑止したい場合は `--fallback skip`）。

# 追加トレースを出力（必要なときだけ）
python -m app.cli_scan \
  --profile current \
  --date 2025-10-01 \
  --rules configs/rules.yaml \
  --metrics out/profiles/current/metrics/p3/p3_metrics_2025-10-01.json \
  --trace-jsonl out/profiles/current/metrics/p3/p3_trace_2025-10-01.jsonl

# ドライランで件数・速度確認
python -m app.cli_scan \
  --profile current \
  --date 2025-10-01 \
  --rules configs/rules.yaml \
  --metrics out/profiles/current/metrics/p3/p3_dry_2025-10-01.json \
  --dry-run --limit 256 \
  --since 1970-01-01 --until 2099-12-31

# A/B 比較（採用前の検証）
python -m app.cli_rules_ab \
  --profile current \
  --date 2025-10-01 \
  --rulesA configs/rules.yaml \
  --rulesB configs/rules_candidate.yaml \
  --out-dir out/profiles/current/metrics/ab \
  --sample-diff 200 \
  --samples-minimal \
  --samples-redact-urls

- `--out-dir` を指定すると `p3_ab_compare.json`・`p3_ab_diff.csv`・`p3_ab_diff_samples.jsonl` が同一ディレクトリにまとまり、既存の `--out-json/--out-csv` 指定は不要です。
- モード解決は **`--lock-mode` > `--dsl-mode` > `MODBOT_DSL_MODE` > YAML `dsl_mode` > warn**。Runpod 上で両側のモードを固定したい場合は `--lock-mode strict` 等を追加してください。
- legacy ルールを検出すると exit=2 で停止します。差分だけ確認したいときは `--allow-legacy` を付けると compare.json に `note="skipped due to legacy ruleset"` が入り、CSV/サンプルは生成されません。
- サンプル JSONL を軽量化するには `--samples-minimal` を、URL を秘匿したい場合は `--samples-redact-urls` を併用してください（URL フィールドは null、テキスト内の URL は `[URL]` に置換）。
```

---

## 5. メトリクスと成果物

### 5.1 JSONL / メトリクス

| フェーズ | 主成果物 | メトリクス |
| --- | --- | --- |
| p1 | `out/profiles/<profile>/p1/p1_YYYY-MM-DD.jsonl` | `out/profiles/<profile>/metrics/p1_metrics_YYYY-MM-DD.json` |
| p2 | `out/profiles/<profile>/p2/p2_YYYY-MM-DD.jsonl` | `out/profiles/<profile>/metrics/p2_metrics_YYYY-MM-DD.json` |
| p3 | `out/profiles/<profile>/p3/findings_YYYY-MM-DD.jsonl` | `out/profiles/<profile>/metrics/p3_run.json`, `out/profiles/<profile>/metrics/p3_ab_compare.json` |

#### p1/p2 メトリクスの集約

```bash
# シャード単位の JSON を合算
python - <<'PY'
import json, glob, collections

def merge(pattern, out_path):
    acc = collections.Counter()
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as fp:
            try:
                data = json.load(fp)
            except Exception:
                continue
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    acc[key] += value
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(acc, out, ensure_ascii=False, indent=2)

merge("out/profiles/current/metrics/p1_metrics_*.json", "out/profiles/current/metrics/p1_metrics.json")
merge("out/profiles/current/metrics/p2_metrics_*.json", "out/profiles/current/metrics/p2_metrics.json")
PY
```

#### p3 メトリクスの補足

`p3_run.json` には `winning` の生カウントに加えて `winning_ratio` と `throughput_rps` が出力されます。`throughput_rps` は **フィルタ通過後に findings として書き出した件数 ÷ 実測ウォールタイム** で計算されるため、重いフィルタや `--limit` 指定時は実体感より小さく見える点に注意してください。走行時間の粗い把握や運用メトリクスのダッシュボード化では、この値を補助指標として扱うことを推奨します。

また、findings の各レコードには `metrics.eval_ms`（評価処理1件あたりのレイテンシ、ミリ秒・小数第3位丸め）が追加されました。レビュー時は以下のように `jq` などで抜き出して比較できます。

```bash
jq -r '.metrics.eval_ms // empty' out/profiles/current/p3/findings_2025-10-01.jsonl | head
```

### 5.2 レビュー用成果物

| 用途 | 出力例 |
| --- | --- |
| A/B 差分確認 | `out/profiles/<profile>/exports/p3_ab_diff.csv` |
| 差分サンプル（JSONL） | `out/profiles/<profile>/exports/p3_ab_diff_samples.jsonl` |
| 不確実域サンプル（Top200） | `out/profiles/<profile>/review/uncertain_top200.csv` |

**不確実域サンプルの抽出**（最小ツール版）:

```bash
python tools/export_uncertain.py \
  --profile current --date 2025-10-01 \
  --out out/profiles/current/review/uncertain_top200.csv \
  --q-thr 0.35 --e-thr 0.20 --eps 0.02 --topn 200
```

---

## 6. レート制御とリトライ指針

1. **QPS は 4.0 から開始**し、429 が出たら即 3.0 → 2.5 … と下げる。
2. **並列（concurrency）も 16–24 程度まで**。429 が止まらない場合は 8–12 に落とす。
3. 429 のレスポンスには `Retry-After` が含まれるので必ず待機。
4. 5xx は指数バックオフで再試行。403/404 は 1 回だけメッセージ再取得でリトライし、失敗したら `expired` として記録する。

---

## 7. 障害時の再開

- **p1/p2**: shard ごとに `.tmp` → 完了時に rename。`--resume` を付けて再実行すれば未完 shard のみ再開します。
- **p3**: `/scan` 実行前に `--dry-run` で件数・速度を確認し、採用前に `cli_rules_ab` で差分をレビューします。差分サンプルは `out/profiles/<profile>/exports/p3_ab_diff_samples.jsonl` に出力されます。

---

## 8. 受入条件（Definition of Done）

- [ ] Runpod 上で p1/p2/p3 が本 Runbook の手順だけで再現できる
- [ ] `out/profiles/<profile>/metrics/p1_metrics_YYYY-MM-DD.json`, `out/profiles/<profile>/metrics/p2_metrics_YYYY-MM-DD.json`, `out/profiles/<profile>/metrics/p3_run.json`, `out/profiles/<profile>/metrics/p3_ab_compare.json` が生成される
- [ ] `out/profiles/<profile>/exports/p3_ab_diff.csv` と `out/profiles/<profile>/review/uncertain_top200.csv` がレビューに利用できる
- [ ] Discord API の rate limit (50 rps) を超過せず 429 が発生した際は調整が記録されている
- [ ] Network Volume / S3 互換 API を使った入出力同期手順が確認済み

---

## 付録 A. GPU 別 推奨初期値

| GPU | p1 `--batch-size` | I/O `--concurrency` | `--qps` 開始値 |
| --- | ---: | ---: | ---: |
| A100 40GB | 48 | 24 | 4.0 |
| A100 80GB | 64–96 | 24 | 4.0 |
| H100 80GB | 64–96 | 24 | 4.0 |
| L40S 48GB | 32–64 | 16–24 | 3.5 |

> 数値は目安です。DC の負荷や CDN 応答を観察しつつ調整してください。429 が発生した場合は必ず QPS → 並列の順に下げます。[^discord-rate]

---

## 付録 B. 参考リンク

- Runpod Pods と Network Volume 概要: <https://docs.runpod.io/pods/> [^runpod-nv]
- Network Volume / S3 互換 API: <https://docs.runpod.io/serverless/storage/s3-api> [^runpod-s3]
- Discord API Rate Limit: <https://github.com/discord/discord-api-docs/blob/main/docs/topics/Rate_Limits.md> [^discord-rate]
- ONNX Runtime Execution Providers: <https://onnxruntime.ai/docs/execution-providers/> [^onnx-ep]

---

[^runpod-nv]: Runpod Documentation – Pods / Storage / Network Volumes
[^runpod-s3]: Runpod Documentation – Serverless Storage (S3-compatible API)
[^discord-rate]: Discord API Docs – Rate Limits (グローバル 50 rps, 429 の `Retry-After`)
[^onnx-ep]: ONNX Runtime Documentation – Execution Providers

## 付録 C. p3 契約チェック

- 出力契約の詳細: `docs/contracts/p3.md`
- チェック手順:
  - `python -m app.cli_contract check-findings --profile current --date 2025-10-01`
  - `python -m app.cli_contract check-report   --profile current --date 2025-10-01`
- CI やローカル検証で上記コマンドの終了コードが 0 であることを確認し、破壊的変更を早期検知します。
