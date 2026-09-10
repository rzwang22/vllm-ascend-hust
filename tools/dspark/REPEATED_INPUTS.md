# Historical 400-request workload

The server-reported input is
`/workspace/dspark-results/p08-r9c-400request-sweep._93zya1c/input-400.jsonl`.
Its first 400 records contain 64 distinct token sequences, maximum length 116.
These server counts have not been independently recomputed on Mac:
`SERVER_NOT_REVALIDATED`. Local tests use a synthetic 400/64 fixture.

## Import and schema

`--allow-repeated-prompts` explicitly permits **existing source occurrences** in
frozen-token imports. Default imports still reject repeated prompts. No extra
requests are generated: no looping, deduplication, sorting, re-rendering,
re-tokenization or prompt numbering. Display text alone is decoded from tokens.

Opt-in manifest schema 2 records `request_instance_count`, `unique_prompt_count`,
`repeated_prompt_policy=preserve_source_occurrences`, an ordered token-sequence
hash and the complete token-hash-to-instance-ID mapping. It omits
`num_unique_samples`. Each record has a unique deterministic
`request:<original-file-sha256>:<source-index>` instance ID (also its `case_id`),
`original_case_id`, the untouched `raw_task`, token hashes, occurrence index and
`replay_of` pointing to the first instance of the identical token sequence.
`original_case_id` uses the selected `--id-field` (default `task_id`), or null
when absent; all other original case fields remain in `raw_task`. The full
original file is copied byte-for-byte, even if only its prefix is used.

`read_manifest` validates file, source, record, token, sequence and mapping
hashes, exact prefix order, unique instance IDs and repetition references.
Schema 1 unique inputs remain supported. Subsets count request instances and
recompute their unique-prompt counts. Profile/performance drivers copy all
manifest assets, including the original file, for independent revalidation.

Async runtime IDs remain the existing unique `batchN-index` values; the manifest
IDs are mapped by input index in each run's `requests.json`. Repeated original
case IDs cannot collide in runtime admission. Neither prompt IDs nor runtime
request IDs are added to the model input.

Prefix caching remains **False**, checked against the effective engine config
and now recorded explicitly in plans, input identity and reports. Both modes use
the same input instances, order, warmup, sampling and client admission settings.
`summary.json/csv/md` report **400 request instances / 64 unique prompts**.
Independent statistics still count engine lifecycles: three runs per mode means
n=3, not n=400. Prompt repetitions do not expand quality coverage. Old reports
without population metadata show unavailable, rather than invented counts.

## Server commands

Use Bash, the existing CANN/custom OPP environment, and the delivered full SHA.
No package installation, OPP build or device reset is needed. Do not reuse older
cost tables: both modes and costs must use this same new SHA. Historical
throughput is reference only.

```bash
cd /workspace/vllm-ascend-hust
SHA='<delivered-full-commit>'
git fetch origin feat/dspark && git merge --ff-only "$SHA"
FROZEN_INPUT=/workspace/dspark-results/p08-r9c-400request-sweep._93zya1c/input-400.jsonl
FROZEN_SHA=$(sha256sum "$FROZEN_INPUT" | awk '{print $1}')
MODEL=/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8
DATA_ROOT=$(mktemp -d /workspace/dspark-results/dspark-repeated-input.XXXXXXXX)
set -o pipefail
python tools/dspark/prepare_performance_data.py \
  --input-jsonl "$FROZEN_INPUT" --expected-source-sha256 "$FROZEN_SHA" \
  --source-name historical-400 --source-revision "$FROZEN_SHA" \
  --kind general --frozen-token-field prompt_token_ids --allow-repeated-prompts \
  --num-samples 400 --max-input-tokens 116 \
  --tokenizer "$MODEL" \
  --tokenizer-revision 9e8679a9db7eec11efed9925f7efb96549077545 \
  --output-dir "$DATA_ROOT/input" 2>&1 | tee "$DATA_ROOT/import.log"
IMPORT_CODES=("${PIPESTATUS[@]}")
printf '%s\n' "${IMPORT_CODES[*]}" > "$DATA_ROOT/import.pipestatus"
MANIFEST="$DATA_ROOT/input/manifest.json"
test "${IMPORT_CODES[0]}" -eq 0 && test "${IMPORT_CODES[1]}" -eq 0
```

Only after import succeeds, independently compare the original prefix and verify
400/64, multiplicities, unique instance IDs and maximum input length:

```bash
python - "$MANIFEST" "$FROZEN_INPUT" <<'PY' | tee "$DATA_ROOT/check.json"
import json
import sys
from collections import Counter
from pathlib import Path
from tools.dspark.prepare_performance_data import read_manifest, _read_jsonl, _sha256_file, input_population
m, rows, _ = read_manifest(sys.argv[1], 400)
original = _read_jsonl(Path(sys.argv[2]))[:400]
expected = [r['prompt_token_ids'] for r in original]
actual = [r['prompt_token_ids'] for r in rows]
assert actual == expected
assert Counter(map(tuple, actual)) == Counter(map(tuple, expected))
assert m['request_instance_count'] == len(actual) == 400
assert m['unique_prompt_count'] == len(set(map(tuple, actual))) == 64
assert len({r['request_instance_id'] for r in rows}) == 400
assert max(map(len, actual)) == 116
assert m['original_source_file_sha256'] == _sha256_file(Path(sys.argv[2]))
print(json.dumps({'status': 'valid', **input_population(m, rows),
                  'original_source_file_sha256': m['original_source_file_sha256']}, indent=2))
PY
CHECK_CODES=("${PIPESTATUS[@]}")
printf '%s\n' "${CHECK_CODES[*]}" > "$DATA_ROOT/check.pipestatus"
test "${CHECK_CODES[0]}" -eq 0 && test "${CHECK_CODES[1]}" -eq 0
```

Only after the data check passes, generate new costs. The existing wrapper
creates a new directory, runs focused tests (including repeated-input tests),
preserves PIPESTATUS and evidence, and stops on errors or occupied NPUs.

```bash
bash tools/dspark/run_dspark_large_batch.sh "$SHA" "$MANIFEST" \
  --stage profile --batches 64 128 256 --num-prompts 400 --model "$MODEL"
```

On success, set `COST_DIR` to that invocation's `SERVER_RESULT_DIR/runs`, then
run one validation per mode and batch:

```bash
COST_DIR='/workspace/dspark-results/dspark-large-batch.<profile-id>/runs'
bash tools/dspark/run_dspark_large_batch.sh "$SHA" "$MANIFEST" \
  --stage validate --cost-dir "$COST_DIR" --batches 64 128 256 \
  --num-prompts 400 --output-len 256 --warmup-prompts 4 --repeats 1 --model "$MODEL"
```

Only after validation passes, collect three independent repeats:

```bash
bash tools/dspark/run_dspark_large_batch.sh "$SHA" "$MANIFEST" \
  --stage repeat --cost-dir "$COST_DIR" --batches 64 128 256 \
  --num-prompts 400 --output-len 256 --warmup-prompts 4 --repeats 3 --model "$MODEL"
```

Default modes remain fixed-K `dspark_graph` and `dspark_confidence_graph`;
client outstanding is uncapped (all requests submitted), natural EOS remains
active and prefix caching remains disabled. Formal measured FULL replay gates
and output/NaN/error checks remain intact. Text/EOS differences do not create a
cross-mode exact-token gate. Same-SHA comparisons are produced per B in
`summary.json`, `summary.csv` and `summary.md`; failure artifacts are retained.

Local validation: 210 tests passed, 3 installed-vLLM/Ascend tests skipped.
Scoped manual pre-commit hooks passed. Full `format.sh ci` was run in a disposable
worktree and encountered existing repository failures (78 unrelated files
modified there, zero delivery files); those changes were not included.
No server input file, model or NPU execution was accessed locally.
