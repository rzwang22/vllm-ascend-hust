# Uneven startup profile NaN: evidence audit and isolated diagnostic

## Verified archive

Read the local `dspark-large-batch.ikliNmVE-evidence.tar.gz`. SHA256:
`0d2cb640193ed4bb77643dd8e80e512e54ce90e55a942c0dd643c587065b486a`.
All 42 archive members were checked before extraction: no absolute/traversing
paths, links or special files. Expanded size: 22,908,182 bytes.
Both complete logs (`runs/b64.log`, `generation.log`), source/focused/status,
plan, command, lifecycle, retained points and streaming records were read.
Plan source SHAs match Plugin `b2a3549ec208530feca4e05edaf638a272e77641` and
Core `897306c43bf800e2480cb5c0f3e2da408d85a2fd`. All nine retained raw-file hashes
were independently verified. The archived manifest passed `read_manifest`:
400 request instances / 64 unique prompts, ordered token-sequence hash
`8454cf91edcabd1e49669e06792a63e572b19f5fc0d85cb115aa3535707bdcfb`.

Actual capture sizes were `[6,12,24,48,96,192,384]`; one B64 engine initialized
and shut down. Nine points were retained, in this order:

1. ctx128-n1-t6-balanced
2. ctx128-n1-t6-skewed
3. ctx128-n2-t6-balanced
4. ctx128-n2-t6-skewed
5. ctx128-n2-t12-balanced
6. ctx128-n2-t12-skewed
7. ctx128-n4-t6-balanced
8. ctx128-n4-t6-skewed
9. ctx128-n4-t12-balanced

The tenth point `ctx128-n4-t12-skewed` failed. It is specified-length profiling,
not a learned confidence decision, and initially has four active requests.

## Observed completion and layout changes

For balanced n4-t12, each of eight ranks has the same 347 event records:
three mixed/prefill draft events; 169 target plus 169 draft events with four
requests and query `[3,3,3,3]`; two target/draft pairs with three requests and
query `[3,3,3]`, capacity 12; finally one target/draft pair with one request,
query `[3]`, capacity 6. All four outputs completed at 512 tokens, in order
`batch9-0`, `batch9-1`, `batch9-2`, `batch9-3` (the middle two nearly simultaneous).
This includes exit/compaction transitions, not just the initial full batch.

For skewed n4-t12:

| External request | Internal suffix | Output tokens received | Completion |
| --- | --- | --- | --- |
| batch10-0 | 9a138f18 | 512 | length, monotonic 4306730.503596243 |
| batch10-1 | 988c6b21 | 397 | EngineDeadError |
| batch10-2 | 8fc7e2a2 | 128 | EngineDeadError |
| batch10-3 | aaa178d6 | 95 | EngineDeadError |

The three survivors each have 11 stream events observed after request 0's
completion. Their delivered-token sequences are respectively
`[4,4,6,6,6,6,6,6,6,6,6]`, `[1,1,4,4,4,4,4,4,4,4,4]`, and eleven ones.
These are frontend delivery events, not worker execution epochs.

`ConfidenceVerification.select()` assigns the specified pattern by the sorted
**current candidate owners**, so after an owner leaves, survivors can be assigned
new lengths. For three pure-decode survivors the code would select `[5,3,0]`
in owner order, and runner query sorting would produce `[1,4,6]`, total 11 at
capacity 12. This is a conditional source inference consistent with delivery
changes, **not the recovered NaN-producer layout**. Mixed admission, outstanding
async work and acceptance timing prevent equating stream events with that input.

All ranks 0–7 log base-logit NaN before owner failure at 05:50:07. Each traceback
is printed twice; no producer epoch is present, so duplicate prints are not
counted as separate failed executions. The later scheduler dump has three
survivors, 18 scheduled tokens and five placeholder candidates per request;
this is pre-selection/queued scheduler state after the error. It cannot identify
the failed target/proposal query spans. Reported computed counts are
`[524,257,227]`, output counts `[397,130,100]`; these also differ from delivered
frontend output and are not substituted for producer metadata.

## Proven boundary and remaining gap

The first **checked** abnormal boundary is the result of
`compute_draft_logits(hidden_states)` in `_execute_sequential_markov_sampling`.
No first-failure worker files, epoch receipts, failing rank snapshots or prior
per-boundary finiteness checks exist in this archive. The failed JSON lacks
`ranks` because its snapshot RPC encountered EngineDeadError.
`request_identity_failure=null` therefore does not establish event ownership.
The admission mapping itself exists and has no recorded mapping error.

Source chain: target output/aux slices and actual `idx_mapping`, query offsets,
sampled/rejected counts enter `prepare_proposal_inputs`; context combination and
`precompute_and_store_context_kv` populate draft context slots; draft backbone
hidden enters the LM head; Markov base logits are checked before proposal
publication. Existing epoch/ownership checks invalidate a consumed proposal
before producing the replacement. A failed replacement is consistent with the
subsequent missing owner, but the exact cross-rank producer/consumer epoch link
is **not proven**. No independent ownership root cause is claimed.

There is no evidence isolating target hidden, context KV, draft hidden or LM
head as the producer. Existing cumulative offsets, sampling/rejection row maps,
per-group block/slot mapping and buffer aliases require runtime evidence at the
exit transition. No arithmetic, KV, attention, sampling, confidence allocation,
Core or custom-op repair is justified yet: **ROOT_CAUSE_NOT_YET_PROVEN**.

## Minimal additional observation

`--profile-nan-diagnostic` is accepted only for isolated B64 profiling. It uses a
separate additional-config key and installs the existing benchmark NaN observer
**after graph capture**. The public async-stream performance guard is unchanged.
It does not install whole-model/layer capture snapshots or change capture tiers.

The default diagnostic executes the original first ten points in one engine,
stopping after `ctx128-n4-t12-skewed`. No points are skipped and each call drains
normally. Existing observer checks target hidden/aux, proposal inputs, combined
context, written context KV, draft hidden and base logits. The first nonfinite
boundary distinguishes propagation from generation at the next boundary.
Worker-local files are written before the exception reaches EngineCore:

- `worker-first-failure/rank-*-first-failure.json`: immutable earliest failure,
  point, execution/proposal epochs, current selection and owner publication rows,
  actual FULL/non-FULL path, query spans, request/state/logits row mapping,
  sampled/rejected counts, positions/lengths, target and draft slot/block tensors,
  storage addresses and the preceding two execution records.
- `rank-*-latest.json`: propagation or most recent normal state. It never
  overwrites the first-failure evidence. Owner/selection values are sampled at
  each recorded stage; a before-target selection can still belong to the prior
  execution, whereas proposal/target-completed records identify the current one.
- Existing point raw JSON, admission receipts and lifecycle remain available.

Integer metadata and finiteness flags are copied; floating hidden/logit/KV
contents are not dumped. This adds diagnostic synchronization and file writes,
may alter reproduction, and is explicitly not a performance measurement.
`diagnostic.json` records the exact point prefix and effective benchmark argv
(including the existing synthetic-profile ignore-EOS setting). All artifacts
remain `performance_eligible=false`. Cost compilation rejects diagnostic-tagged
identity, and successful diagnostic runs return before writing `cost-profile.json`.
Completion without failure is reported as `completed_without_observed_failure`,
not a NaN fix or a usable cost profile. Normal profiling is unchanged by default.

## Server: one independent diagnostic only

Use the delivered exact SHA. The script preserves CANN/custom OPP, TP8+EP/MRV2,
K5, target FULL_DECODE_ONLY/draft eager, memory ratio 0.9, max model/token budget
8192, contexts 128/2048, output budget 512, warmup 2 and five retained samples.
It runs focused tests first, creates a fresh directory, checks resources and
keeps logs, all PIPESTATUS, worker files and evidence even on failure. It neither
kills other jobs nor starts B128/B256 or validate/repeat. No OPP build is needed.

```bash
cd /workspace/vllm-ascend-hust
set -o pipefail
SHA='<delivered-40-character-commit>'
git fetch origin feat/dspark && git merge --ff-only "$SHA"
```

After synchronization succeeds:

```bash
MANIFEST=/workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json
mkdir -p /workspace/dspark-results
DIAG_LOG=$(mktemp /workspace/dspark-results/profile-nan-driver.XXXXXXXX)
bash tools/dspark/run_dspark_profile_nan.sh "$SHA" "$MANIFEST" 2>&1 | tee "$DIAG_LOG"
DIAG_CODES=("${PIPESTATUS[@]}")
printf '%s\n' "${DIAG_CODES[*]}" > "$DIAG_LOG.pipestatus"
```

Return the new archive, especially every rank's first-failure and previous
executions. If no failure occurs, retain all evidence and report that observation
overhead may have changed the reproducer. A standalone failed-point run could
be a later control, but does not replace this cross-point test.

Local CPU/source/mock validation is not NPU validation. **SERVER_NOT_REVALIDATED**.

Local regression result: **323 passed, 3 skipped** across profile diagnostics,
existing NaN/replay boundaries, context, ID mapping, lifecycle, repeated inputs,
performance, confidence and graph RPC/replay tests. The three skips require
installed vLLM/Ascend. Changed-file pre-commit passed. The required full
`format.sh ci` ran in an isolated worktree and failed on existing baseline
issues; it auto-modified 78 unrelated files and no files in this change. No
unrelated formatting edits are included. These results do not prove the NaN
producer or an NPU fix.
