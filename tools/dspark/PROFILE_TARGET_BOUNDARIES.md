# DSpark profile: bounded target internal boundaries

The resulting PQHA55hF run narrowed the interval to layer 0 output → layer 1
output. See [the new audit and local layer runbook](PROFILE_TARGET_LAYER.md).
The evidence and PENDING labels below describe the preceding delivery.

**Diagnostic supplementation. Root cause UNKNOWN. New NPU verification PENDING.**
This follows the `auxiliary-transfers` reproduction. It changes neither Core,
custom operators, weights, confidence allocation, sampling, mixed verification
lengths nor target/draft ACLGraph dispatch. This work belongs to
vllm-ascend-hust / vllm-hust; it has no SpecRhythm dependency.

## Independently checked evidence

The local archive `dspark-large-batch.f9kqffNA-evidence.tar.gz` was opened and
safely extracted. SHA256 is
`472fa63f85876ef857387abbb00660ff1690f81b12779bbce60fd43a02326744`.
All regular-file hashes matched their tar members: **96 members, 169717044
expanded bytes**. Archive contents were evidence, not execution instructions.
The separately mentioned pasted-text attachment was not available locally;
none of the conclusions below depends on it.

The local Plugin checkout, freshly fetched `origin/feat/dspark`, and actual
running Plugin all matched `da7c760b608488235725bd9de2a9afdd58304683`; both local
repositories were clean. Core was and remains
`897306c43bf800e2480cb5c0f3e2da408d85a2fd`. There were no intervening commits to
reconcile. The delivery commit adds only the changes described below.

[PROFILE_f9kqffNA_AUDIT.json](PROFILE_f9kqffNA_AUDIT.json) contains per-rank member
hashes, independently compared histories, transitions/owner epochs, compact
boundary results, checkpoint provenance and log line anchors. Assertions used
all five first-error files on **every rank**, not only rank 0. These are offline
checks; they are not new NPU tests. Server `focused.log` reports **918 passed,
14 warnings** for the old Plugin. Nine completed-point file hashes match
`runs/b64/retained.json`. This does not certify timing as usable costs.

| Execution / new proposal | Raw/persistent/consumed auxiliaries 40/41/42 | Draft head input/output |
| --- | --- | --- |
| 1805 / 1792 | All recorded rows finite, equal transfers | `both_finite` |
| 1806 / 1793 | All recorded rows finite, equal transfers | `both_finite` |
| 1807 / 1794 | Target row 0 NaN; every other row, including padding, finite | Candidates 0–4 hidden/logits NaN; other candidates finite |

There is no Inf in these records. All ranks report `recording_error=null`,
98/98 completed combined packets, numeric counts `both_finite=97`,
`hidden_nonfinite=1`. Auxiliary coverage counts are 95 FULL and three explicit
non-FULL records. The last three records are all FULL, with no missing boundary.
All raw and consume receipts equal that record's execution (1805, 1806, 1807),
`raw_replay_verified=true`; no transfer reports a differing row.

Evidence anchors under `runs/b64/worker-first-failure/`:

- `rank-N-first-nan.json`: complete three-round numeric and auxiliary histories.
- `rank-N-auxiliary-first-nan.json`: first valid auxiliary NaN, preceding two
  target rounds, device integers, current combined `head_flags`.
- `rank-N-first-failure.json`: original Markov exception, execution/epoch/owner
  records. First-error snapshots are exclusive and cannot be overwritten by
  the following owner/EngineDead failures.
- The corresponding `first-nonfinite` files have identical auxiliary histories.
  Counters in earlier snapshots may precede the next latch increment.

The configured tenth point is four requests, ell `[5,3,0,0]`, 12 target tokens,
capacity 12, prompt 128, output 512. **The failing execution is different:**

| Packed request row | Actual request ID | Local pool row | Query length / ell | Target positions / sequence length |
| --- | --- | --- | --- | --- |
| 0 | `batch10-3-94151eee` | 63 | 1 / 0 | 222 / 223 |
| 1 | `batch10-2-9de4223b` | 62 | 4 / 3 | 251–254 / 255 |
| 2 | `batch10-1-897ec365` | 61, or 60 on ranks 3/6 | 6 / 5 | 524–529 / 530 |

Actual query starts are `[0,1,5,11]`: three requests, 11 valid target rows,
capacity 12; the draft generates 15 candidates. The padding position is the
old value 643 and its token ID is 90. Both are outside the valid query prefix.
The affected target input ID is 58; its next five draft positions are 223–227
under the inspected proposal arithmetic. Rejected counts are `[0,0,0]` and
sampled counts `[1,4,6]` in all three rounds. These are target/proposal integers,
not a measurement of the draft's internal arithmetic in this experiment.

Every rank retains transitions 1710→1711, 1711→1712 and **1797→1798**. The last
is four to three requests, nine executions before the NaN. Thus neither the
exit iteration nor a universal `[63,62,61]` pool mapping should be asserted.
The all-rank record still does not identify the first failing rank/operator.

## Replay freshness and numerical scope

At the running SHA, `ModelWithContext` calls the original compiled Target,
then `AuxiliaryCapture.model_outputs` copies its three actual returned aux
tensors into separately owned per-captured-size buffers. These copies and their
receipt copies are inside the captured FULL closure, before frozen Core's
persistent-output copies. Each real replay clears receipts to -1 and arms an
execution tensor. `DiagnosticReplayGraph` calls the original graph once; after
return it snapshots receipts and reduces raw/persistent statistics. Consumption
reduces the actual valid-row concatenation before context projection and owns
another receipt snapshot. All flags/integers survive independently of reusable
buffers until the one combined host transfer, before the original Markov check.

Capture/warmup runs have no observer execution owner. An omitted/no-op replay
cannot publish the new execution ID. A later replay cannot alter the owned
flags already taken from this one. The source placement, current receipts and
matching values support **raw Target return already nonfinite**, ahead of the
persistent transfer. They do not prove that the `mean` result was bad at the
instant of its computation: compiler storage reuse inside the rest of Target
before return is still within the unobserved interval.

The earliest observed bad boundary is the returned auxiliary for layer 40.
Its model input token IDs/positions/query starts/sequence lengths were recorded;
its incoming hidden state and actual attention/KV read contents were not.
Layer 40 is therefore an endpoint of the known interval, not the first proven
NaN-producing decoder layer. The existing R9 all-layer full-tensor banks would
be too broad and assume consecutive uniform capture tiers. Only their existing
model call sites and replay protocol are reused here.

## Metadata and state source audit

| Field/path | What was checked | Remaining coverage gap |
| --- | --- | --- |
| Request order, pool mapping, effective query ends | CPU/device query and pool fields agree on each rank; zero rejects retain ends `[1,5,11]`; IDs remain stable over the last three rounds | Earlier pool reuse and all historical request-state values are not in the numeric ring |
| Input IDs/positions | Actual capture-bound input prefixes equal current Target and proposal prefixes, all 11 valid rows | Padding still contains old values; equality of inputs says nothing about their produced hidden states |
| Captured attention query/seq fields | All named captured root/decode query arrays have `[0,1,5,11,11,…]`; all seq arrays have `[223,255,530,0,…]` | Previous collector looked for `positions`, but DSA calls the field `decode.input_positions`; actual RoPE, `start_pos`, SAS/QLI metadata were not read |
| Padding contract | `fill_varlen_query_padding` gives zero-query tails; `prepare_pos_seq_lens` zeros non-request seq rows; slot kernel fills tail with PAD_ID. Non-A5 slots are block/offset pairs, not flat IDs | Actual slot values and block contents were not exported; source intent is not device validation |
| `is_padding` | Archive records twelve ones. Runner updates it only under Core `VLLM_MOE_SKIP_PADDING`; Core's generic MoE consumer is also gated. DSA uses its own query/seq/slot metadata | The active environment value was not recorded; this vector alone cannot prove an inverted mask or padding producer |
| SWA/compressor/indexer | DSA decode consumes the real group's block table, query prefix, seq/start positions, RoPE and SAS/QLI metadata. Compressor metadata runs at consumption inside the graph. Its kernel assigns zero compressed rows to zero-length query entries | Actual KV/state tensors, sparse indices, compressed prefix values and allocated block-table prefixes were not numerically observed |
| Capture versus replay objects | Capture-bound query/seq buffers are persistent views. Builders refresh their persistent slot, start-pos, SAS/QLI buffers during preparation; DSA's graph task-update is a no-op | Scalars baked at capture describe capacity, not actual requests. All kernel read ranges and derived metadata have not been validated on device |
| Target versus Draft | Draft creates its own attention builders and batch-local dictionaries; `_build_draft_forward_metadata` uses draft groups/query/seq. Context slots are cloned before shared slot storage is reused for queries. Existing target/draft KV occupied-byte checks remain | Shared backing allocation/addresses are not proof of valid KV contents. No archive proof excludes an incorrect earlier state write or an async custom-op dependency |

Relevant source: `worker/v2/model_runner.py`, `input_batch.py`, `block_table.py`,
`model_states/default.py`, `attn_utils.py`; `attention/dsa_v1.py`; non-A5
`device/device_op.py`; `csrc/attention/compressor_metadata/op_kernel/compressor_metadata.h`;
`worker/v2/spec_decode/dspark/speculator.py`; and frozen Core
`v1/worker/gpu/{input_batch,model_runner,cudagraph_utils}.py`.

The actual non-A5 selector uses `npu_sparse_attn_sharedkv`, whose local
[operator contract](../../csrc/attention/sparse_attn_sharedkv/README.md) defines
TND query prefixes as nondecreasing (equal adjacent entries are permitted).
For PA_ND it ignores `cu_seqlens_ori_kv`/`cu_seqlens_cmp_kv`; `seqused_q` is also
documented as ineffective. These are not interchangeable token-validity masks.
The compressor-metadata kernel explicitly emits no compressed rows when query
length is zero. This checks the relevant conventions, not every kernel input.

No concrete out-of-range index, incorrect copy extent, interface violation or
broken stream dependency was proved. Plausible remaining paths include Target
attention/KV or compressor/indexer state, HC/norm, MoE/communication and internal
output reuse. Matching query/seq fields excludes only those particular recorded
mismatches. B64 memory capacity, padding, request exit and `ell=0` are not proved
causes. In particular, ell=0 means one target query and still drafts K5.

## Two error chains

All ranks' histories show this sequence:

1. Execution 1805 consumes owners 1791, prepares 1792, passes Markov and publishes
   owners 1792. Execution 1806 consumes 1792 and publishes 1793 successfully.
2. Execution 1807 enters with owners 1793. Old verification/release completes;
   `proposal_prepare.return` binds new epoch 1794 with no published owners.
3. `markov.error` has attempt 1794, `markov_step_epoch=null`,
   `published_proposal_step_epoch=null`, and an empty owner map. `_build_core_proposal`
   is after successful Markov; this proposal is not published.
4. Later log stacks, on every rank, enter a new `execute_model` and fail in
   `ConfidenceVerification.select` before Target execution, because
   scheduled candidate requests have no current owners. This checks a new
   scheduler call; it does not consume a successfully returned 1794 proposal.

The old exclusive first-failure file ends at 1807. The subsequent call is later
than 1807 (1808 if it is the immediately following call), but **its exact numeric
execution and scheduler producer epoch were not saved**. Logs establish call
ordering and source establishes the selection site; do not label 1808 as a
measured archive field. The new diagnostic writes up to four deduplicated error
events with current execution/epochs, scheduler receipt and current owners to
close that small CPU-only gap. Nested rethrows do not replace the first failure.

No independent pre-NaN owner lifecycle failure is demonstrated. The observed
owner error is consistent with the first-failure chain. Original EngineCore
error still names the numerical failure. Cleanup completed within the existing
8-second bound; no cleanup error/timeout, supervisor `signals_sent=[]`. There
is no reason to repeat the earlier snapshot-RPC/indefinite-exit investigation.

## New default-off target-boundaries mode

`--profile-experiment target-boundaries` keeps the auxiliary-transfer and head
observations for comparison, without re-enabling the drafter upstream probes.
At model construction, before compile/profile, it binds a small flag bank to
selected existing Target call sites. The plan is derived from the first
configured auxiliary layer, with early layers and ten-layer checkpoints; it is
bounded to 24 cuts and capture capacity 384. For this model it has **15 cuts**:

1. Token embedding output.
2. Layer outputs 0, 1, 2, 3 (early SWA and compression regimes), 9, 19, 29, 39.
3. Layer 40: normalized attention input, attention return (including output
   projection/communication), HC attention residual update, normalized FFN
   input, MoE/FFN return, HC FFN output update.

This brackets early faults by layer/interval, and separates six stages if the
layer 39 endpoint is finite. It cannot resolve an unobserved interval's exact
layer/operator in the same run. HC pre and norm remain grouped. The plan is
chosen because layer 40's auxiliary is already an observed endpoint, not because
layer 40 was assumed to be the producer. All later raw aux observations remain.

Each cut reduces NaN/Inf across the row's full feature/HC axes **in the compiled
model and actual graph replay**, and writes two flags plus a device execution
receipt. No Python forward hook is required to execute again on replay.
Unselected writes specialize to no device operations. The symbolic row minimum
is retained explicitly: a large 8192-row profile graph must work at smaller
capture sizes even when frozen Core bypasses Dynamo guards.

A shared flag bank suffices across capture shapes: immediately after a replay,
the observer takes owned copies of its flags and per-cut receipts. It verifies
all cuts against the same actual raw replay receipt. Missing/stale cuts have
`INVALID_RECEIPT` and no finite bracket; non-FULL is `NON_FULL_UNOBSERVED`.
Actual capture-bound `decode.input_positions` and `start_pos` are also copied
before replay, deduplicated and bounded to 32 tensors of at most 384 integers.
No new block-table or KV dump is claimed.

Results are under `auxiliary.rounds[].target_internal` in the same three-target
history, with point/rank/execution/proposal IDs and actual row mapping.
`valid_row_brackets` reports each affected valid row's last **observed** finite
and first **observed** nonfinite endpoint; absence of a bracket requires checking
coverage before interpreting it. Padding rows have null request identity and
remain separately visible, but do not latch a valid-request failure.
`root_cause=UNKNOWN` is deliberately retained.

`rank-N-target-first-nonfinite.json` exclusively saves the first valid-row Target
anomaly and preceding two records, flushed/fsynced before the Markov guard.
First-nan/first-failure files also contain these rounds. `rank-N-error-events.json`
is an atomic, bounded record of up to four distinct errors; it never overwrites
the original exclusive first-failure file. If Target produces no proposal,
a partial record is drained on error/next execution/point change with a null
proposal epoch. Normal finite rounds add no writes.

## Observation cost and limits

New persistent bank: **11648 bytes/rank** at 15 cuts × 384 rows (two boolean
flags), fifteen int64 receipts and one epoch scalar. This is additional to the
auxiliary-transfer raw banks (~17.86 MiB/rank) and does not include compiler/graph
workspaces. Reductions scan each selected full feature row. At capacity 384,
a boolean mask for `[384,4,4096]` occupies 6 MiB; at capacity 12 it occupies
192 KiB. Compiler fusion/reuse and NPU peak temporary memory remain unmeasured.
Do not equate compact evidence size with zero device overhead.

Each replay adds 30 row reductions, 15 flag writes and 15 receipt writes;
arming clears/sets the compact receipt/epoch buffers outside the graph. Owned
post-replay copies add 3000 packet bytes at capacity 12 for flags/receipts,
plus bounded position/start metadata. These join the existing auxiliary/head
packet: **one blocking D2H per completed proposal/rank**, no per-cut host wait
and no new global synchronization. Partial targets may drain another packet
without reaching the head, as in auxiliary-transfers. Counters report actual
packets; record sizes follow current capacity, not historical execution 1807.

The additional reductions, graph memory scheduling, compact copies, wait payload
and exceptional file writes can alter reproduction. Flag receipts validate
execution of this observation path, not correctness of every target kernel.
Default-off mode creates no target banks, reductions, copies or synchronization.
Existing baseline/metadata/context-sync/numeric/upstream/auxiliary meanings remain.
Every result stays `performance_eligible=false`, and diagnostic cost identities
remain rejected by cost compilation.

## One next server run

Use the delivered commit SHA as `DSpark_SHA` below. The final handoff supplies the
literal SHA; the commands verify it before execution. Run from the existing CANN
and custom-OPP shell. They use only a clean fast-forward of `feat/dspark`, create
no merge commit, preserve the old result directories, and never alter Core.
The strict options are confined to a child Bash. Parent failure handling leaves
the interactive terminal alive. No command after a failed preflight starts NPU.

```bash
DSpark_SHA='<literal signed-off SHA in the handoff>'
if bash -s -- "$DSpark_SHA" <<'BASH'
set -euo pipefail
sha=$1
plugin=/workspace/vllm-ascend-hust
core=/workspace/vllm-hust
manifest=/workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json
test -z "$(git -C "$plugin" status --porcelain)"
test -z "$(git -C "$core" status --porcelain)"
test "$(git -C "$core" rev-parse HEAD)" = 897306c43bf800e2480cb5c0f3e2da408d85a2fd
test "$(git -C "$plugin" branch --show-current)" = feat/dspark
git -C "$plugin" fetch origin feat/dspark
test "$(git -C "$plugin" rev-parse origin/feat/dspark)" = "$sha"
git -C "$plugin" pull --ff-only origin feat/dspark
test "$(git -C "$plugin" rev-parse HEAD)" = "$sha"
bash "$plugin/tools/dspark/run_dspark_profile_control.sh" "$sha" "$manifest" target-boundaries
BASH
then
  printf 'Diagnostic command completed; inspect coverage and numerical evidence.\n'
else
  DSpark_RC=$?
  printf 'Diagnostic/preflight exit=%s; stop here and retain the first error.\n' "$DSpark_RC"
fi
```

The entry prints a new `SERVER_RESULT_DIR` immediately, then records every phase
log/exit code. It checks source, checkpoint, frozen input provenance and the
installed focused tests before loading the engine. Config/index/confidence-head
hashes and the input manifest hash must match the audited run. The original
archive lacks all-target-weight hashes; unchanged model directory and revision
are required, and full weight equivalence is not retroactively asserted.

The same model path/revision, TP8+EP, MRV2, K5, BF16/Ascend quantization, greedy
sampling/seed 0, asynchronous scheduler, original synthetic generator, single
engine and first-ten point order are retained. Target FULL_DECODE_ONLY, eager
draft, captures `[6,12,24,48,96,192,384]`, 8192 limits, block size 32, memory 0.9,
contexts `[128,2048]`, output 512, warmup 2 and samples 5 are unchanged.
There is no B128/B256 continuation, full cost table or performance comparison.
A fresh run can differ in request UUIDs, timing and epochs; this is **not a
same-state graph/eager control**. No such control is claimed or scheduled.

For this mode only, the existing owned-process supervisor adds a **3600-second
wall-clock limit**, starting before engine load, and a local stop-file input.
The existing 20-second failure grace and 5-second TERM grace stay unchanged;
RPC timeout 120 seconds and cleanup timeout 8 seconds also stay unchanged.
The limit produces failure evidence, never a synthetic PASS or timeout increase.
A stop file created during preflight prevents the NPU phase from starting.
Only the child session it created is signalled. First numerical/engine failure
wins over a later stop/timeout, and any failure stops subsequent points/stages.

While the command runs, a second shell can inspect or request a controlled stop.
Set `DSpark_RUN` to the exact directory printed by this invocation, not a glob
or the newest directory belonging to another job:

```bash
DSpark_RUN='/workspace/dspark-results/dspark-large-batch.<printed suffix>'
cat "$DSpark_RUN/runs/b64-supervisor.json"
cat "$DSpark_RUN/runs/b64/frontend-operation.json"
tail -n 60 "$DSpark_RUN/generation.log"
# Optional controlled stop of this diagnostic only; allow up to 20+5 seconds.
touch "$DSpark_RUN/STOP"
```

Do not use Ctrl-C/killall as the normal stop path. The original wrapper exports
`$DSpark_RUN-evidence.tar.gz` and its SHA256 even after generation failure.
The export footer now preserves a nonzero original phase exit even if tar/hash
also fails. `status.txt`, per-phase `*.pipestatus` and the original rank failure
remain authoritative; export failures cannot turn the numerical run into PASS.

If export itself failed, wait for supervisor completion, then export to a new
retry filename. This separate command records only export status and cannot
replace the original `status.txt` or worker evidence:

```bash
if bash -s -- "$DSpark_RUN" <<'BASH'
set -euo pipefail
run=$1
test -f "$run/status.txt"
retry=$(mktemp --suffix=.tar.gz "$run-evidence.retry.XXXXXXXX")
tar -czf "$retry" -C "$(dirname "$run")" "$(basename "$run")"
sha256sum "$retry" > "$retry.sha256"
printf 'Retry evidence: %s\n' "$retry"
BASH
then
  printf 'Export completed; original run status is unchanged.\n'
else
  DSpark_EXPORT_RC=$?
  printf 'Export failed (%s); preserve the result directory.\n' "$DSpark_EXPORT_RC"
fi
```

Minimum return: **one evidence tar.gz and its SHA256 file**. It must include all
ranks' `target-first-nonfinite`, `first-nan`, `auxiliary-first-nan`, `first-failure`,
`error-events` when generated, plus latest snapshots, ten-point plan/results,
commands, provenance, phase exit statuses, generation log and cleanup/supervisor
receipts. Missing anomaly files on a finite run are expected; return the archive
anyway. No full hidden/logits/KV tensor export is requested.

## Acceptance and next decision

First validate SHAs/provenance, original point prefix and bounds, all-rank
`recording_error=null`, completed packet counts and current receipts. On a bad
FULL round, require `target_internal.coverage=FULL`, all 15 cuts and raw/consume
receipts tied to the same execution, actual valid/padding mapping and the prior
two target records. A missing receipt is unavailable evidence, not finite.

- Earliest bad cut before layer 40: continue within that earlier layer/interval.
- Layer 39 output finite, one layer 40 stage first bad: narrow to the operations
  between those endpoints, including their actual input/state reads.
- All internal cuts finite but raw aux 40 bad: investigate HC mean, output
  lifetime/reuse or the remaining interval to model return; the copy after raw
  return is already separately bracketed.
- Early embedding bad: investigate input embedding/weights/communication path.
- No NaN: report **not reproduced under target-boundaries**. It is not a repair,
  proof of graph correctness or a usable profile.

Only a subsequent demonstrated defect and NPU verification can change the status
to a production repair. This delivery stops after the commit/runbook handoff and
waits for the user-run server result.

## Local validation

On local Python 3.12 / CPU Torch 2.14.0: **460 passed, 3 skipped** in the
related standalone suite; **98 passed, 110 deselected** in the capture/metadata
source and ABI selection. The three skips require installed vLLM/Ascend;
unselected runtime variants are delegated to the server focused suite.
Changed-file manual pre-commit hooks, shell syntax and `git diff --check` passed.
Required `bash format.sh ci` ran with the staged patch in an isolated worktree
and returned 1: the same eight failing hooks and same 78 automatically modified
files as the preceding delivery, with no changed task file among them. Existing
failures are Ruff check/format, codespell, typos, Markdown, workflow/shell lint
and forbidden-import checks; unrelated automatic edits were discarded.

CPU/mock tests
execute real decoder statements, actual output-copy/replay wrappers, profile
factory and capture ABI, with leaf NPU kernels replaced. The new tests cover
layer/submodule faults, NaN/Inf row identity, zero-query/padding layouts, request
reorder/exit/reused local pool rows, dynamic FX replay without guards, receipt
freshness, snapshots surviving reuse, partial records, first-failure storage
errors, deduplicated later errors, scope/default-off behavior and owned-process
stop/deadline/export status handling.

Installed vLLM/Ascend runtime variants and real DSA/HC/MoE kernels are not
available on the local Mac. Server focused tests are a prerequisite in the
runbook; they must not be confused with executing the NPU reproduction itself.
