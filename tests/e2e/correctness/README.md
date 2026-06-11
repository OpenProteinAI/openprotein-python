# E2E Correctness Tests (foundational inference)

These tests assert that foundational-inference outputs are *correct*, not merely
well-formed. They complement the smoke suite in `tests/e2e/`.

## Layers
- **Invariants** (no prod): determinism, batch-invariance, reduction math, attention
  trimming/normalization.
- **Biological ground-truth** (no prod): native-residue recovery, zero-shot
  variant-effect correlation vs AMIE DMS, sequence recovery on a native backbone.
- **Prod-baseline regression** (marked `differential`): outputs match prod (old
  workers) within a tolerance calibrated from prod run-to-run jitter.

## Markers
- `correctness` — all tests here (also carry `e2e`).
- `differential` — the subset that compares against a committed prod baseline.

## Cache strategy (important)

The backend caches inference **per-environment**, keyed by sequence (and for PoET by
`prompt_id`/`query_id`). Fixed sequences ⇒ cache hits, which makes *same-input* checks
(determinism, batch-invariance) vacuous unless the call genuinely recomputes.

- **PoET / PoET-2** are the workhorse for genuine-every-run checks: each probe mints a
  **fresh prompt/query from fixed content** (`support.fresh_prompt`/`fresh_query`) →
  cache miss → real recompute, still reproducible against a baseline. Batch-invariance
  uses *two different* fresh prompts so neither call is cache-served.
- **esm2 / esmc + attention** have no bust lever. They keep only cache-robust checks
  (reduction math, attn trim/normalization, semantic similarity) + ground-truth. Their
  fixed-sequence baselines are genuine **only on the first run after a deploy / cache
  clear** (cold-cache caveat, noted in each such test).
- **Predictors** retrain genuinely every run (fresh assay → fresh `embeddings_id`).

Tolerance: `--baseline-repeats` jitter is ~zero for cached outputs, so the floors
(`RTOL_FLOOR`/`ATOL_FLOOR` in `baselines.py`) govern. Tune them from the `max abs/rel
dev` reported by the first staging assert run (the real prod-vs-staging drift).

## Running

Invariant + ground-truth only (no baseline needed), against staging:
```bash
pytest tests/e2e/correctness -m "correctness and not differential"
```

Full correctness suite against staging (differential tests skip if no baseline):
```bash
pytest tests/e2e/correctness -m correctness
```

### Refreshing baselines (two-phase)
1. Point `~/.openprotein/config.toml` at **prod** (old workers).
2. Capture, sampling jitter for tolerance calibration (run single-process):
   ```bash
   pytest tests/e2e/correctness -m differential --update-baselines --baseline-repeats=3 -p no:xdist
   ```
   This (re)writes `tests/e2e/correctness/baselines/<model_id>.json` and `.arrays.npz`.
3. Review the JSON diffs, point config at **staging**, and run:
   ```bash
   pytest tests/e2e/correctness -m correctness
   ```
4. Commit refreshed baselines with a message noting the prod backend + date.

Baselines are keyed by `(model_id, model_version, input_id, output_type)` and are
intentionally produced from **fixed** well-known sequences (cache hits are fine).
