# E2E Correctness Tests (foundational inference)

These tests assert that foundational-inference outputs are *correct*, not merely
well-formed. They complement the smoke suite in `tests/e2e/`.

## Layers
- **Invariants**: determinism, batch-invariance, reduction math, attention
  trimming/normalization.
- **Biological ground-truth**: native-residue recovery, zero-shot
  variant-effect correlation vs AMIE DMS, sequence recovery on a native backbone.
- **Baseline regression** (marked `differential`): outputs match a committed
  baseline within a tolerance calibrated from run-to-run jitter.

## Markers
- `correctness` — all tests here (also carry `e2e`).
- `differential` — the subset that compares against a committed baseline.

## Cache strategy

Inference is cached by sequence (and for PoET by `prompt_id`/`query_id`). Fixed
sequences hit the cache, which makes *same-input* checks (determinism,
batch-invariance) vacuous unless the call genuinely recomputes.

- **PoET / PoET-2** are the workhorse for genuine-every-run checks: each probe mints a
  fresh prompt/query from fixed content (`support.fresh_prompt`/`fresh_query`), which
  forces a recompute and stays reproducible against a baseline. Batch-invariance
  uses two different fresh prompts so neither call is cache-served.
- **esm2 / esmc + attention** can't be forced to recompute. They keep only
  cache-robust checks (reduction math, attn trim/normalization, semantic similarity)
  plus ground-truth. Their fixed-sequence baselines are genuine only on a cold cache.
- **Predictors** retrain every run (fresh assay, fresh `embeddings_id`).

Tolerance: `--baseline-repeats` jitter is ~zero for cached outputs, so the floors
(`RTOL_FLOOR`/`ATOL_FLOOR` in `baselines.py`) govern. Tune them from the `max abs/rel
dev` reported by an assert run.

## Running

Invariant + ground-truth only (no baseline needed):
```bash
pytest tests/e2e/correctness -m "correctness and not differential"
```

Full correctness suite (differential tests skip if no baseline):
```bash
pytest tests/e2e/correctness -m correctness
```

### Baselines: capture, then compare
1. Point `~/.openprotein/config.toml` at the API endpoint to capture from.
2. Capture, sampling jitter for tolerance calibration (run single-process):
   ```bash
   pytest tests/e2e/correctness -m differential --update-baselines --baseline-repeats=3 -p no:xdist
   ```
   This (re)writes `tests/e2e/correctness/baselines/<model_id>.json` and `.arrays.npz`.
3. Review the JSON diffs, then compare:
   ```bash
   pytest tests/e2e/correctness -m correctness
   ```
4. Commit the refreshed baselines.

Baselines are keyed by `(model_id, model_version, input_id, output_type)` and are
produced from **fixed** well-known sequences (cache hits are fine).
