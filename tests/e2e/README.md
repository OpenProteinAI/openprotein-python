# E2E Test Suite

This directory contains end-to-end tests for the OpenProtein Python SDK.

## Setup

### Prerequisites

1. Install dependencies including pytest-xdist for parallel execution:
```bash
pip install -e ".[dev]"
# or with pixi
pixi install
```

2. Set environment variables for authentication:
```bash
export OPENPROTEIN_USERNAME="your_username"
export OPENPROTEIN_PASSWORD="your_password"
```

### Test Data

The test suite uses three dataset sizes:

1. **Small** (~41 sequences): `tests/data/AMIE_PSEAE.csv` (included)
2. **Medium** (~1000 sequences): `tests/data/PLACEHOLDER_MEDIUM_1000.csv` (you need to provide)
3. **Large** (~10000 sequences): `tests/data/PLACEHOLDER_LARGE_10000.csv` (you need to provide)

Tests will skip if medium/large datasets are not available.

## Running Tests

### Run all E2E tests
```bash
pytest tests/e2e -m e2e
```

### Run with parallel execution (recommended)
```bash
# Use all available cores
pytest tests/e2e -m e2e -n auto

# Use specific number of workers
pytest tests/e2e -m e2e -n 4
```

### Run specific test modules
```bash
# Embeddings tests only
pytest tests/e2e/test_embeddings_e2e.py -m e2e -n auto

# Predictor tests only
pytest tests/e2e/test_predictor_e2e.py -m e2e -n auto

# Assay data tests only
pytest tests/e2e/test_assaydata_e2e.py -m e2e -n auto

# Align tests only
pytest tests/e2e/test_align_e2e.py -m e2e -n auto
```

### Run specific test cases
```bash
# Test specific embedding model
pytest tests/e2e/test_embeddings_e2e.py::test_embedding_single_model[esm2_t33_650M_UR50D-1280] -m e2e

# Test specific predictor model
pytest tests/e2e/test_predictor_e2e.py::test_predictor_training_single_model[esm2_t33_650M_UR50D-MEAN] -m e2e
```

### Verbose output
```bash
pytest tests/e2e -m e2e -v -s
```

## Test Coverage

### Embeddings (`test_embeddings_e2e.py`)
- ✅ Parametrized tests across 4 models: ESM2, prot-seq, PoET, PoET-2
- ✅ Different reduction types (MEAN, MAX)
- ✅ Parallel model execution
- ✅ Batch size variations (1, 10, 100 sequences)
- ✅ Varied sequence lengths (short, medium, long, very long)
- ✅ Edge cases: empty sequences, invalid amino acids

### Assay Data (`test_assaydata_e2e.py`)
- ✅ Upload and retrieval workflow
- ✅ NaN value handling
- ✅ Data slicing and boundaries
- ✅ Parametrized tests across dataset sizes (small, medium, large)
- ✅ Metadata consistency
- ✅ Sequence validation
- ✅ Error handling for invalid slices

### Predictor (`test_predictor_e2e.py`)
- ✅ Parametrized training across 4 embedding models
- ✅ Parallel training of multiple predictors
- ✅ Multitask prediction (1, 2, 3 properties)
- ✅ Batch prediction (1, 10, 50 sequences)
- ✅ Parametrized tests across dataset sizes
- ✅ Predictor retrieval by ID
- ✅ Error handling: invalid sequences, empty sequences

### Align (`test_align_e2e.py`)
- ✅ MSA workflow and caching
- ✅ MSA sampling
- ✅ MAFFT alignment and caching
- ✅ ClustalO alignment
- ✅ Antibody numbering (AbNumber)

### Other Modules
- Fold tests (`test_fold_e2e.py`)
- Prompt tests (`test_prompt_e2e.py`)
- SVD tests (`test_svd_e2e.py`)
- UMAP tests (`test_umap_e2e.py`)
- Design tests (`test_design_e2e.py` - currently skipped)

## Shared Fixtures

Defined in `conftest.py`:

- `session`: Session-scoped OpenProtein connection
- `protein_complex_with_msa`: Pre-computed MSA for fold tests
- `assay_small`: Small assay dataset (~41 sequences)
- `assay_medium`: Medium assay dataset (~1000 sequences)
- `assay_large`: Large assay dataset (~10000 sequences)
- `test_sequences_varied`: Sequences of varying lengths
- `test_sequences_short`: Short sequences for quick tests

## Parallel Execution Strategy

The test suite is designed for parallel execution:

1. **Session-scoped fixtures** are shared across all workers
2. **Independent tests** can run concurrently
3. **Parametrized tests** distribute across workers automatically
4. **Backend parallelism**: Multiple models can process jobs concurrently

### Recommended Parallel Configurations

```bash
# Fast iteration (4 workers)
pytest tests/e2e -m e2e -n 4

# Maximum throughput (auto-detect cores)
pytest tests/e2e -m e2e -n auto

# Sequential (for debugging)
pytest tests/e2e -m e2e
```

## Timeouts

Default timeouts are configured per test type:

- Embeddings: 10 minutes
- Predictor training: 20 minutes
- Align (MSA): 10 minutes
- Fold: 10 minutes

Adjust in individual test files if needed.

## Adding New Tests

### Parametrized Test Example

```python
@pytest.mark.e2e
@pytest.mark.parametrize("model_id,expected_dim", [
    ("esm2_t33_650M_UR50D", 1280),
    ("prot-seq", 1024),
])
def test_new_feature(session: OpenProtein, model_id: str, expected_dim: int):
    model = session.embedding.get_model(model_id)
    # ... test logic
```

### Using Shared Fixtures

```python
@pytest.mark.e2e
def test_with_assay(session: OpenProtein, assay_small: AssayDataset):
    # assay_small is automatically created and reused
    predictor = session.embedding.esm2.fit_gp(
        assay=assay_small,
        properties=[assay_small.measurement_names[0]]
    )
```

## Troubleshooting

### Tests timing out
- Increase timeout values in test files
- Check backend status
- Reduce batch sizes for testing

### Parallel execution issues
- Some tests may have race conditions
- Use `-n 1` to run sequentially for debugging
- Check for shared state between tests

### Missing fixtures
- Ensure medium/large CSV files are present
- Tests will skip if fixtures are unavailable

### Authentication errors
- Verify OPENPROTEIN_USERNAME and OPENPROTEIN_PASSWORD are set
- Check credentials are valid

## Future Improvements

- [ ] Add more edge cases for error handling
- [ ] Test timeout scenarios explicitly
- [ ] Add rate limiting tests
- [ ] Test concurrent job submission limits
- [ ] Add performance benchmarks
- [ ] Test with malformed data
- [ ] Add integration tests for full workflows (embed → train → predict → design)
