# Changelog

## 0.18.0 (2026-11-24)


### ⚠ BREAKING CHANGES

* **models:** the five `session.models` handles from v0.17.0 (`rfdiffusion`, `proteinmpnn`, `solublempnn`, `boltzgen`, `esmif1`) submit through the models API. `get_metadata()` is removed, and `metadata` now returns the model's card, `openprotein.models.ModelMetadata`: `model_id` replaces `id`, `methods` lists what the model runs, and `output_types`, `input_tokens`, `output_tokens` and `token_descriptions` are gone. `single_site` and `indel` take `base_sequence`. ESM-IF1, ProteinMPNN and SolubleMPNN `generate` takes `query`, `design` and the model's parameters by keyword and returns a `SequenceGenerationFuture`. RFdiffusion and BoltzGen `generate` returns a `StructureDesignFuture`. `get()` and `wait()` on a `SequenceGenerationFuture` return the generated sequences as strings, where v0.17.0 returned rows with `sequence` and `score`. Read the scores with `get(output="loglikelihood")`.
* **models:** on `session.models.esmif1`, `score` returns a `LogLikelihoodFuture` and `single_site` a `SingleSiteScoreFuture`. Their `get()` and `wait()` return the scores alone, as floats with `nan` and infinities included: one per scored sequence in the order submitted, or for `single_site` 20 per scanned position, positions in sequence order and substitutions in `openprotein.utils.sequence.AMINO_ACIDS` order, the position's own residue included. v0.17.0 returned `Score` rows with `name`, `sequence`, `score` and `query_id`, and `SingleSiteScore` rows with `mut_code`.
* **models:** ESM-IF1, ProteinMPNN and SolubleMPNN `generate` raises `TypeError` for a `query` and a `design` together, since the design's structures are the queries. v0.17.0 sent both. ESM-IF1 takes one query, so a list of several raises `ValueError`.
* **models:** RFdiffusion and BoltzGen design from a query only. `generate` takes `query`, `num_samples` (was `N`) and `random_seed`, plus `diffusion_batch_size`, `step_scale` and `noise_scale` for BoltzGen, and the API refuses any other keyword. RFdiffusion's `contigs`, `structure_file`, `inpaint_seq`, `provide_seq`, `hotspot`, `T`, `partial_T`, `use_active_site_model`, `use_beta_model`, `symmetry`, `order`, `add_potential`, `scaffold_target_structure_file` and `scaffold_target_use_struct` are gone, and so are BoltzGen's `design_spec`, `structure_file`, `scaffolds`, `scaffold_set` and `extra_structure_files`. `openprotein.scaffolds` is removed. Describe the design with the query, e.g. `Protein.from_expr(60)` for a 60-residue de novo design. `predict` is no longer an alias of `generate` on these two handles, so call `generate`.
* **models:** `openprotein.models.base.ProteinModel` and the `openprotein.models.foundation` modules (`boltzgen`, `boltzgen_schema`, `esmif1`, `proteinmpnn`, `rfdiffusion`, `solublempnn`) are removed. The class names v0.17.0 exported from `openprotein.models` still resolve, see Deprecations.
* **molecules:** `openprotein.chains` and `openprotein.protein` are removed; import from `openprotein.molecules`. `Protein.make_cif_string()` and `make_pdb_string()` are removed; use `to_string()` and `to_string(format="pdb")`. All four warned in v0.17.0 that they would be removed.
* **embeddings:** PoET-2 `score`, `indel` and `single_site` raise `InvalidParameterError` for `decoder_type="mlm"`. Leave it out to score with the `clm` decoder.
* **prompt:** `Query.get_as_complex()` keeps the chain ids of the structure it was created from. v0.17.0 returned a single-chain mmCIF query under chain `A`; read the chain from the returned complex's keys.
* **align:** `MSAFuture.get()`, `MSAFuture.wait()` and `session.align.get_msa(job_id)` read the stored MSA: a3m for a search or an uploaded MSA, FASTA for MAFFT, Clustal Omega and AbNumber. Read before a search finishes, it raises `HTTPError` 409 `msa <id> is RUNNING`, and an MSA that is not stored raises `HTTPError` 404.
* **base:** every request made without `timeout` uses the session timeout, 180 s unless set with `openprotein.connect(..., timeout=...)`. v0.17.0 applied it to POST only, so a slow GET could wait indefinitely. Raise the session timeout, or set `session.timeout = None` for no limit.
* **base:** when retries on a failing response are spent, the SDK raises its own error for the last response (`openprotein.errors.HTTPError`, or `openprotein.base.CloudFrontError` / `openprotein.base.TimeoutError`) instead of `requests.exceptions.RetryError`.
* **jobs:** a future with no repr of its own prints as `Name(job_id=..., status=..., args=...)`. v0.17.0 printed the job record.
* **molecules:** writing a `Protein` with a length range, such as `Protein.from_expr("10..20")`, to a structure raises `ValueError`. v0.17.0 raised `RuntimeError`.
* **utils:** `openprotein.utils.uuid.is_valid_uuid` accepts any UUID version by default, since job ids are not all version 4. Pass `version=4` for the v0.17.0 check.
* **api:** `PATH_PREFIX` is removed from the clustering, design, embeddings, fold, predictor, svd and umap `api` modules, with no deprecation.
* **errors:** `openprotein.errors.UPGRADE_MESSAGE` and `openprotein.base.SKEW_SECONDS` are removed, with no deprecation. `openprotein.config` is unchanged.
* **common:** `openprotein.common.residue_contants` is renamed to `openprotein.common.residue_constants`, with no alias.


### Deprecations

Each of these still works in 0.18.0, warns with a `FutureWarning` that names the replacement, and is removed in 0.19.0.

* **align:** `session.align.mafft_file`, `clustalo_file` and `abnumber_file`: pass sequences and names to `mafft`, `clustalo` and `abnumber`.
* **align:** `session.align.upload_msa` and `openprotein.align.api.msa_post`: store an MSA with `session.data.MSA.create(f.read(), format="a3m")` (formats `a2m`, `a3m`, `csv`, `fasta`, `sto`) and sample it with `MSA.sample_prompt`, which takes `replicates`, `max_msa_sequences`, `max_msa_tokens`, `seed` and `force_include_first` in place of `num_ensemble_prompts`, `num_sequences`, `num_residues`, `random_seed` and `always_include_seed_sequence`, and returns a `session.data.Prompt`. For a seed sequence, use `session.align.create_msa`.
* **align:** `openprotein.align.api.mafft_post`, `clustalo_post` and `abnumber_post` called with `sequence_file`: pass sequences and names to the same functions.
* **align:** `openprotein.align.api.prompt_post`: use `MSAFuture.sample_prompt`, or `session.data.MSA(id=msa_id).sample_prompt(...)` with the arguments above.
* **align:** `AlignFuture.get_input`, `openprotein.align.api.get_input` and `openprotein.align.api.get_align_job_inputs`: for the MSA use `MSAFuture.wait()` or `session.data.MSA(id=...).to_string()`, for an MSA job's seed use `get_seed`, and for a prompt use `Prompt.get()`. Other align jobs' input has no replacement, so keep your own copy.
* **data:** `session.data.create`, `get`, `list` and `load_assay`, and `len(session.data)`: use `session.data.Assay.create`, `session.data.Assay(id=...)`, `session.data.Assay.list()` and `len(session.data.Assay.list())`.
* **data:** `update(assay_name=, assay_description=)` on an assay: use `update(name=, description=)`.
* **models:** `generate(seed=...)` and `single_site(sequence=...)` on ESM-IF1, ProteinMPNN and SolubleMPNN: use `random_seed=` and `base_sequence=`.
* **models:** `openprotein.models.BoltzGenModel`, `ESMIF1Model`, `ProteinMPNNModel` and `RFdiffusionModel`, also under `openprotein.models.models`: use `openprotein.models.BoltzGen`, `ESMIF1`, `ProteinMPNN` and `RFdiffusion`. The names resolve to the new handles, whose methods changed (see Breaking Changes).
* **models:** `openprotein.models.SolubleMPNNModel`: use `session.models.solublempnn`.
* **models:** the embedding and fold classes imported from `openprotein.models` (`ESMModel`, `OpenProteinModel`, `PoETModel`, `PoET2Model`, `AlphaFold2Model`, `Boltz1Model`, `Boltz1xModel`, `Boltz2Model`, `ESMFoldModel`, `ESMFold2Model`, `ESMFold2FastModel`, `MiniFoldModel`, `ProtenixModel`, `ProtenixV2Model`, `RosettaFold3Model`): import them from `openprotein.embeddings` and `openprotein.fold`.
* **models:** `openprotein.models.BoltzGenFuture` and `RFdiffusionFuture`: submit designs with `session.models.boltzgen.generate(...)` and `session.models.rfdiffusion.generate(...)`, which return `openprotein.jobs.StructureDesignFuture`. `session.jobs.get(id)` loads a v0.17.0 design job as a `StructureGenerationFuture`.


### Features

* **models:** `session.models.<model>` reaches every listed model through a handle class per base model: `ESM2`, `ESMC`, `PoET`, `PoET2`, `Boltz2`, `ProteinMPNN`, `ESMIF1`, `RFdiffusion`, `BoltzGen` and the rest, importable from `openprotein.models`. `session.models.get_model(id)` loads a handle by id.
* **models:** each handle has `embed`, `logits`, `attn`, `score`, `single_site`, `indel`, `generate`, `fold` and `predict`, as the model allows, plus `metadata` and `methods`. `embed(assay)` embeds an assay dataset.
* **models:** `design=` on ESM-IF1, ProteinMPNN, SolubleMPNN and PoET-2 `generate` takes an RFdiffusion or BoltzGen job or its id, finished or not. The generation runs once the design finishes, with each designed structure as a query, so ESM-IF1 generates from a design with several structures.
* **models:** `session.models.list_models(verbose=False, *, path=None)` lists system models, across every page. `path` keeps the models with a matching method path. Fitted models are listed by `session.models.GP.list`, `SVD.list` and `UMAP.list`.
* **models:** `session.models.bioemu`, `alphaflow` and `esmflow` sample structure ensembles with `sample(inputs, num_samples=...)`. Each takes its own typed input: `BioEmuInput`, `AlphaFlowInput` or `ESMFlowInput`.
* **models:** `session.models.opendde` and `opendde_abag` fold with OpenDDE.
* **models:** estimators `session.models.SVD(n_components=None, seed=None)`, `UMAP(n_components=None, n_neighbors=None, min_dist=None)` and `GP(kernel="rbf", period=None, alpha=None, name=None, description=None)` hold a fit's settings. `fit(X)` fits on an embeddings job, and UMAP and GP also take a fitted SVD as `X`. `fit(model=..., sequences=...)` on SVD and UMAP, or `fit(model=..., assay=...)`, has a system model or a fitted SVD embed the input as part of the fit. `GP.fit` takes `assay=` and `property=`, and `assay=` alone trains on one-hot features.
* **models:** `fit` submits straight away and returns an `SVDFitFuture`, `UMAPFitFuture` or `GPFitFuture`, which keeps `dataset`. `wait()` returns the fitted `SVDModel`, `UMAPModel` or `GPModel` and raises if the fit failed. A fit with the wrong mix of `X`, `model=`, `sequences=` and `assay=` raises before anything is sent.
* **models:** `SVDFitFuture.transform` and `UMAPFitFuture.transform` take sequences or an assay and submit a projection that runs once the fit finishes. `embed` is the same call, and `fit_transform` on SVD and UMAP returns the same fit future as `fit`.
* **models:** `SVD(id=...)`, `UMAP(id=...)` and `GP(id=...)` on `session.models` load a fitted model by its id, and `list(limit=None, offset=None)` lists the caller's fits.
* **models:** fitted models: `SVDModel` has `transform` (or `embed`), `n_components`, `requested_n_components`, `embedding`, `fit_umap`, `fit_gp` and `delete`; `UMAPModel` has `transform` (or `embed`), `spec` and `delete`; `GPModel` has `predict`, `single_site`, `crossvalidate`, `evaluations`, `training`, `spec` and `delete`.
* **models:** `fit_svd`, `fit_umap` and `fit_gp` on a protein language model handle fit with that model.
* **models:** `GPModel.crossvalidate(n_splits=...)` or `crossvalidate(folds=...)` with integer folds returns a `CrossvalidateFuture`, whose `get()` and `wait()` return `CrossvalidateResults`.
* **predictor:** `kernel`, `mean` and `noise` read a trained GP's hyperparameters on `session.predictor` predictors and on `session.models` GPs. `kernel` is typed by kernel: `RBFKernel`, `PeriodicKernel`, `RationalQuadraticKernel` and the rest. An ensemble, or a GP trained before hyperparameters were recorded, raises `HTTPError` 404.
* **jobs:** a `session.models` future reads its outputs with `outputs`, `get_item`, `stream`, `units` and `unit(i)`, which raise `APIError` on jobs that store no outputs. Every future has `args` and `dataset`. `units` and `unit(i)` return `FutureUnit` handles, whose `status` is a `JobStatus`. The typed result futures add `len()`, iteration and indexing.
* **jobs:** every future has `name`, `description` and `rename(...)`, except that `openprotein.prompt.Prompt` and `PredictorModel` keep the name they were created with. `session.jobs.list(name=...)` matches a name exactly and `session.jobs.update` renames a job by id.
* **jobs:** a job type the client does not know returns an `UnknownJobFuture` instead of raising. Its `status` and `wait_until_done` work, and `get()` and `wait()` raise `NotImplementedError`.
* **jobs:** `wait(...)` passes keyword arguments on to `get`. `wait(timeout=...)` caps each poll and sleep by the time left and raises `TimeoutException`; a stalled poll is retried first, so a wait can run past its timeout.
* **jobs:** `wait()` on a canceled `session.models` job raises `JobFailedException`.
* **jobs:** `JobStatus.CREATED`, and `openprotein.errors.JobOutputsNotReady` for an output that is not complete yet.
* **data:** `session.data.MSA`, `session.data.Prompt` and `session.data.Assay` store MSAs, prompts and assays: `create`, `list` across every page up to `limit`, `session.data.MSA(id=...)` to load one, and `update`, `delete` and `refresh` on each.
* **data:** a stored MSA, prompt or assay answers a future's calls: `done`, `wait_until_done`, `get`, `wait`, `job`, `job_type`, `created_date`, `start_date` and `end_date`. `wait()` returns it once it is ready, and `get()` returns it if it is ready and raises `JobOutputsNotReady` if not. One that no job made is always ready, and its job fields raise `openprotein.errors.NoJobError`.
* **data:** iterating a stored MSA yields its `(name, sequence)` rows with insertions left out. An a2m or Stockholm upload whose rows don't all mark the same insert columns by lowercase and `.` keeps every column. `to_string(format=None)` renders it, by default in its stored format.
* **data:** `MSA.sample_prompt(...)` samples a stored MSA and returns the stored `Prompt`, which iterates as one list of sequences per replicate. `session.jobs.get` loads that sampling job as the same `Prompt`.
* **data:** an MSA, prompt or assay whose job has not finished reads back with that job's `status` and `job_id`.
* **data:** assay text metadata: `session.data.Assay.create(..., metadata_columns=[...])`, `Assay.metadata_names` and `get_slice(include_metadata=True)`.
* **data:** `AssayDataset` is now `Assay`, and the old name stays.
* **embeddings:** `single_site` on the ESM models (`ESMModel`) and on AbLang2.
* **fold:** OpenDDE: `OpenDDEModel` and `OpenDDEAbagModel` (`opendde`, `opendde-abag`), with `OpenDDEConfidence`, and a `FoldConfidence` schema for fold confidence files.
* **fold:** `fold(seed=...)` on Protenix, Boltz, OpenDDE and RosettaFold-3.
* **fold:** `fold` takes a single complex, protein or sequence as well as a list.
* **align:** `session.align.create_msa(pairing=...)` sets cross-chain pairing for a complex seed, as an `openprotein.align.schemas.Pairing`.
* **codecs:** `openprotein.codecs` decodes stored result files: `msa-parquet-v1`, `ndarray-v1`, `prompt-npz-v1`, JSON, mmCIF and FASTA text. `msa-parquet-v1` needs `pyarrow`, installed with the `msa` extra.
* **base:** deprecation warnings sent by the API are shown as `FutureWarning`s.


### Bug Fixes

* **base:** every unsuccessful response raises `HTTPError`. v0.17.0 could return some 4xx responses to the caller.
* **base:** requests to an `http://` backend are retried like `https://` ones.
* **data:** `session.data.list()` returns every assay. v0.17.0 returned the first 100.
* **data:** an assay's `update` sends JSON, which the API accepts. v0.17.0 sent form data and got a 400.
* **data:** creating an assay keeps its stored sequence length, which is null for sequences of mixed lengths, and leaves the table's column names as they were.
* **fold:** a template mapped to chain `A` is accepted on a lone `Protein`.
* **fold:** a `:`-delimited string folds as one multichain complex on models that take templates.
* **fold:** a batch whose later complexes have a chain the first lacks folds, and a shared template is checked against every complex before anything is uploaded.
* **prompt:** a hyphenated id in any letter case is used as an id.
* **prompt:** `get_query` reads a query that was uploaded as a PDB file.
* **design:** design constraints leave out chain-break positions, so a multichain constraint submits.
* **molecules:** `Complex.to_string("cif")` numbers its model from 1. v0.17.0 wrote model 0.
* **molecules:** selenomethionine `SE` atoms are read as `SD` instead of being dropped.
* **molecules:** `rmsd` returns the exact value, without an epsilon under the root.

## 0.17.0 (2026-08-22)


### ⚠ BREAKING CHANGES

* **molecules:** Complex no longer sorts chains by ID. Chains iterate and serialize in insertion order, so code relying on get_chains() returning alphabetical order must sort explicitly. Combining with & or | appends the new chain last rather than sorting it into place, and both now return a new Complex instead of mutating the left operand. Complex is also no longer hashable, matching its mutable-mapping semantics. Structures parsed from CIF or PDB are unaffected.

### Features

* expose unnumbered_policy and job warnings on alignments
* **prompt:** curated antibody prompts (VH, VL, VH-VL, VL-VH) as attributes on the prompt API
* **molecules:** dict-like Complex interface with insertion-ordered chains


## 0.16.2 (2026-07-20)


### Bug Fixes

* export single site future

## 0.16.1 (2026-07-14)

## 0.16.0 (2026-07-14)


### Features

* expose GP kernel selection in the client
* add solublempnn model


### Bug Fixes

* **molecules:** treat B-factors &gt;100 as experimental in pLDDT heuristic
* split ':'-delimited sequences into one multichain complex
