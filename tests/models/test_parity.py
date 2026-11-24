"""Each legacy submodule call has a ``session.models`` call submitting the same work.

One case per call in the parity inventory. A submission case names the card
method and the request body, with the card's param names and defaults written
out; a read case names the route it reads. Cards and params schemas come from
``tests/data/models_params.json``, and ``ModelsSession`` checks every body
against its method's schema.

Each reader family decodes a stored value of the shape the results routes
serve.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from jsonschema import ValidationError

from openprotein.data.assaydataset import AssayDataset
from openprotein.jobs import Job
from openprotein.jobs.futures import (
    AttentionFuture,
    EmbeddingFuture,
    Future,
    IndelScoreFuture,
    LogitsFuture,
    LogLikelihoodFuture,
    PredictionFuture,
    SequenceGenerationFuture,
    SingleSitePredictionFuture,
    SingleSiteScoreFuture,
    StructureDesignFuture,
    StructurePredictionFuture,
    SVDFitFuture,
    UMAPFitFuture,
)
from openprotein.models import (
    GPModel,
    ModelsAPI,
)
from openprotein.models.schemas import MethodDescriptor
from openprotein.molecules import Complex, Ligand, Protein, Structure
from tests.models.support import SUBMITTED_JOB, Stored, check_body

Q = "c0ffee00-0000-4000-8000-000000000001"
Q2 = "c0ffee00-0000-4000-8000-000000000002"
SEQ = ["MKT"]
PROTEIN = {"protein": {"id": "A", "sequence": "MKT"}}
PROTEIN_MSA = {"protein": {"id": "A", "sequence": "MKT", "msa_id": None}}
LIGAND = {"ligand": {"id": "L", "ccd": "ATP"}}


@dataclass
class Case:
    """A legacy call and its replacement.

    ``posts`` are the expected ``(path, body)`` submissions, in order;
    ``reads`` the ``(verb, path)`` other requests that must be made.
    """

    call: Callable[[ModelsAPI, Any], Any]
    posts: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    reads: list[tuple[str, str]] = field(default_factory=list)


def _ligand_complex() -> Complex:
    complex = Complex({"A": Protein(sequence="MKT"), "L": Ligand(ccd="ATP")})
    complex.get_protein("A").msa = Protein.single_sequence_mode
    return complex


def _assay() -> AssayDataset:
    rows = pd.DataFrame({"sequence": ["MKT", "AC", "DE", "FG", "HI"]})
    assay = MagicMock(spec=AssayDataset, id="assay-1", page_size=2)
    assay.get_slice.side_effect = lambda start, end: rows.iloc[start:end]
    return assay


DESIGN_ID = "d0000000-0000-4000-8000-000000000001"


def _design(session) -> Future:
    job = Job.model_validate(
        {
            "job_id": DESIGN_ID,
            "job_type": "/models",
            "status": "SUCCESS",
            "created_date": "2026-10-21T08:00:00Z",
        }
    )
    future = Future(session, job)
    future._resolved_descriptor = MethodDescriptor(
        path="generate/structures", outputs=["structures"]
    )
    return future


def _fit_future(session, job_id: str) -> Future:
    job = Job.model_validate(
        {
            "job_id": job_id,
            "job_type": "/models",
            "status": "SUCCESS",
            "created_date": "2026-10-21T08:00:00Z",
        }
    )
    return Future(session, job)


def _gp(models: ModelsAPI) -> GPModel:
    predictor = models.get_model("gp/p-1")
    assert isinstance(predictor, GPModel)
    return predictor


def _embedded(model: str, reduction: Any = "MEAN", **params) -> tuple[str, dict]:
    return (
        f"v1/models/{model}/batch/embeddings",
        {"sequences": SEQ, "reduction": reduction, **params},
    )


CASES: dict[str, Case] = {
    # session.embedding
    "embedding.list_models": Case(
        lambda m, s: m.list_models(verbose=True), reads=[("GET", "v1/models")]
    ),
    "embedding.get_model.metadata": Case(
        lambda m, s: m.get_model("poet-2").metadata,
        reads=[("GET", "v1/models/poet-2")],
    ),
    "embedding.prot_seq.embed": Case(
        lambda m, s: m.prot_seq.embed(SEQ), posts=[_embedded("prot-seq")]
    ),
    "embedding.rotaprot.embed(reduction=None)": Case(
        lambda m, s: m.rotaprot_large_uniref50w.embed(SEQ, reduction=None),
        posts=[_embedded("rotaprot-large-uniref50w", None)],
    ),
    "embedding.ablang2.embed": Case(
        lambda m, s: m.ablang2.embed(SEQ), posts=[_embedded("ablang2")]
    ),
    "embedding.esm2.embed": Case(
        lambda m, s: m.esm2.embed(SEQ), posts=[_embedded("esm2_t33_650M_UR50D")]
    ),
    "embedding.poet.embed(prompt)": Case(
        lambda m, s: m.poet.embed(SEQ, prompt="prompt-1"),
        posts=[_embedded("poet", prompt_id="prompt-1")],
    ),
    "embedding.poet2.embed(prompt, query, decoder)": Case(
        lambda m, s: m.poet2.embed(
            SEQ,
            prompt="prompt-1",
            query=Q,
            use_query_structure_in_decoder=False,
            decoder_type="mlm",
            reduction=None,
        ),
        posts=[
            _embedded(
                "poet-2",
                None,
                prompt_id="prompt-1",
                query_id=Q,
                use_query_structure_in_decoder=False,
                decoder_type="mlm",
            )
        ],
    ),
    "embedding.prot_seq.logits": Case(
        lambda m, s: m.prot_seq.logits(SEQ),
        posts=[("v1/models/prot-seq/batch/logits", {"sequences": SEQ})],
    ),
    "embedding.poet2.logits(prompt, query, decoder)": Case(
        lambda m, s: m.poet2.logits(
            SEQ, prompt="prompt-1", query=Q, decoder_type="clm"
        ),
        posts=[
            (
                "v1/models/poet-2/batch/logits",
                {
                    "sequences": SEQ,
                    "prompt_id": "prompt-1",
                    "query_id": Q,
                    "decoder_type": "clm",
                },
            )
        ],
    ),
    "embedding.prot_seq.attn": Case(
        lambda m, s: m.prot_seq.attn(SEQ),
        posts=[("v1/models/prot-seq/batch/attn", {"sequences": SEQ})],
    ),
    "embedding.poet.score(prompt)": Case(
        lambda m, s: m.poet.score(SEQ, prompt="prompt-1"),
        posts=[
            (
                "v1/models/poet/batch/loglikelihood",
                {"sequences": SEQ, "prompt_id": "prompt-1"},
            )
        ],
    ),
    "embedding.poet2.score(prompt, query, decoder)": Case(
        lambda m, s: m.poet2.score(
            SEQ,
            prompt="prompt-1",
            query=Q,
            use_query_structure_in_decoder=True,
            decoder_type="clm",
        ),
        posts=[
            (
                "v1/models/poet-2/batch/loglikelihood",
                {
                    "sequences": SEQ,
                    "prompt_id": "prompt-1",
                    "query_id": Q,
                    "use_query_structure_in_decoder": True,
                    "decoder_type": "clm",
                },
            )
        ],
    ),
    "embedding.poet.indel(insert)": Case(
        lambda m, s: m.poet.indel("MKT", insert="A", prompt="prompt-1"),
        posts=[
            (
                "v1/models/poet/indel/scores",
                {"base_sequence": "MKT", "insert": "A", "prompt_id": "prompt-1"},
            )
        ],
    ),
    "embedding.poet2.indel(delete, decoder)": Case(
        lambda m, s: m.poet2.indel(
            "MKT", delete=[1, 2], prompt="prompt-1", decoder_type="clm"
        ),
        posts=[
            (
                "v1/models/poet-2/indel/scores",
                {
                    "base_sequence": "MKT",
                    "delete": [1, 2],
                    "prompt_id": "prompt-1",
                    "decoder_type": "clm",
                },
            )
        ],
    ),
    "embedding.poet.single_site": Case(
        lambda m, s: m.poet.single_site("MKT", prompt="prompt-1"),
        posts=[
            (
                "v1/models/poet/single-site/scores",
                {"base_sequence": "MKT", "prompt_id": "prompt-1"},
            )
        ],
    ),
    "embedding.poet2.single_site(query, decoder)": Case(
        lambda m, s: m.poet2.single_site(
            "MKT",
            prompt="prompt-1",
            query=Q,
            use_query_structure_in_decoder=True,
            decoder_type="clm",
        ),
        posts=[
            (
                "v1/models/poet-2/single-site/scores",
                {
                    "base_sequence": "MKT",
                    "prompt_id": "prompt-1",
                    "query_id": Q,
                    "use_query_structure_in_decoder": True,
                    "decoder_type": "clm",
                },
            )
        ],
    ),
    "embedding.esmc.single_site": Case(
        lambda m, s: m.esmc_300m.single_site("MKT"),
        posts=[("v1/models/esmc-300m/single-site/scores", {"base_sequence": "MKT"})],
    ),
    "embedding.poet.generate": Case(
        lambda m, s: m.poet.generate(
            prompt="prompt-1",
            num_samples=100,
            temperature=1.0,
            top_k=20,
            top_p=1.0,
            max_length=1000,
            random_seed=7,
        ),
        posts=[
            (
                "v1/models/poet/generate/sequences",
                {
                    "prompt_id": "prompt-1",
                    "num_samples": 100,
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 1.0,
                    "max_length": 1000,
                    "random_seed": 7,
                },
            )
        ],
    ),
    "embedding.poet2.generate(query list, ensemble)": Case(
        lambda m, s: m.poet2.generate(
            prompt="prompt-1",
            query=[Q, Q2],
            use_query_structure_in_decoder=True,
            ensemble_weights=[1.0, 2.0],
            ensemble_method="arithmetic",
            num_samples=10,
            random_seed=7,
        ),
        posts=[
            (
                "v1/models/poet-2/generate/sequences",
                {
                    "prompt_id": "prompt-1",
                    "query_id": [Q, Q2],
                    "use_query_structure_in_decoder": True,
                    "ensemble_weights": [1.0, 2.0],
                    "ensemble_method": "arithmetic",
                    "num_samples": 10,
                    "random_seed": 7,
                },
            )
        ],
    ),
    "embedding.poet2.generate(design)": Case(
        lambda m, s: m.poet2.generate(design=_design(s), random_seed=7),
        posts=[
            (
                "v1/models/poet-2/generate/sequences",
                {"design_id": DESIGN_ID, "random_seed": 7},
            )
        ],
    ),
    "embedding.poet.fit_svd(prompt)": Case(
        lambda m, s: m.poet.fit_svd(SEQ, prompt="prompt-1", reduction="MEAN"),
        posts=[
            (
                "v1/models/svd",
                {
                    "output": "embeddings",
                    "sequences": SEQ,
                    "model_id": "poet",
                    "prompt_id": "prompt-1",
                    "reduction": "MEAN",
                },
            )
        ],
    ),
    "embedding.poet2.fit_svd(assay, query, decoder)": Case(
        lambda m, s: m.poet2.fit_svd(
            assay=_assay(),
            n_components=2,
            prompt="prompt-1",
            query=Q,
            use_query_structure_in_decoder=False,
            decoder_type="mlm",
        ),
        posts=[
            (
                "v1/models/svd",
                {
                    "n_components": 2,
                    "output": "embeddings",
                    "assay_id": "assay-1",
                    "model_id": "poet-2",
                    "prompt_id": "prompt-1",
                    "query_id": Q,
                    "use_query_structure_in_decoder": False,
                    "decoder_type": "mlm",
                },
            )
        ],
    ),
    "embedding.esm2.fit_umap": Case(
        lambda m, s: m.esm2.fit_umap(assay=_assay()),
        posts=[
            (
                "v1/models/umap",
                {
                    "model_id": "esm2_t33_650M_UR50D",
                    "sequences": ["MKT", "AC", "DE", "FG", "HI"],
                    "reduction": "MEAN",
                },
            )
        ],
    ),
    "embedding.poet2.fit_gp(prompt, query)": Case(
        lambda m, s: m.poet2.fit_gp(
            "assay-1",
            ["fitness"],
            reduction="MEAN",
            kernel="rational_quadratic",
            alpha=0.5,
            prompt="prompt-1",
            query=Q,
        ),
        posts=[
            (
                "v1/models/gp",
                {
                    "features": {
                        "model_id": "poet-2",
                        "reduction": "MEAN",
                        "prompt_id": "prompt-1",
                        "query_id": Q,
                    },
                    "kernel": {"type": "rational_quadratic", "alpha": 0.5},
                    "training_dataset": {
                        "assay_id": "assay-1",
                        "properties": ["fitness"],
                    },
                },
            )
        ],
    ),
    # session.fold
    "fold.list_models": Case(
        lambda m, s: m.list_models(path=["structures"]),
        reads=[("GET", "v1/models")],
    ),
    "fold.get_model.metadata": Case(
        lambda m, s: m.get_model("boltz-2").metadata,
        reads=[("GET", "v1/models/boltz-2")],
    ),
    "fold.esmfold.fold": Case(
        lambda m, s: m.esmfold.fold("MKT", num_recycles=4),
        posts=[
            (
                "v1/models/esmfold/batch/structures",
                {"complexes": [[PROTEIN]], "num_recycles": 4},
            )
        ],
    ),
    "fold.esmfold2.fold": Case(
        lambda m, s: m.esmfold2.fold(
            "MKT", num_samples=1, num_recycles=3, num_steps=100, random_seed=7
        ),
        posts=[
            (
                "v1/models/esmfold2/batch/structures",
                {
                    "complexes": [[PROTEIN_MSA]],
                    "num_samples": 1,
                    "num_recycles": 3,
                    "num_steps": 100,
                    "random_seed": 7,
                },
            )
        ],
    ),
    "fold.esmfold2_fast.fold": Case(
        lambda m, s: m.esmfold2_fast.fold(
            "MKT", num_samples=1, num_recycles=3, num_steps=100, random_seed=7
        ),
        posts=[
            (
                "v1/models/esmfold2-fast/batch/structures",
                {
                    "complexes": [[PROTEIN]],
                    "num_samples": 1,
                    "num_recycles": 3,
                    "num_steps": 100,
                    "random_seed": 7,
                },
            )
        ],
    ),
    "fold.minifold.fold": Case(
        lambda m, s: m.minifold.fold("MKT", num_recycles=4),
        posts=[
            (
                "v1/models/minifold/batch/structures",
                {"complexes": [[PROTEIN]], "num_recycles": 4},
            )
        ],
    ),
    "fold.alphafold2.fold": Case(
        lambda m, s: m.af2.fold("MKT", num_models=1, num_recycles=None, num_relax=0),
        posts=[
            (
                "v1/models/alphafold2/batch/structures",
                {
                    "complexes": [[PROTEIN_MSA]],
                    "num_models": 1,
                    "num_recycles": None,
                    "num_relax": 0,
                },
            )
        ],
    ),
    "fold.boltz_1.fold(use_potentials)": Case(
        lambda m, s: m.boltz1.fold(
            "MKT",
            num_samples=1,
            num_recycles=3,
            num_steps=200,
            step_scale=1.638,
            use_potentials=True,
            random_seed=7,
        ),
        posts=[
            (
                "v1/models/boltz-1/batch/structures",
                {
                    "complexes": [[PROTEIN_MSA]],
                    "num_samples": 1,
                    "num_recycles": 3,
                    "num_steps": 200,
                    "step_scale": 1.638,
                    "use_potentials": True,
                    "random_seed": 7,
                },
            )
        ],
    ),
    "fold.boltz_1x.fold": Case(
        lambda m, s: m.boltz1x.fold("MKT", num_samples=2, random_seed=7),
        posts=[
            (
                "v1/models/boltz-1x/batch/structures",
                {"complexes": [[PROTEIN_MSA]], "num_samples": 2, "random_seed": 7},
            )
        ],
    ),
    "fold.boltz_2.fold(constraints, properties, method)": Case(
        lambda m, s: m.boltz2.fold(
            _ligand_complex(),
            constraints=[{"bond": {"atom1": ["A", 1, "N"], "atom2": ["L", 1, "C1"]}}],
            properties=[{"affinity": {"binder": "L"}}],
            method="X-RAY DIFFRACTION",
            random_seed=7,
        ),
        posts=[
            (
                "v1/models/boltz-2/batch/structures",
                {
                    "complexes": [[PROTEIN_MSA, LIGAND]],
                    "constraints": [
                        {"bond": {"atom1": ["A", 1, "N"], "atom2": ["L", 1, "C1"]}}
                    ],
                    "properties": [{"affinity": {"binder": "L"}}],
                    "method": "X-RAY DIFFRACTION",
                    "random_seed": 7,
                },
            )
        ],
    ),
    "fold.protenix.fold": Case(
        lambda m, s: m.protenix.fold(
            "MKT", num_samples=1, num_recycles=10, num_steps=200, random_seed=7
        ),
        posts=[
            (
                "v1/models/protenix/batch/structures",
                {
                    "complexes": [[PROTEIN_MSA]],
                    "num_samples": 1,
                    "num_recycles": 10,
                    "num_steps": 200,
                    "random_seed": 7,
                },
            )
        ],
    ),
    "fold.protenix_v2.fold": Case(
        lambda m, s: m.protenix_v2.fold("MKT", random_seed=7),
        posts=[
            (
                "v1/models/protenix-v2/batch/structures",
                {"complexes": [[PROTEIN_MSA]], "random_seed": 7},
            )
        ],
    ),
    "fold.rosettafold_3.fold": Case(
        lambda m, s: m.rf3.fold(
            "MKT", num_samples=1, num_recycles=10, num_steps=50, random_seed=7
        ),
        posts=[
            (
                "v1/models/rosettafold-3/batch/structures",
                {
                    "complexes": [[PROTEIN_MSA]],
                    "num_samples": 1,
                    "num_recycles": 10,
                    "num_steps": 50,
                    "random_seed": 7,
                },
            )
        ],
    ),
    # 0.17.0 session.models handles
    "models.rfdiffusion.generate": Case(
        lambda m, s: m.rfdiffusion.generate(query=Q, num_samples=2, random_seed=7),
        posts=[
            (
                "v1/models/rfdiffusion/generate/structures",
                {"query_id": Q, "num_samples": 2, "random_seed": 7},
            )
        ],
    ),
    "models.proteinmpnn.embed": Case(
        lambda m, s: m.proteinmpnn.embed(SEQ, query=Q),
        posts=[
            (
                "v1/models/proteinmpnn/batch/embeddings",
                {"sequences": SEQ, "reduction": "MEAN", "query_id": Q},
            )
        ],
    ),
    "models.proteinmpnn.score": Case(
        lambda m, s: m.proteinmpnn.score(
            SEQ, query=Q, decoding_orders=1, random_seed=1
        ),
        posts=[
            (
                "v1/models/proteinmpnn/batch/loglikelihood",
                {
                    "sequences": SEQ,
                    "query_id": Q,
                    "decoding_orders": 1,
                    "random_seed": 1,
                },
            )
        ],
    ),
    "models.proteinmpnn.single_site": Case(
        lambda m, s: m.proteinmpnn.single_site("MKT", query=Q, random_seed=1),
        posts=[
            (
                "v1/models/proteinmpnn/single-site/scores",
                {"base_sequence": "MKT", "query_id": Q, "random_seed": 1},
            )
        ],
    ),
    "models.proteinmpnn.generate(design)": Case(
        lambda m, s: m.proteinmpnn.generate(
            design=_design(s),
            num_samples=100,
            temperature=0.1,
            random_seed=7,
        ),
        posts=[
            (
                "v1/models/proteinmpnn/generate/sequences",
                {
                    "design_id": DESIGN_ID,
                    "num_samples": 100,
                    "temperature": 0.1,
                    "random_seed": 7,
                },
            )
        ],
    ),
    "models.solublempnn.generate": Case(
        lambda m, s: m.solublempnn.generate(query=Q, num_samples=10, random_seed=7),
        posts=[
            (
                "v1/models/solublempnn/generate/sequences",
                {"query_id": Q, "num_samples": 10, "random_seed": 7},
            )
        ],
    ),
    "models.esmif1.score": Case(
        lambda m, s: m.esmif1.score(SEQ, query=Q),
        posts=[
            (
                "v1/models/esm-if1/batch/loglikelihood",
                {"sequences": SEQ, "query_id": Q},
            )
        ],
    ),
    "models.esmif1.single_site": Case(
        lambda m, s: m.esmif1.single_site("MKT", query=Q),
        posts=[
            (
                "v1/models/esm-if1/single-site/scores",
                {"base_sequence": "MKT", "query_id": Q},
            )
        ],
    ),
    "models.esmif1.generate": Case(
        lambda m, s: m.esmif1.generate(
            query=Q, num_samples=100, temperature=1.0, random_seed=7
        ),
        posts=[
            (
                "v1/models/esm-if1/generate/sequences",
                {
                    "query_id": Q,
                    "num_samples": 100,
                    "temperature": 1.0,
                    "random_seed": 7,
                },
            )
        ],
    ),
    "models.boltzgen.generate": Case(
        lambda m, s: m.boltzgen.generate(
            query=Q,
            num_samples=1,
            diffusion_batch_size=2,
            step_scale=1.0,
            noise_scale=1.0,
            random_seed=7,
        ),
        posts=[
            (
                "v1/models/boltzgen/generate/structures",
                {
                    "query_id": Q,
                    "num_samples": 1,
                    "diffusion_batch_size": 2,
                    "step_scale": 1.0,
                    "noise_scale": 1.0,
                    "random_seed": 7,
                },
            )
        ],
    ),
    # session.svd
    "svd.fit_svd(assay)": Case(
        lambda m, s: m.prot_seq.fit_svd(
            assay=_assay(), n_components=2, reduction="MEAN"
        ),
        posts=[
            (
                "v1/models/svd",
                {
                    "n_components": 2,
                    "output": "embeddings",
                    "assay_id": "assay-1",
                    "model_id": "prot-seq",
                    "reduction": "MEAN",
                },
            )
        ],
    ),
    "svd.get_svd": Case(
        lambda m, s: m.SVD(id="s-1"), reads=[("GET", "v1/models/svd/s-1")]
    ),
    "svd.list_svd(limit, offset)": Case(
        lambda m, s: m.SVD.list(limit=10, offset=5),
        reads=[("GET", "v1/models")],
    ),
    "svd.delete_svd": Case(
        lambda m, s: m.get_model("svd/s-1").delete(),
        reads=[("DELETE", "v1/models/svd/s-1")],
    ),
    "svd.SVDModel.embed": Case(
        lambda m, s: m.get_model("svd/s-1").embed(SEQ),
        posts=[("v1/models/svd/s-1/batch/embeddings", {"sequences": SEQ})],
    ),
    "svd.SVDModel.get_inputs": Case(
        lambda m, s: _fit_future(s, "s-1").get(output="sequences"),
        reads=[("OUTPUT", "s-1/sequences")],
    ),
    "svd.SVDModel.fit_umap": Case(
        lambda m, s: m.get_model("svd/s-1").fit_umap(
            sequences=["MKT", "AC", "DE", "FG", "HI"]
        ),
        posts=[
            (
                "v1/models/umap",
                {
                    "model_id": "svd/s-1",
                    "sequences": ["MKT", "AC", "DE", "FG", "HI"],
                },
            )
        ],
    ),
    "svd.SVDModel.fit_gp": Case(
        lambda m, s: m.get_model("svd/s-1").fit_gp("assay-1", ["fitness"]),
        posts=[
            (
                "v1/models/gp",
                {
                    "features": {"model_id": "svd/s-1"},
                    "kernel": {"type": "rbf"},
                    "training_dataset": {
                        "assay_id": "assay-1",
                        "properties": ["fitness"],
                    },
                },
            )
        ],
    ),
    # session.umap
    "umap.fit_umap(PLM, assay)": Case(
        lambda m, s: m.esm2_t33_650M_UR50D.fit_umap(
            assay=_assay(),
            reduction="MEAN",
            n_components=3,
            n_neighbors=10,
            min_dist=0.2,
        ),
        posts=[
            (
                "v1/models/umap",
                {
                    "model_id": "esm2_t33_650M_UR50D",
                    "sequences": ["MKT", "AC", "DE", "FG", "HI"],
                    "reduction": "MEAN",
                    "n_components": 3,
                    "n_neighbors": 10,
                    "min_dist": 0.2,
                },
            )
        ],
    ),
    "umap.get_umap": Case(
        lambda m, s: m.UMAP(id="u-1"), reads=[("GET", "v1/models/umap/u-1")]
    ),
    "umap.list_umap(limit, offset)": Case(
        lambda m, s: m.UMAP.list(limit=10, offset=5),
        reads=[("GET", "v1/models")],
    ),
    "umap.delete_umap": Case(
        lambda m, s: m.get_model("umap/u-1").delete(),
        reads=[("DELETE", "v1/models/umap/u-1")],
    ),
    "umap.UMAPModel.embed": Case(
        lambda m, s: m.get_model("umap/u-1").embed(SEQ),
        posts=[("v1/models/umap/u-1/batch/embeddings", {"sequences": SEQ})],
    ),
    "umap.UMAPModel.sequences": Case(
        lambda m, s: _fit_future(s, "u-1").get(output="sequences"),
        reads=[("OUTPUT", "u-1/sequences")],
    ),
    "umap.UMAPModel.embeddings": Case(
        lambda m, s: _fit_future(s, "u-1").get(output="embeddings"),
        reads=[("OUTPUT", "u-1/embeddings")],
    ),
    # session.predictor
    "predictor.fit_gp": Case(
        lambda m, s: m.esm2_t33_650M_UR50D.fit_gp(
            "assay-1",
            ["fitness"],
            reduction="MEAN",
            kernel="periodic",
            period=3.0,
            name="gp",
            description="d",
        ),
        posts=[
            (
                "v1/models/gp",
                {
                    "features": {
                        "model_id": "esm2_t33_650M_UR50D",
                        "reduction": "MEAN",
                    },
                    "kernel": {"type": "periodic", "period": 3.0},
                    "training_dataset": {
                        "assay_id": "assay-1",
                        "properties": ["fitness"],
                    },
                    "name": "gp",
                    "description": "d",
                },
            )
        ],
    ),
    "predictor.get_predictor": Case(
        lambda m, s: m.GP(id="p-1"), reads=[("GET", "v1/models/gp/p-1")]
    ),
    "predictor.list_predictors(limit, offset)": Case(
        lambda m, s: m.GP.list(limit=10, offset=5),
        reads=[("GET", "v1/models")],
    ),
    "predictor.delete_predictor": Case(
        lambda m, s: m.get_model("gp/p-1").delete(),
        reads=[("DELETE", "v1/models/gp/p-1")],
    ),
    "predictor.PredictorModel.predict": Case(
        lambda m, s: _gp(m).predict(SEQ),
        posts=[("v1/models/gp/p-1/batch/predictions", {"sequences": SEQ})],
    ),
    "predictor.PredictorModel.single_site": Case(
        lambda m, s: _gp(m).single_site("MKT"),
        posts=[("v1/models/gp/p-1/single-site/predictions", {"base_sequence": "MKT"})],
    ),
    "predictor.PredictorModel.crossvalidate": Case(
        lambda m, s: _gp(m).crossvalidate(n_splits=5),
        posts=[("v1/models/gp/p-1/crossvalidate", {"n_splits": 5})],
    ),
    "predictor.CVResultFuture.get": Case(
        lambda m, s: _gp(m).evaluations("cv-1"),
        reads=[("GET", "v1/models/gp/p-1/evaluations/cv-1")],
    ),
    "predictor.PredictorModel.kernel": Case(
        lambda m, s: _gp(m).kernel,
        reads=[("GET", "v1/models/gp/p-1/hyperparameters")],
    ),
    "predictor.PredictorModel.traingraphs": Case(
        lambda m, s: _gp(m).training(),
        reads=[("GET", "v1/models/gp/p-1/training")],
    ),
    "predictor.get_predictor(include_stats)": Case(
        lambda m, s: _gp(m).evaluations(),
        reads=[("GET", "v1/models/gp/p-1/evaluations")],
    ),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_the_replacement_submits_the_same_work(models_session, model_card, case):
    for family in ("svd", "umap", "gp"):
        models_session.get_routes[f"v1/models/{family}"] = []
    models_session.get_routes |= {
        "v1/models": [],
        "v1/models/svd/s-1": model_card("svd/s-1") | {"svd": {"id": "s-1"}},
        "v1/models/umap/u-1": model_card("umap/u-1"),
        "v1/models/gp/p-1": model_card("gp/p-1")
        | {"spec": {"kernel": {"type": "RBF"}}},
        "v1/models/gp/p-1/hyperparameters": {
            "fitness": {"lengthscale": [0.8], "outputscale": [2.0]}
        },
    }
    models = ModelsAPI(models_session.session)
    with patch("openprotein.jobs.jobs.JobsAPI") as jobs:
        jobs.return_value._iter_job_outputs.return_value = iter([])
        case.call(models, models_session.session)
        outputs = [
            ("OUTPUT", f"{call.args[0]}/{call.args[1]}")
            for call in jobs.return_value._iter_job_outputs.call_args_list
        ]

    assert models_session.submissions == case.posts
    session = models_session.session
    made = [
        *(("GET", call.args[0]) for call in session.get.call_args_list),
        *(("DELETE", call.args[0]) for call in session.delete.call_args_list),
        *outputs,
    ]
    for read in case.reads:
        assert read in made


@pytest.mark.parametrize(
    "path, body",
    [
        (
            "v1/models/poet-2/batch/loglikelihood",
            {"sequences": SEQ, "decoder_type": "mlm"},
        ),
        ("v1/models/esm2_t6_8M_UR50D/batch/embeddings", {"sequences": SEQ, "x": 1}),
        ("v1/models/poet/batch/logits", {"sequences": SEQ, "reduction": "MEAN"}),
        ("v1/models/esm-if1/generate/sequences", {"query_id": "not-a-uuid"}),
        ("v1/models/umap", {"model_id": "esm2", "sequences": ["M", "K", "T"]}),
        ("v1/models/gp/p-1/crossvalidate", {"n_splits": 1}),
        ("v1/models/prot-seq/single-site/scores", {"base_sequence": "MKT"}),
    ],
)
def test_the_contract_check_refuses_a_body_main_refuses(path, body):
    with pytest.raises((AssertionError, ValidationError)):
        check_body(path, body)


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.prot_seq.single_site("MKT"),
        lambda m: m.esm1b.single_site("MKT"),
        lambda m: m.ablang2.attn(SEQ),
        lambda m: m.proteinmpnn.indel("MKT", insert="A"),
        lambda m: m.esmfold.generate(),
        lambda m: m.get_model("svd/s-1").predict(SEQ),
    ],
)
def test_a_verb_on_a_method_the_card_lacks_submits_nothing(models_session, call):
    with pytest.raises(ValueError, match="has no"):
        call(ModelsAPI(models_session.session))
    assert models_session.submissions == []


CIF = (Path(__file__).parents[1] / "data" / "test_protein.cif").read_text()


@dataclass
class Read:
    """A submission, the outputs its job stores, and a check of each decoded value."""

    call: Callable[[ModelsAPI], Future]
    future: type[Future]
    stored: dict[str, list[Any]]
    check: dict[str, Callable[[Any], Any]]


def _array(*shape: int) -> Stored:
    return Stored(
        "ndarray-v1",
        np.arange(1, np.empty(shape).size + 1, dtype=np.float32).reshape(shape),
    )


def _is_array(*shape: int) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, np.ndarray) and v.shape == shape and v.any()


READS: dict[str, Read] = {
    "embeddings": Read(
        lambda m: m.prot_seq.embed(SEQ, reduction=None),
        EmbeddingFuture,
        {"embeddings": [_array(3, 4)]},
        {"embeddings": _is_array(3, 4)},
    ),
    "logits": Read(
        lambda m: m.esm2.logits(SEQ),
        LogitsFuture,
        {"logits": [_array(3, 20)]},
        {"logits": _is_array(3, 20)},
    ),
    "attn": Read(
        lambda m: m.prot_seq.attn(SEQ),
        AttentionFuture,
        {"attn": [_array(2, 3, 3)]},
        {"attn": _is_array(2, 3, 3)},
    ),
    "fold": Read(
        lambda m: m.boltz2.fold("MKT", random_seed=1),
        StructurePredictionFuture,
        {
            "structures": [Stored("mmcif-utf8-v1", CIF)],
            "plddt": [_array(1, 3)],
            "pae": [_array(1, 3, 3)],
            "ipae": [_array(1)],
            "confidence": [Stored("json-utf8-v1", [{"ptm": 0.8, "iptm": 0.7}])],
            "affinity": [Stored("json-utf8-v1", {"affinity_pred_value": 1.2})],
        },
        {
            "structures": lambda value: (
                isinstance(value, Structure)
                and len(value) > 0
                and value[0].get_chains()
            ),
            "plddt": _is_array(1, 3),
            "pae": _is_array(1, 3, 3),
            "ipae": _is_array(1),
            "confidence": lambda v: v == [{"ptm": 0.8, "iptm": 0.7}],
            "affinity": lambda v: v == {"affinity_pred_value": 1.2},
        },
    ),
    "rosettafold-3 score": Read(
        lambda m: m.rf3.fold("MKT", random_seed=1),
        StructurePredictionFuture,
        {"score": [Stored(None, b"ptm,iptm\n0.8,0.7\n")]},
        {"score": lambda v: v.equals(pd.DataFrame({"ptm": [0.8], "iptm": [0.7]}))},
    ),
    "loglikelihood": Read(
        lambda m: m.poet.score(SEQ, prompt="prompt-1"),
        LogLikelihoodFuture,
        {"loglikelihood": [-12.5]},
        {"loglikelihood": lambda v: v == -12.5},
    ),
    "replicate loglikelihood": Read(
        lambda m: m.poet2.score(SEQ, prompt="prompt-1"),
        LogLikelihoodFuture,
        {"loglikelihood": [[-12.5, -11.0]]},
        {"loglikelihood": lambda v: v == [-12.5, -11.0]},
    ),
    "single-site scores": Read(
        lambda m: m.esmc_300m.single_site("MK"),
        SingleSiteScoreFuture,
        {"score": [0.1 * i for i in range(40)]},
        {"score": lambda v: isinstance(v, float)},
    ),
    "indel scores": Read(
        lambda m: m.poet.indel("MKT", delete=[1], prompt="prompt-1"),
        IndelScoreFuture,
        {"score": [-3.0, -2.0, -1.0, -4.0]},
        {"score": lambda v: isinstance(v, float)},
    ),
    "generate": Read(
        lambda m: m.proteinmpnn.generate(query=Q, num_samples=2, random_seed=1),
        SequenceGenerationFuture,
        {"sequences": ["MKT", "MRT"], "loglikelihood": [-3.5, -4.0]},
        {
            "sequences": lambda v: isinstance(v, str) and v,
            "loglikelihood": lambda v: isinstance(v, float),
        },
    ),
    "generate structures": Read(
        lambda m: m.rfdiffusion.generate(query=Q, random_seed=1),
        StructureDesignFuture,
        {"structures": [Stored("mmcif-utf8-v1", CIF)]},
        {"structures": lambda value: isinstance(value, Complex) and value.get_chains()},
    ),
    "gp predictions": Read(
        lambda m: _gp(m).predict(SEQ),
        PredictionFuture,
        {"predictions": [{"fitness": {"mean": 0.5, "variance": 0.1}}]},
        {"predictions": lambda v: v == {"fitness": {"mean": 0.5, "variance": 0.1}}},
    ),
    "svd projection": Read(
        lambda m: m.get_model("svd/s-1").embed(SEQ),
        EmbeddingFuture,
        {"embeddings": [_array(8)]},
        {"embeddings": _is_array(8)},
    ),
    "umap fit set": Read(
        lambda m: m.get_model("svd/s-1").fit_umap(sequences=["MKT", "AC", "DE", "FG"]),
        UMAPFitFuture,
        {
            "sequences": ["MKT", "AC", "DE", "FG"],
            "embeddings": [_array(2), _array(2), _array(2), _array(2)],
        },
        {"sequences": lambda v: isinstance(v, str) and v, "embeddings": _is_array(2)},
    ),
    "svd fit set": Read(
        lambda m: m.esm2.fit_svd(SEQ, n_components=2),
        SVDFitFuture,
        {"sequences": ["MKT"], "embeddings": [_array(2)]},
        {"sequences": lambda v: v == "MKT", "embeddings": _is_array(2)},
    ),
    "gp single-site predictions": Read(
        lambda m: _gp(m).single_site("MK"),
        SingleSitePredictionFuture,
        {"predictions": [{"fitness": {"mean": 0.5, "variance": 0.1}}]},
        {"predictions": lambda v: v == {"fitness": {"mean": 0.5, "variance": 0.1}}},
    ),
}


@pytest.mark.parametrize("read", READS.values(), ids=READS.keys())
def test_each_reader_decodes_its_stored_values(models_session, read):
    """Every output a job stores reads back decoded, one value per position."""
    job_id = SUBMITTED_JOB["job_id"]
    for output, values in read.stored.items():
        models_session.outputs[(job_id, output)] = values

    future = read.call(ModelsAPI(models_session.session))
    assert type(future) is read.future

    for output, check in read.check.items():
        values = future.get(output=output)
        assert len(values) == len(read.stored[output]) > 0
        assert all(check(value) for value in values), output
