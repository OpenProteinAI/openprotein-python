from collections.abc import Iterator
from typing import Any

import numpy as np
from typing_extensions import assert_type

from openprotein.data.prompts import Prompt
from openprotein.jobs import (
    AttentionFuture,
    CrossvalidateFuture,
    CrossvalidateResults,
    EmbeddingFuture,
    Future,
    FutureUnit,
    GPFitFuture,
    IndelScoreFuture,
    JobStatus,
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
    ESMIF1,
    GP,
    SVD,
    UMAP,
    AlphaFlow,
    BioEmu,
    ESMFlow,
    GPModel,
    Model,
    PoET,
    PoET2,
    ProteinLanguageModel,
    ProteinMPNN,
    RFdiffusion,
    StructureGenerativeModel,
    StructurePredictionModel,
    SVDModel,
    UMAPModel,
)
from openprotein.molecules import Complex


def task_result_types(
    embedding: EmbeddingFuture, design: StructureDesignFuture
) -> None:
    assert_type(embedding.get(), list[np.ndarray])
    assert_type(embedding.get_item(0), np.ndarray)
    assert_type(design.get(), list[Complex])
    assert_type(design.wait(), list[Complex])
    assert_type(design[0], Complex)
    assert_type(design.stream(), Iterator[Complex])
    assert_type(design.unit(0), FutureUnit[Complex])
    assert_type(design.units, list[FutureUnit[Complex]])
    assert_type(design.units[0].status, JobStatus)
    assert_type(design.unit(0).get(), Complex)
    assert_type(design.get(output="confidence"), list[Any])


def fitted_result_types(
    gp: GPFitFuture, svd: SVDFitFuture, umap: UMAPFitFuture
) -> None:
    assert_type(gp.get(), GPModel)
    assert_type(gp.wait(), GPModel)
    assert_type(svd.get(), SVDModel)
    assert_type(svd.wait(), SVDModel)
    assert_type(svd.get(output="embeddings"), list[Any])
    assert_type(svd.wait(output="embeddings"), list[Any])
    assert_type(umap.get(), UMAPModel)
    assert_type(umap.wait(), UMAPModel)
    assert_type(svd.transform(["MKT"]), EmbeddingFuture)
    assert_type(umap.embed(["MKT"]), EmbeddingFuture)


def prompt_result_types(prompt: Prompt) -> None:
    assert_type(prompt.wait(), Prompt)
    assert_type(prompt[0], list[str])
    assert_type(list(prompt), list[list[str]])


def crossvalidation_result_types(future: CrossvalidateFuture) -> None:
    assert_type(future.get(), CrossvalidateResults)
    assert_type(future.wait(), CrossvalidateResults)


def submission_types(
    plm: ProteinLanguageModel,
    rfdiffusion: RFdiffusion,
    predictor: StructurePredictionModel,
    gp: GPModel,
) -> None:
    assert_type(plm.embed(["MKT"]), EmbeddingFuture)
    assert_type(rfdiffusion.generate(), StructureDesignFuture)
    assert_type(predictor.fold("MKT"), StructurePredictionFuture)
    assert_type(plm.single_site("MKT"), SingleSiteScoreFuture)
    assert_type(gp.single_site("MKT"), SingleSitePredictionFuture)


def estimator_types(svd: SVD, umap: UMAP, gp: GP) -> None:
    assert_type(svd.fit(), SVDFitFuture)
    assert_type(svd.fit_transform(), SVDFitFuture)
    assert_type(umap.fit(), UMAPFitFuture)
    assert_type(umap.fit_transform(), UMAPFitFuture)
    assert_type(gp.fit(), GPFitFuture)


def generic_model_types(model: Model) -> None:
    assert_type(model.attn(["MKT"]), AttentionFuture)
    assert_type(model.embed(["MKT"]), EmbeddingFuture)
    assert_type(model.fold(["MKT"]), StructurePredictionFuture)
    assert_type(
        model.generate(), SequenceGenerationFuture | StructureDesignFuture | Future
    )
    assert_type(model.indel("MKT"), IndelScoreFuture)
    assert_type(model.logits(["MKT"]), LogitsFuture)
    assert_type(model.predict(["MKT"]), PredictionFuture)
    assert_type(model.score(["MKT"]), LogLikelihoodFuture)
    assert_type(
        model.single_site("MKT"),
        SingleSiteScoreFuture | SingleSitePredictionFuture | Future,
    )


def protein_language_model_types(plm: ProteinLanguageModel) -> None:
    assert_type(plm.fit_gp("assay-id", ["fitness"], reduction="MEAN"), GPFitFuture)
    assert_type(plm.fit_svd(["MKT"]), SVDFitFuture)
    assert_type(plm.fit_umap(["MKT"]), UMAPFitFuture)


def poet_types(poet: PoET) -> None:
    assert_type(poet.embed(["MKT"]), EmbeddingFuture)
    assert_type(poet.fit_gp("assay-id", ["fitness"], reduction="MEAN"), GPFitFuture)
    assert_type(poet.fit_svd(["MKT"]), SVDFitFuture)
    assert_type(poet.fit_umap(["MKT"]), UMAPFitFuture)
    assert_type(poet.generate(), SequenceGenerationFuture)
    assert_type(poet.indel("MKT"), IndelScoreFuture)
    assert_type(poet.logits(["MKT"]), LogitsFuture)
    assert_type(poet.score(["MKT"]), LogLikelihoodFuture)
    assert_type(poet.single_site("MKT"), SingleSiteScoreFuture)


def poet2_types(poet2: PoET2) -> None:
    assert_type(poet2.attn(["MKT"], query="MKT"), AttentionFuture)
    assert_type(poet2.embed(["MKT"], query="MKT"), EmbeddingFuture)
    assert_type(
        poet2.fit_gp("assay-id", ["fitness"], reduction="MEAN", query="MKT"),
        GPFitFuture,
    )
    assert_type(poet2.fit_svd(["MKT"], query="MKT"), SVDFitFuture)
    assert_type(poet2.fit_umap(["MKT"], query="MKT"), UMAPFitFuture)
    assert_type(poet2.generate(query="MKT"), SequenceGenerationFuture)
    assert_type(poet2.logits(["MKT"], query="MKT"), LogitsFuture)
    assert_type(poet2.score(["MKT"], query="MKT"), LogLikelihoodFuture)
    assert_type(poet2.single_site("MKT", query="MKT"), SingleSiteScoreFuture)


def inverse_folding_types(inverse: ESMIF1, proteinmpnn: ProteinMPNN) -> None:
    assert_type(inverse.generate(), SequenceGenerationFuture)
    assert_type(inverse.score(["MKT"]), LogLikelihoodFuture)
    assert_type(inverse.single_site("MKT"), SingleSiteScoreFuture)
    assert_type(proteinmpnn.embed(["MKT"]), EmbeddingFuture)


def structure_model_types(
    bioemu: BioEmu,
    alphaflow: AlphaFlow,
    esmflow: ESMFlow,
    generative: StructureGenerativeModel,
) -> None:
    assert_type(
        bioemu.sample([bioemu.Input("MKT", msa=None)]), StructurePredictionFuture
    )
    assert_type(
        alphaflow.sample([alphaflow.Input("MKT", msa="msa-id")]),
        StructurePredictionFuture,
    )
    assert_type(esmflow.sample([esmflow.Input("MKT")]), StructurePredictionFuture)
    assert_type(generative.generate(), StructureDesignFuture)


def fitted_model_types(gp: GPModel, svd: SVDModel, umap: UMAPModel) -> None:
    assert_type(gp.crossvalidate(), CrossvalidateFuture)
    assert_type(svd.fit_gp("assay-id", ["fitness"]), GPFitFuture)
    assert_type(svd.fit_umap(["MKT"]), UMAPFitFuture)
    assert_type(svd.transform(["MKT"]), EmbeddingFuture)
    assert_type(umap.transform(["MKT"]), EmbeddingFuture)
