"""Family handles convert SDK objects to card params."""

import copy
import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

from openprotein.align.msa import MSAFuture
from openprotein.errors import HTTPError
from openprotein.jobs import Job, JobStatus
from openprotein.jobs.futures import Future
from openprotein.models import (
    ESMIF1,
    BoltzGen,
    ModelsAPI,
    PoET,
    PoET2,
    ProteinMPNN,
    ProtSeq,
    RFdiffusion,
    StructureGenerationFuture,
    StructureGenerationJob,
)
from openprotein.models.schemas import MethodDescriptor
from openprotein.molecules import Complex, Protein
from openprotein.prompt import Query
from openprotein.prompt.schemas import QueryMetadata
from tests.models import support
from tests.models.support import PARAMS, created_query

QUERY_UUID = "c0ffee00-0000-4000-8000-000000000001"


def _query(session, query_id: str = QUERY_UUID) -> Query:
    return Query(
        session,
        QueryMetadata(id=query_id, created_date="2026-10-21T08:00:00Z"),
    )


DESIGN_UUID = "d0000000-0000-4000-8000-000000000001"


def _design(session, status: str = "SUCCESS") -> Future:
    job = Job.model_validate(
        {
            "job_id": DESIGN_UUID,
            "job_type": "/models/rfdiffusion/generate/structures",
            "status": status,
            "created_date": "2026-10-21T08:00:00Z",
        }
    )
    future = Future(session, job)
    future._resolved_descriptor = MethodDescriptor(
        path="generate/structures", outputs=["structures"]
    )
    return future


@pytest.fixture
def models(models_session) -> ModelsAPI:
    return ModelsAPI(models_session.session)


class TestSeeds:
    @pytest.mark.parametrize(
        "call",
        [
            lambda m: m.poet.generate(prompt="prompt-1"),
            lambda m: m.proteinmpnn.generate(query=QUERY_UUID),
            lambda m: m.esmif1.generate(query=QUERY_UUID),
            lambda m: m.rfdiffusion.generate(query_id=QUERY_UUID),
            lambda m: m.boltzgen.generate(query_id=QUERY_UUID),
        ],
        ids=["poet", "proteinmpnn", "esm-if1", "rfdiffusion", "boltzgen"],
    )
    def test_an_unset_random_seed_is_not_sent(self, models, models_session, call):
        call(models)

        assert "random_seed" not in models_session.submitted[1]

    def test_a_given_random_seed_is_kept(self, models, models_session):
        models.poet.generate(prompt="prompt-1", random_seed=7)

        assert models_session.submitted[1]["random_seed"] == 7

    def test_a_425_resubmit_sends_the_same_body_again(
        self, models, models_session, monkeypatch
    ):
        monkeypatch.setattr("openprotein.jobs._submission.time.sleep", lambda _: None)
        pending = MagicMock(status_code=425, headers={"Retry-After": "0"})
        accepted = models_session._post
        answers = iter([HTTPError(pending)])

        def post(path, **kwargs):
            if path != "v1/prompt/query":
                error = next(answers, None)
                if error is not None:
                    accepted(path, **kwargs)
                    raise error
            return accepted(path, **kwargs)

        models_session.session.post.side_effect = post

        models.proteinmpnn.generate(query=QUERY_UUID)

        first, second = models_session.submissions
        assert first == second
        assert "random_seed" not in first[1]

    def test_a_scoring_seed_with_a_card_default_is_left_to_the_card(
        self, models, models_session
    ):
        models.proteinmpnn.score(["MKT"], query=QUERY_UUID)

        assert "random_seed" not in models_session.submitted[1]


class TestPrompt:
    def test_a_prompt_object_sends_its_id(self, models, models_session):
        from openprotein.data.prompts.prompts import Prompt
        from openprotein.data.prompts.schemas import PromptMetadata

        prompt = Prompt(
            models_session.session,
            PromptMetadata(id="prompt-1", created_at="2026-10-21T08:00:00Z"),
        )

        models.poet.score(["MKT"], prompt=prompt)

        assert models_session.submitted == (
            "v1/models/poet/batch/loglikelihood",
            {"sequences": ["MKT"], "prompt_id": "prompt-1"},
        )

    @pytest.mark.parametrize(("model", "verb"), [("poet", "score"), ("poet2", "embed")])
    def test_a_prompt_that_is_not_ready_sends_its_id(
        self, models, models_session, model, verb
    ):
        from openprotein.data.prompts import Prompt, PromptMetadata

        prompt = Prompt(
            models_session.session,
            PromptMetadata(id="prompt-9", status=JobStatus.PENDING, job_id="prompt-9"),
        )

        getattr(getattr(models, model), verb)(["MKT"], prompt=prompt)

        assert models_session.submitted[1]["prompt_id"] == "prompt-9"

    def test_prompt_and_prompt_id_together_are_refused(self, models, models_session):
        with pytest.raises(TypeError, match="prompt or prompt_id"):
            models.poet.score(["MKT"], prompt="a", prompt_id="b")
        assert models_session.submissions == []

    @pytest.mark.parametrize("prompt", [1, b"prompt", ["prompt-1"]])
    def test_a_wrong_prompt_type_raises_before_any_request(
        self, models, models_session, prompt
    ):
        with pytest.raises(TypeError, match="prompt must be"):
            models.poet2.embed(["MKT"], prompt=prompt)
        models_session.session.post.assert_not_called()


class TestQuery:
    def test_a_sequence_creates_a_query(self, models, models_session):
        models.poet2.embed(["MKT"], query="MKT")

        assert [name for name, _ in models_session.queries] == ["query.fasta"]
        assert models_session.submitted[1]["query_id"] == created_query(1)

    def test_a_query_id_or_query_object_sends_its_id(self, models, models_session):
        models.poet2.embed(["MKT"], query=QUERY_UUID)
        models.poet2.embed(["MKT"], query=_query(models_session.session))

        assert [body["query_id"] for _, body in models_session.submissions] == [
            QUERY_UUID,
            QUERY_UUID,
        ]
        assert models_session.queries == []

    def test_a_protein_and_a_complex_create_queries(self, models, models_session):
        protein = Protein(sequence="MKT")
        complex = Complex({"A": Protein(sequence="MKT"), "B": Protein(sequence="AC")})

        models.poet2.generate(query=[protein, complex], random_seed=1)

        assert models_session.submitted[1]["query_id"] == [
            created_query(1),
            created_query(2),
        ]
        assert len(models_session.queries) == 2

    @pytest.mark.parametrize("query", [1, {"id": QUERY_UUID}, [QUERY_UUID, 2]])
    def test_a_wrong_query_type_raises_before_any_request(
        self, models, models_session, query
    ):
        with pytest.raises(TypeError, match="query must be"):
            models.poet2.generate(query=query)
        models_session.session.post.assert_not_called()


class TestDesign:
    @pytest.mark.parametrize("status", ["PENDING", "RUNNING", "SUCCESS"])
    def test_a_design_is_sent_as_its_id_whatever_its_status(
        self, models, models_session, status
    ):
        design = _design(models_session.session, status=status)

        models.proteinmpnn.generate(design=design, random_seed=1)

        assert models_session.submitted == (
            "v1/models/proteinmpnn/generate/sequences",
            {"design_id": DESIGN_UUID, "random_seed": 1},
        )
        assert models_session.queries == []

    def test_a_design_id_is_sent_as_it_is(self, models, models_session):
        models.esmif1.generate(design=DESIGN_UUID)

        assert models_session.submitted == (
            "v1/models/esm-if1/generate/sequences",
            {"design_id": DESIGN_UUID},
        )

    def test_a_legacy_structure_generation_future_sends_its_id(
        self, models, models_session
    ):
        job = StructureGenerationJob(
            job_id=DESIGN_UUID,
            job_type="/models/design",
            status=JobStatus.RUNNING,
            created_date="2026-10-21T08:00:00Z",
        )
        design = StructureGenerationFuture(models_session.session, job, N=1)

        models.esmif1.generate(design=design, random_seed=1)

        assert models_session.submitted[1] == {
            "design_id": DESIGN_UUID,
            "random_seed": 1,
        }

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"query": QUERY_UUID},
            {"query_id": QUERY_UUID},
        ],
        ids=["query", "query_id"],
    )
    def test_a_query_and_a_design_are_refused_before_any_upload(
        self, models, models_session, kwargs
    ):
        with pytest.raises(TypeError, match="pass query or design, not both"):
            models.proteinmpnn.generate(
                design=_design(models_session.session), **kwargs
            )
        assert models_session.queries == []
        models_session.session.post.assert_not_called()

    def test_a_design_and_a_design_id_are_refused(self, models, models_session):
        with pytest.raises(TypeError, match="pass design or design_id, not both"):
            models.proteinmpnn.generate(
                design=_design(models_session.session), design_id=DESIGN_UUID
            )
        models_session.session.post.assert_not_called()

    def test_generating_from_neither_a_query_nor_a_design_is_refused(
        self, models, models_session
    ):
        with pytest.raises(TypeError, match="needs a query or a design"):
            models.proteinmpnn.generate()
        models_session.session.post.assert_not_called()

    @pytest.mark.parametrize(
        "path, call",
        [
            (
                "esm-if1/generate/sequences",
                lambda m: m.esmif1.generate(query=[QUERY_UUID]),
            ),
            (
                "proteinmpnn/single-site/scores",
                lambda m: m.proteinmpnn.single_site("MKT", query=[QUERY_UUID]),
            ),
            (
                "rfdiffusion/generate/structures",
                lambda m: m.rfdiffusion.generate(query=[QUERY_UUID]),
            ),
        ],
        ids=["esm-if1 generate", "proteinmpnn single site", "rfdiffusion"],
    )
    def test_one_query_in_a_list_is_sent_as_its_id(
        self, models, models_session, path, call
    ):
        call(models)

        submitted_path, body = models_session.submitted
        assert submitted_path == f"v1/models/{path}"
        assert body["query_id"] == QUERY_UUID

    def test_a_wrong_design_type_raises_before_any_request(
        self, models, models_session
    ):
        with pytest.raises(TypeError, match="design must be"):
            models.proteinmpnn.generate(query="MKT", design="design-1")
        models_session.session.post.assert_not_called()

    def test_a_future_that_is_not_a_design_raises_before_any_request(
        self, models, models_session
    ):
        """A future whose job serves no outputs is refused by type, with no request."""
        job = Job.model_validate(
            {
                "job_id": "msa-1",
                "job_type": "/align/mafft",
                "status": "SUCCESS",
                "created_date": "2026-10-21T08:00:00Z",
            }
        )
        msa = MSAFuture(models_session.session, job)
        models_session.session.get.reset_mock()

        with pytest.raises(TypeError, match="MSAFuture"):
            models.proteinmpnn.generate(design=msa)
        models_session.session.get.assert_not_called()
        models_session.session.post.assert_not_called()

    def test_a_models_job_of_another_method_is_not_a_design(
        self, models, models_session
    ):
        job = Job.model_validate(
            {
                "job_id": DESIGN_UUID,
                "job_type": "/models/esm2_t6_8M_UR50D/batch/embeddings",
                "status": "SUCCESS",
                "created_date": "2026-10-21T08:00:00Z",
            }
        )
        models_session.session.get.reset_mock()

        with pytest.raises(TypeError, match="not generate/structures"):
            models.proteinmpnn.generate(design=Future(models_session.session, job))
        models_session.session.get.assert_not_called()
        models_session.session.post.assert_not_called()

    @pytest.mark.parametrize(
        "call",
        [
            lambda m, d: m.esmif1.generate(query=["MKT", "MKV"]),
            lambda m, d: m.proteinmpnn.score(["MKT"], query=["MKT", "MKV"]),
            lambda m, d: m.poet2.single_site("MKT", query=["MKT", "MKV"]),
            lambda m, d: m.rfdiffusion.generate(query=["MKT", "MKV"]),
        ],
        ids=["esm-if1 list", "score", "single site", "rfdiffusion"],
    )
    def test_several_queries_to_a_one_query_method_raise_before_any_upload(
        self, models, models_session, call
    ):
        with pytest.raises(ValueError, match="takes one query, not 2"):
            call(models, None)
        assert models_session.queries == []
        models_session.session.post.assert_not_called()

    @pytest.mark.parametrize(
        "model_id, call",
        [
            (
                "rfdiffusion",
                lambda m: m.rfdiffusion.generate(query=[QUERY_UUID, QUERY_UUID]),
            ),
            (
                "boltzgen",
                lambda m: m.boltzgen.generate(query=[QUERY_UUID, QUERY_UUID]),
            ),
        ],
    )
    def test_a_structure_design_takes_several_queries_where_its_query_id_takes_a_list(
        self, models, models_session, monkeypatch, model_id, call
    ):
        path = f"{model_id}/generate/structures"
        schema = copy.deepcopy(PARAMS["methods"][path]["params"])
        schema["properties"]["query_id"] = PARAMS["methods"][
            "proteinmpnn/generate/sequences"
        ]["params"]["properties"]["query_id"]
        models_session.get_routes[f"v1/models/{path}/params"] = schema
        monkeypatch.setattr(support, "check_body", lambda path, body: None)

        call(models)

        assert models_session.submitted[1]["query_id"] == [QUERY_UUID, QUERY_UUID]

    @pytest.mark.parametrize(
        "model_id, call",
        [
            (
                "proteinmpnn",
                lambda m: m.proteinmpnn.generate(query=[QUERY_UUID, QUERY_UUID]),
            ),
            ("poet-2", lambda m: m.poet2.generate(query=[QUERY_UUID, QUERY_UUID])),
        ],
    )
    def test_a_generate_takes_one_query_where_its_query_id_takes_no_list(
        self, models, models_session, model_id, call
    ):
        path = f"{model_id}/generate/sequences"
        schema = copy.deepcopy(PARAMS["methods"][path]["params"])
        schema["properties"]["query_id"] = PARAMS["methods"][
            "esm-if1/generate/sequences"
        ]["params"]["properties"]["query_id"]
        models_session.get_routes[f"v1/models/{path}/params"] = schema

        with pytest.raises(ValueError, match="takes one query, not 2"):
            call(models)
        models_session.session.post.assert_not_called()


class TestPLM:
    @pytest.mark.parametrize(
        "alias, model_id",
        [("esm2", "esm2_t33_650M_UR50D"), ("esmc_300m", "esmc-300m")],
    )
    def test_esm2_and_esmc_score_single_site_mutants(
        self, models, models_session, alias, model_id
    ):
        getattr(models, alias).single_site(b"MKT")

        assert models_session.submitted == (
            f"v1/models/{model_id}/single-site/scores",
            {"base_sequence": "MKT"},
        )

    @pytest.mark.parametrize("alias", ["prot_seq", "ablang2", "prott5_xl", "poet"])
    def test_embed_pools_by_mean_by_default(self, models, models_session, alias):
        getattr(models, alias).embed(["MKT"])

        assert models_session.submitted[1]["reduction"] == "MEAN"

    def test_embed_none_is_per_residue(self, models, models_session):
        models.prot_seq.embed(["MKT"], reduction=None)

        assert models_session.submitted[1]["reduction"] is None

    def test_logits_pass_chain_params(self, models, models_session):
        models.prot_seq.logits(["MKT:AC"], chain_mode="independent")

        assert models_session.submitted == (
            "v1/models/prot-seq/batch/logits",
            {"sequences": ["MKT:AC"], "chain_mode": "independent"},
        )

    def test_fit_svd_submits_the_fit(self, models, models_session):
        models.esm2.fit_svd(["MKT"], n_components=8, reduction="MEAN")

        assert models_session.submitted == (
            "v1/models/svd",
            {
                "n_components": 8,
                "output": "embeddings",
                "sequences": ["MKT"],
                "model_id": "esm2_t33_650M_UR50D",
                "reduction": "MEAN",
            },
        )

    def test_fit_umap_submits_the_fit_with_its_defaults(self, models, models_session):
        models.prot_seq.fit_umap(["MKT", "AC", "DE", "FG"])

        assert models_session.submitted == (
            "v1/models/umap",
            {
                "model_id": "prot-seq",
                "sequences": ["MKT", "AC", "DE", "FG"],
                "reduction": "MEAN",
            },
        )

    def test_fit_gp_submits_the_training(self, models, models_session):
        models.ablang2.fit_gp("assay-1", ["fitness"], reduction="MEAN")

        assert models_session.submitted == (
            "v1/models/gp",
            {
                "features": {
                    "model_id": "ablang2",
                    "reduction": "MEAN",
                },
                "kernel": {"type": "rbf"},
                "training_dataset": {
                    "assay_id": "assay-1",
                    "properties": ["fitness"],
                },
            },
        )

    def test_poet2_fits_carry_its_embedding_params(self, models, models_session):
        models.poet2.fit_svd(
            ["MKT"],
            n_components=4,
            prompt="prompt-1",
            query=QUERY_UUID,
            decoder_type="mlm",
        )
        models.poet2.fit_gp(
            "assay-1",
            ["fitness"],
            reduction="MEAN",
            prompt="prompt-1",
            query=QUERY_UUID,
            use_query_structure_in_decoder=False,
        )

        svd, gp = (body for _, body in models_session.submissions)
        assert (svd["prompt_id"], svd["query_id"], svd["decoder_type"]) == (
            "prompt-1",
            QUERY_UUID,
            "mlm",
        )
        assert gp["features"] == {
            "model_id": "poet-2",
            "reduction": "MEAN",
            "prompt_id": "prompt-1",
            "query_id": QUERY_UUID,
            "use_query_structure_in_decoder": False,
        }

    def test_a_wrong_assay_type_raises_before_any_request(self, models, models_session):
        with pytest.raises(TypeError, match="assay must be"):
            models.esm2.fit_gp(3, ["fitness"], reduction="MEAN")
        with pytest.raises(TypeError, match="assay must be"):
            models.esm2.fit_svd(assay=3)
        models_session.session.post.assert_not_called()


class TestChecksBeforeRequests:
    """Every SDK object is type-checked before any request."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda m: m.poet2.fit_gp(
                3, ["fitness"], reduction="MEAN", query=Protein(sequence="MKT")
            ),
            lambda m: m.poet2.fit_svd(assay=3, query=Protein(sequence="MKT")),
            lambda m: m.poet2.fit_umap(assay=3, query=Protein(sequence="MKT")),
            lambda m: m.poet2.generate(query=Protein(sequence="MKT"), design=3),
            lambda m: m.poet2.embed(["MKT"], prompt=3, query=Protein(sequence="MKT")),
            lambda m: m.proteinmpnn.generate(query=[Protein(sequence="MKT"), 3]),
        ],
    )
    def test_a_wrong_type_raises_before_any_request(self, models, models_session, call):
        with pytest.raises(TypeError):
            call(models)
        models_session.session.post.assert_not_called()
        models_session.session.get.assert_not_called()

    def test_an_object_with_its_param_raises_before_any_request(
        self, models, models_session
    ):
        with pytest.raises(TypeError, match="query or query_id"):
            models.poet2.embed(
                ["MKT"], query=Protein(sequence="MKT"), query_id=QUERY_UUID
            )
        models_session.session.post.assert_not_called()
        models_session.session.get.assert_not_called()


class TestPoET:
    def test_indel_sends_insert_and_delete(self, models, models_session):
        models.poet2.indel("MKT", delete=[1, 2], prompt="prompt-1", decoder_type="clm")

        assert models_session.submitted == (
            "v1/models/poet-2/indel/scores",
            {
                "base_sequence": "MKT",
                "delete": [1, 2],
                "prompt_id": "prompt-1",
                "decoder_type": "clm",
            },
        )

    @pytest.mark.parametrize("handle,model_id", [("poet", "poet"), ("poet2", "poet-2")])
    @pytest.mark.parametrize(
        "params", [{}, {"include_gaps": False}, {"include_gaps": True}]
    )
    def test_single_site_forwards_include_gaps(
        self, models, models_session, handle, model_id, params
    ):
        getattr(models, handle).single_site(b"A-:C-", prompt="prompt-1", **params)

        assert models_session.submitted == (
            f"v1/models/{model_id}/single-site/scores",
            {"base_sequence": "A-:C-", "prompt_id": "prompt-1", **params},
        )

    def test_single_site_takes_the_prompt_and_query(self, models, models_session):
        models.poet2.single_site(b"MKT", prompt="prompt-1", query=QUERY_UUID)

        assert models_session.submitted == (
            "v1/models/poet-2/single-site/scores",
            {"base_sequence": "MKT", "prompt_id": "prompt-1", "query_id": QUERY_UUID},
        )


class TestInverseFolding:
    @pytest.mark.parametrize("alias", ["proteinmpnn", "solublempnn", "esmif1"])
    def test_scoring_sends_the_query(self, models, models_session, alias):
        model = getattr(models, alias)
        model.score(["MKT"], query=QUERY_UUID)
        model.single_site("MKT", query=QUERY_UUID)

        assert [body for _, body in models_session.submissions] == [
            {"sequences": ["MKT"], "query_id": QUERY_UUID},
            {"base_sequence": "MKT", "query_id": QUERY_UUID},
        ]

    @pytest.mark.parametrize("alias", ["proteinmpnn", "solublempnn"])
    def test_embed_pools_by_mean_by_default(self, models, models_session, alias):
        getattr(models, alias).embed(["MKT"], query=QUERY_UUID)
        getattr(models, alias).embed(["MKT"], query=QUERY_UUID, reduction=None)

        assert [body for _, body in models_session.submissions] == [
            {"sequences": ["MKT"], "query_id": QUERY_UUID, "reduction": "MEAN"},
            {"sequences": ["MKT"], "query_id": QUERY_UUID, "reduction": None},
        ]


class TestDesignModels:
    def test_rfdiffusion_sends_the_query_and_its_params(self, models, models_session):
        models.rfdiffusion.generate(
            query=Protein.from_expr(50), num_samples=2, random_seed=3
        )

        path, body = models_session.submitted
        assert path == "v1/models/rfdiffusion/generate/structures"
        assert body == {
            "query_id": created_query(1),
            "num_samples": 2,
            "random_seed": 3,
        }

    def test_rfdiffusion_generate_takes_a_query_and_params(self):
        assert list(inspect.signature(RFdiffusion.generate).parameters) == [
            "self",
            "query",
            "params",
        ]

    def test_boltzgen_sends_the_query_and_its_params(self, models, models_session):
        models.boltzgen.generate(
            query=Complex({"A": Protein(sequence="MKT")}),
            num_samples=2,
            diffusion_batch_size=1,
            random_seed=5,
        )

        path, body = models_session.submitted
        assert path == "v1/models/boltzgen/generate/structures"
        assert body == {
            "query_id": created_query(1),
            "num_samples": 2,
            "diffusion_batch_size": 1,
            "random_seed": 5,
        }

    def test_boltzgen_generate_takes_a_query_and_params(self):
        assert list(inspect.signature(BoltzGen.generate).parameters) == [
            "self",
            "query",
            "params",
        ]

    @pytest.mark.parametrize("model_id", ["boltzgen", "rfdiffusion"])
    def test_a_design_model_takes_the_catalog_query_id(
        self, models, models_session, model_id
    ):
        getattr(models, model_id).generate(query_id=QUERY_UUID, random_seed=7)

        assert models_session.submitted[1] == {"query_id": QUERY_UUID, "random_seed": 7}

    @pytest.mark.parametrize("model_id", ["boltzgen", "rfdiffusion"])
    def test_a_design_model_refuses_a_query_with_query_id(
        self, models, models_session, model_id
    ):
        with pytest.raises(TypeError, match="query or query_id"):
            getattr(models, model_id).generate(
                query=Protein(sequence="MKT"), query_id=QUERY_UUID
            )
        models_session.session.post.assert_not_called()


class TestHandleNames:
    """The 0.17.0 handle names are the models API handles."""

    @pytest.mark.parametrize(
        "handle, model_id",
        [
            (RFdiffusion, "rfdiffusion"),
            (ProteinMPNN, "proteinmpnn"),
            (ESMIF1, "esm-if1"),
            (BoltzGen, "boltzgen"),
        ],
    )
    def test_each_resolves_from_session_models(self, models, handle, model_id):
        resolved: Any = models.get_model(model_id)

        assert type(resolved) is handle
        assert handle(models.session).id == model_id

    def test_family_handles(self, models):
        assert type(models.poet) is PoET
        assert type(models.poet2) is PoET2
        assert type(models.prot_seq) is ProtSeq

    def test_solublempnn_is_a_proteinmpnn_checkpoint(self, models):
        resolved: Any = models.get_model("solublempnn")

        assert (type(resolved), resolved.id) == (ProteinMPNN, "solublempnn")


class TestParamsSchema:
    def test_a_handle_reads_a_methods_params_schema_once(self, models, models_session):
        handle = models.esmfold

        first = handle._params_schema("batch/structures")
        second = handle._params_schema("batch/structures")

        assert first is second
        assert "complexes" in first["properties"]
        reads = [
            call.args[0]
            for call in models_session.session.get.call_args_list
            if call.args[0].endswith("/params")
        ]
        assert reads == ["v1/models/esmfold/batch/structures/params"]


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.poet2.score(["MKT"], decoder_type="mlm"),
        lambda m: m.poet2.single_site("MKT", decoder_type="mlm"),
        lambda m: m.poet2.indel("MKT", insert="A", decoder_type="mlm"),
        lambda m: m.poet.generate(prompt="prompt-1", query_id=QUERY_UUID),
        lambda m: m.rfdiffusion.generate(),
        lambda m: m.boltzgen.generate(),
    ],
    ids=[
        "poet-2 mlm score",
        "poet-2 mlm single site",
        "poet-2 mlm indel",
        "poet with a query_id",
        "rfdiffusion without a query",
        "boltzgen without a query",
    ],
)
def test_a_refusal_from_the_models_api_reaches_the_caller(models, models_session, call):
    refused = MagicMock(status_code=422, text="refused", url="v1/models")
    models_session.session.post.side_effect = HTTPError(refused)

    with pytest.raises(HTTPError) as caught:
        call(models)

    assert caught.value.status_code == 422


@pytest.mark.parametrize(
    ("model_id", "param", "value"),
    [
        ("boltzgen", "design_spec", {"entities": []}),
        ("boltzgen", "scaffolds", {"s.cif": QUERY_UUID}),
        ("boltzgen", "extra_structure_files", {"t.cif": QUERY_UUID}),
        ("boltzgen", "scaffold_set", "default_vhh"),
        ("rfdiffusion", "contigs", "50-50"),
        ("rfdiffusion", "hotspot", "A10"),
        ("rfdiffusion", "scaffold_target_query_id", QUERY_UUID),
        ("rfdiffusion", "symmetry", "cyclic"),
    ],
)
def test_a_design_model_sends_a_param_it_does_not_name_as_given(
    models, models_session, model_id, param, value
):
    refused = MagicMock(status_code=422, text="refused", url="v1/models")
    models_session.session.post.side_effect = HTTPError(refused)

    with pytest.raises(HTTPError) as caught:
        getattr(models, model_id).generate(query_id=QUERY_UUID, **{param: value})

    assert caught.value.status_code == 422
    assert models_session.session.post.call_args.kwargs["json"] == {
        "query_id": QUERY_UUID,
        param: value,
    }
