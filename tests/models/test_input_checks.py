"""Input checks of ``session.models`` run locally, before any upload or submission."""

import pytest

from openprotein.errors import InvalidParameterError
from openprotein.models import ModelsAPI
from openprotein.models.svd import SVDModel
from openprotein.molecules import Complex, Ligand
from tests.models.support import card


@pytest.fixture
def models(models_session) -> ModelsAPI:
    return ModelsAPI(models_session.session)


def _nothing_sent(models_session) -> None:
    assert models_session.queries == []
    assert models_session.submissions == []


class TestEmptyQueryBatch:
    """A query or design that resolves to no targets is refused, not dropped."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda m: m.poet2.score(["AC"], query=[]),
            lambda m: m.poet2.generate(query=[], num_samples=2),
            lambda m: m.poet2.embed(["AC"], query=[]),
        ],
        ids=["score", "generate", "embed"],
    )
    def test_an_empty_query_list_raises(self, models, models_session, call):
        with pytest.raises(ValueError, match="query=None"):
            call(models)
        _nothing_sent(models_session)

    def test_an_omitted_query_is_sent_without_query_id(self, models, models_session):
        models.poet2.score(["AC"])
        assert "query_id" not in models_session.submitted[1]

    def test_one_query_is_sent_as_a_bare_id(self, models, models_session):
        models.poet2.score(["AC"], query=["AC"])
        assert isinstance(models_session.submitted[1]["query_id"], str)


class TestQueryEntryChecks:
    """A query entry that can't be uploaded is refused before any request."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda m: m.poet2.generate(query=":"),
            lambda m: m.poet2.score(["AC"], query=":"),
        ],
        ids=["generate", "score"],
    )
    def test_an_invalid_query_sequence_makes_no_request(
        self, models, models_session, call
    ):
        _ = models.poet2
        models_session.session.get.reset_mock()
        with pytest.raises(InvalidParameterError):
            call(models)
        models_session.session.get.assert_not_called()
        models_session.session.post.assert_not_called()


class TestBareStringBatch:
    """A bare str or bytes where a list of sequences is expected raises TypeError."""

    @pytest.mark.parametrize("value", ["ACD", b"ACD"], ids=["str", "bytes"])
    @pytest.mark.parametrize(
        "call",
        [
            lambda m, v: m.esm2.embed(v),
            lambda m, v: m.esm2.logits(v),
            lambda m, v: m.esm2.score(v),
            lambda m, v: m.esm2.fit_svd(v),
            lambda m, v: m.esm2.fit_umap(v),
            lambda m, v: m.proteinmpnn.embed(v, query="MKT"),
        ],
        ids=["embed", "logits", "score", "fit_svd", "fit_umap", "proteinmpnn_embed"],
    )
    def test_a_bare_sequence_raises_before_any_request(
        self, models, models_session, call, value
    ):
        with pytest.raises(TypeError, match=r'\["ACD"\]'):
            call(models, value)
        models_session.session.post.assert_not_called()

    def test_a_single_base_sequence_stays_scalar(self, models, models_session):
        models.esm2.single_site("ACD")
        assert models_session.submitted[1]["base_sequence"] == "ACD"

    def test_projection_sequences_must_be_a_list(self, models):
        svd = SVDModel._from_record(models.session, {"id": "s-1"})
        with pytest.raises(TypeError, match="list of sequences"):
            svd.embed("ACD")


def _without(models_session, model_id: str, path: str) -> None:
    listing = card(model_id)
    listing["methods"] = [m for m in listing["methods"] if m["path"] != path]
    models_session.get_routes[f"v1/models/{model_id}"] = listing


class TestLocalChecksBeforeUpload:
    """A call that fails a local check uploads no query and submits nothing."""

    def test_one_invalid_query_in_a_batch_uploads_none(self, models, models_session):
        with pytest.raises(InvalidParameterError, match="no residues"):
            models.poet2.generate(query=["AC", ":"])
        _nothing_sent(models_session)

    def test_a_non_protein_complex_in_a_batch_uploads_none(
        self, models, models_session
    ):
        ligand = Complex({"A": Ligand(ccd="ATP")})
        with pytest.raises(InvalidParameterError, match="protein chains"):
            models.poet2.generate(query=["AC", ligand])
        _nothing_sent(models_session)

    def test_a_valid_batch_uploads_every_query(self, models, models_session):
        models.poet2.generate(query=["AC", "DE"])
        assert len(models_session.queries) == 2
        assert len(models_session.submitted[1]["query_id"]) == 2

    def test_a_method_the_card_lacks_is_refused_before_the_query_upload(
        self, models, models_session
    ):
        with pytest.raises(ValueError, match="has no attn"):
            models.poet2.attn(["AC"], query="AC")
        _nothing_sent(models_session)

    def test_an_inverse_folding_score_checks_its_method_first(
        self, models, models_session
    ):
        _without(models_session, "proteinmpnn", "batch/loglikelihood")
        with pytest.raises(ValueError, match="has no score"):
            models.proteinmpnn.score(["AC"], query="AC")
        _nothing_sent(models_session)

    def test_an_inverse_folding_embed_checks_its_method_first(
        self, models, models_session
    ):
        _without(models_session, "proteinmpnn", "batch/embeddings")
        with pytest.raises(ValueError, match="has no embed"):
            models.proteinmpnn.embed(["AC"], query="AC")
        _nothing_sent(models_session)

    def test_an_inverse_folding_embed_of_a_bare_sequence_uploads_no_query(
        self, models, models_session
    ):
        with pytest.raises(TypeError, match="list of sequences"):
            models.proteinmpnn.embed("AC", query="AC")
        _nothing_sent(models_session)

    def test_an_inverse_folding_generate_checks_its_method_first(
        self, models, models_session
    ):
        _without(models_session, "proteinmpnn", "generate/sequences")
        with pytest.raises(ValueError, match="has no generate"):
            models.proteinmpnn.generate(query="AC")
        _nothing_sent(models_session)

    def test_a_bare_sequence_with_a_query_uploads_no_query(
        self, models, models_session
    ):
        with pytest.raises(TypeError, match="list of sequences"):
            models.poet2.score("AC", query="AC")
        _nothing_sent(models_session)

    @pytest.mark.parametrize("fit", ["fit_svd", "fit_umap"])
    def test_a_poet2_fit_without_a_source_uploads_no_query(
        self, models, models_session, fit
    ):
        with pytest.raises(ValueError, match="pass one of sequences or assay"):
            getattr(models.poet2, fit)(query="AC")
        _nothing_sent(models_session)

    @pytest.mark.parametrize("fit", ["fit_svd", "fit_umap"])
    def test_a_poet_fit_without_a_source_submits_nothing(
        self, models, models_session, fit
    ):
        with pytest.raises(ValueError, match="pass one of sequences or assay"):
            getattr(models.poet, fit)()
        _nothing_sent(models_session)


class TestFitSvdSource:
    """fit_svd takes exactly one of sequences or assay."""

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"sequences": ["MKT"], "assay": "assay-1"}],
        ids=["neither", "both"],
    )
    @pytest.mark.parametrize("handle", ["esm2", "poet", "poet2"])
    def test_neither_or_both_raises_before_any_request(
        self, models, models_session, handle, kwargs
    ):
        with pytest.raises(ValueError, match="pass one of sequences or assay"):
            getattr(models, handle).fit_svd(**kwargs)
        models_session.session.post.assert_not_called()
