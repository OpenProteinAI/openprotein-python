"""Sequences and queries are validated before any request is made."""

import pytest

from openprotein.models import ModelsAPI
from tests.models.test_handles import QUERY_UUID


def _requests(models_session) -> int:
    session = models_session.session
    return (
        session.get.call_count
        + session.post.call_count
        + session.put.call_count
        + session.delete.call_count
    )


CASES = {
    "poet2.score": lambda m, bad: m.poet2.score([bad], query="AC"),
    "poet2.logits": lambda m, bad: m.poet2.logits([bad], query="AC"),
    "poet2.attn": lambda m, bad: m.poet2.attn([bad], query="AC"),
    "poet2.embed": lambda m, bad: m.poet2.embed([bad], query="AC"),
    "poet2.fit_svd": lambda m, bad: m.poet2.fit_svd([bad], query="AC"),
    "poet2.fit_umap": lambda m, bad: m.poet2.fit_umap([bad], query="AC"),
    "poet2.single_site": lambda m, bad: m.poet2.single_site(bad, query="AC"),
    "poet.score": lambda m, bad: m.poet.score([bad]),
    "poet.embed": lambda m, bad: m.poet.embed([bad]),
    "poet.single_site": lambda m, bad: m.poet.single_site(bad),
    "poet.indel": lambda m, bad: m.poet.indel(bad, insert="A"),
    "proteinmpnn.score": lambda m, bad: m.proteinmpnn.score([bad], query=QUERY_UUID),
    "proteinmpnn.embed": lambda m, bad: m.proteinmpnn.embed([bad], query=QUERY_UUID),
    "proteinmpnn.single_site": lambda m, bad: m.proteinmpnn.single_site(
        bad, query=QUERY_UUID
    ),
}

EMPTY_QUERY_CASES = {
    "poet2.generate": lambda m: m.poet2.generate(query=[]),
    "poet2.score": lambda m: m.poet2.score(["AC"], query=[]),
    "poet2.embed": lambda m: m.poet2.embed(["AC"], query=[]),
    "proteinmpnn.generate": lambda m: m.proteinmpnn.generate(query=[]),
    "proteinmpnn.score": lambda m: m.proteinmpnn.score(["AC"], query=[]),
}


@pytest.mark.parametrize("verb", CASES)
def test_undecodable_bytes_are_refused_before_any_request(models_session, verb):
    models = ModelsAPI(models_session.session)

    with pytest.raises(ValueError, match="not valid UTF-8") as raised:
        CASES[verb](models, b"\xff")

    assert isinstance(raised.value.__cause__, UnicodeDecodeError)
    assert _requests(models_session) == 0


def test_undecodable_batch_entry_names_its_index(models_session):
    models = ModelsAPI(models_session.session)

    with pytest.raises(ValueError, match=r"sequences\[1\]"):
        models.poet2.score(["AC", b"\xff"], query="AC")

    assert _requests(models_session) == 0


def test_a_non_text_batch_entry_names_its_index(models_session):
    models = ModelsAPI(models_session.session)

    with pytest.raises(TypeError, match=r"sequences\[0\]"):
        models.poet2.score([1], query="AC")

    assert _requests(models_session) == 0


@pytest.mark.parametrize("verb", EMPTY_QUERY_CASES)
def test_empty_query_is_refused_before_any_request(models_session, verb):
    models = ModelsAPI(models_session.session)

    with pytest.raises(ValueError, match="query has no entries; pass query=None"):
        EMPTY_QUERY_CASES[verb](models)

    assert _requests(models_session) == 0
