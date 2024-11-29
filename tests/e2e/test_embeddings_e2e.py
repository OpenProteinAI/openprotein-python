import pytest
import time

import numpy as np
from openprotein.base import APISession
from AWSTools.Batchtools.batch_utils import fakeseq
from openprotein.api.embedding import *
from openprotein.api.embedding import (
    embedding_get,
    embedding_model_get,
    embedding_models_list_get,
)
from openprotein.api.jobs import load_job
from openprotein.base import APISession
from openprotein.jobs import *
from tests.conf import BACKEND


class Static:
    job_id: str
    svd_id: str


STATIC = Static()


@pytest.fixture
def api_session():
    with open("./secrets.config", "r") as f:
        secrets = json.load(f)
    sess = APISession(
        username=secrets["username"], password=secrets["password"], backend=BACKEND
    )
    yield sess


SEQUENCES = [fakeseq(5).encode() for i in range(3)]
print(f"USING BACKEND: {BACKEND} ")


@pytest.fixture(autouse=False)
def test_embedding_model_post(api_session):
    model_id = "prot-seq"
    sequences = [b"AAAPPPLLL", b"AAAPPPLLK"]
    job = embedding_model_post(api_session, model_id, sequences)
    job = job.job
    assert isinstance(job, Job)
    assert job.job_id is not None
    STATIC.job_id = job.job_id

    # job.wait_until_done(api_session)
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    time.sleep(4)
    # job.wait_until_done(api_session)

    job_details = load_job(api_session, STATIC.job_id)
    sequences = embedding_get_sequences(api_session, job_id=STATIC.job_id)
    assert sorted(sequences) == sorted([b"AAAPPPLLL", b"AAAPPPLLK"])
    assert job_details


@pytest.fixture(autouse=False)
def test_svd_post(api_session):
    sequences = [b"AAAPPPLLL"]
    job = svd_fit_post(
        api_session, model_id="prot-seq", sequences=sequences, reduction="MEAN"
    )
    assert isinstance(job, SVDJob)
    STATIC.svd_id = job.job_id


def test_embedding_model_get(api_session):
    model_id = "prot-seq"
    model_metadata = embedding_model_get(api_session, model_id)
    assert isinstance(model_metadata, ModelMetadata)
    assert model_metadata.model_id == model_id


def test_embedding_models_get(api_session):
    models = embedding_models_list_get(api_session)
    assert isinstance(models, list)
    assert all(isinstance(model, str) for model in models)


def test_embedding_get_sequences(api_session, test_embedding_model_post):
    job_id = STATIC.job_id
    sequences = embedding_get_sequences(api_session, job_id)
    assert isinstance(sequences, list)
    assert all(isinstance(seq, bytes) for seq in sequences)


def test_embedding_get_sequence_result(api_session, test_embedding_model_post):
    job_id = STATIC.job_id
    sequence = b"AAAPPPLLL"
    result = embedding_get_sequence_result(api_session, job_id, sequence)
    assert isinstance(result, bytes)

    arr = decode_embedding(result)
    assert isinstance(arr, np.ndarray)


def test_embedding_get(api_session, test_embedding_model_post):
    job_id = STATIC.job_id
    job = embedding_get(api_session, job_id)
    job = job.job

    assert isinstance(job, Job)
    assert job.job_id == job_id


def test_embedding_result_future_sequences(api_session, test_embedding_model_post):
    job_id = STATIC.job_id
    sequences = [b"AAAPPPLLL"]
    future = EmbeddingResultFuture(
        api_session,
        Job(job_id=job_id, job_type="/embeddings/logits", status="PENDING"),
        sequences=sequences,
    )
    assert future.sequences == sequences


def test_embedding_result_future_get_item(api_session, test_embedding_model_post):

    job_id = STATIC.job_id
    sequence = b"AAAPPPLLL"
    future = EmbeddingResultFuture(
        api_session,
        Job(job_id=job_id, job_type="/embeddings/logits", status="PENDING"),
        sequences=[sequence],
    )
    result = future.get_item(sequence)
    assert isinstance(result, np.ndarray)


def test_protembed_model_embed(api_session):

    model_id = "prot-seq"
    model = ProtembedModel(api_session, model_id)
    sequences = [b"AAAPPPLLL"]
    future = model.embed(sequences)
    assert isinstance(future, EmbeddingResultFuture)


def test_protembed_model_logits(api_session):

    model_id = "prot-seq"
    model = ProtembedModel(api_session, model_id)
    sequences = [b"AAAPPPLLL"]
    future = model.logits(sequences)
    assert isinstance(future, EmbeddingResultFuture)


def tst_svd_get(api_session, test_svd_post):

    meta = svd_get(api_session, STATIC.svd_id)
    assert isinstance(meta, SVDMetadata)


def test_svd_model_embed(api_session, test_svd_post):
    svd_id = STATIC.svd_id
    meta = svd_get(api_session, STATIC.svd_id)
    svd_model = SVDModel(session=api_session, metadata=meta)
    sequences = [b"AAAPPPLLL"]
    future = svd_model.embed(sequences)
    assert isinstance(future, EmbeddingResultFuture)


def test_embedding_model_logits_post(api_session):
    model_id = "prot-seq"
    sequences = [b"AAAPPPLLL"]
    job = embedding_model_logits_post(api_session, model_id, sequences)
    job = job.job
    assert isinstance(job, Job)
    assert job.job_id is not None


def test_embedding_model_attn_post(api_session):
    model_id = "prot-seq"
    sequences = [b"AAAPPPLLL"]
    job = embedding_model_attn_post(api_session, model_id, sequences)
    job = job.job

    assert isinstance(job, Job)
    assert job.job_id is not None


def test_svd_embed_post(api_session, test_svd_post):
    svd_id = STATIC.svd_id
    sequences = [b"AAAPPPLLL"]
    job = svd_embed_post(api_session, svd_id, sequences)
    job = job.job

    assert isinstance(job, Job)
    assert job.job_id is not None


def test_embedding_api_list_models(api_session):
    api = EmbeddingAPI(api_session)
    models = api.list_models()
    assert isinstance(models, list)
    assert all(isinstance(model, ProtembedModel) for model in models)


def test_embedding_api_get_model(api_session):
    api = EmbeddingAPI(api_session)
    model = api.get_model("prot-seq")
    assert isinstance(model, ProtembedModel)
    assert model.id == "prot-seq"
    future = model.embed([b"AAAPPPLLL"], reduction="MEAN")
    assert isinstance(future, EmbeddingResultFuture)
    time.sleep(2)
    assert future
    assert future.wait_until_done(timeout=600)  # dev scales to 0 so slow
    assert future.status == "SUCCESS"


@pytest.mark.skip("embedding.embed() not available")
def test_embedding_api_embed(api_session):
    api = EmbeddingAPI(api_session)
    sequences = [b"AAAPPPLLL"]
    future = api.embed("prot-seq", sequences)
    assert isinstance(future, EmbeddingResultFuture)


@pytest.mark.skip("embedding.embed() not available")
def test_embedding_api_fit_svd(api_session):
    api = EmbeddingAPI(api_session)
    sequences = [b"AAAPPPLLL"]
    svd_model = api.fit_svd("prot-seq", sequences)
    assert isinstance(svd_model, SVDModel)


def test_embedding_api_get_svd(api_session, test_svd_post):
    api = EmbeddingAPI(api_session)
    svd_model = api.get_svd(STATIC.svd_id)
    assert isinstance(svd_model, SVDModel)


def test_embedding_api_list_svd(api_session):
    api = EmbeddingAPI(api_session)
    svd_models = api.list_svd()
    assert isinstance(svd_models, list)
    assert all(isinstance(model, SVDModel) for model in svd_models)


def test_futures(api_session):
    # can be long
    api = EmbeddingAPI(api_session)
    # sequences = [fakeseq(5).encode() for i in range(3)]
    sequences = [fakeseq(5).encode()] + [b"AAAPPPLLL", b"AAAPPPLLK"]
    model = api.get_model("prot-seq")
    assert isinstance(model, ProtembedModel)
    assert model.id == "prot-seq"
    future = model.embed(sequences, reduction="MEAN")
    assert future.sequences == sequences
    time.sleep(2)  # job reg takes a moment for BE
    assert future.wait_until_done(timeout=600)  # dev scales to 0 so slow
    assert isinstance(future.wait(), list)
    assert len(future.wait()) == len(sequences)

    future.refresh()
    assert future.status == "SUCCESS"
