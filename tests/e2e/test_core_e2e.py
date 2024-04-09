import pytest
import json
from tests.conf import BACKEND
import time
import openprotein
from openprotein.api.train import *
import pandas as pd
from AWSTools.Batchtools.batch_utils import fakeseq


class Static:
    train: str = None
    predict: str = None


STATIC = Static()
print(f"USING BACKEND: {BACKEND} ")


@pytest.fixture
def api_session():
    with open("./secrets.config", "r") as f:
        secrets = json.load(f)
    sess = openprotein.connect(
        username=secrets["username"], password=secrets["password"], backend=BACKEND
    )
    yield sess


def test_dataset_create(api_session):
    dataset = pd.read_csv("./tests/data/AMIE_PSEAE.csv")
    assay = api_session.data.create(dataset, "Dataset Name", "Dataset description")
    assert assay
    assert assay.id
    d = assay.get_slice(start=3, end=5)
    assert isinstance(d, pd.DataFrame)
    assert d.shape[0] == 2


@pytest.fixture(autouse=False)
def test_dataset_train(api_session):
    dataset = pd.read_csv("./tests/data/AMIE_PSEAE.csv")
    assay = api_session.data.create(dataset, "Dataset Name", "Dataset description")
    train = api_session.train.create_training_job(
        assay, measurement_name=["acetamide_normalized_fitness"], model_name="mymodel"
    )  # name the resulting model

    assert train
    assert train.job.status in ["PENDING", "RUNNING", "SUCCEEDED"]
    assert train.wait_until_done(timeout=500)
    results = train.wait(verbose=True)
    assert len(results) > 0
    STATIC.train = train.job.job_id
    yield train


def test_train_predict(api_session, test_dataset_train):
    d = pd.read_csv("./tests/data/AMIE_PSEAE.csv")
    seqlen = len(d.sequence[0])
    p_seqs = [fakeseq(seqlen) for i in range(3)]
    train = test_dataset_train
    pjob = train.predict(sequences=p_seqs)
    assert isinstance(pjob, PredictFuture)
    assert pjob.wait_until_done(timeout=500)

    results = pjob.wait(verbose=True)
    assert len(results)
    assert len(results["acetamide_normalized_fitness"]) > 1

    results = results["acetamide_normalized_fitness"]
    for i in p_seqs:
        assert results[i]
        assert results[i]["mean"]
        assert results[i]["variance"]
