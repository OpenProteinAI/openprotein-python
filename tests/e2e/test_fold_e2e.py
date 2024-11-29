import pytest
import json
from tests.conf import BACKEND, TIMEOUT
from openprotein.api.fold import *
import time
import openprotein
from openprotein.api.align import msa_post, MSAFuture, MSAJob
from openprotein.jobs import job_get

from AWSTools.Batchtools.batch_utils import fakeseq


class Static:
    esmfold_id: str = None
    msa_id: str = None


STATIC = Static()


@pytest.fixture
def api_session():
    with open("./secrets.config", "r") as f:
        secrets = json.load(f)
    sess = openprotein.connect(
        username=secrets["username"], password=secrets["password"], backend=BACKEND
    )
    yield sess


SEQUENCES = [b"LAAAPPPLLL"]
AF_SEQUENCE = "MYRMQLLSCIALSLALVTNSAPTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGMYRMQLLSCIALSLALVTNSAPTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGSEP"
print(f"USING BACKEND: {BACKEND} ")


def test_fold_models_get(api_session):
    models = fold_models_list_get(api_session)
    assert isinstance(models, list)
    assert all(isinstance(model, str) for model in models)
    assert "esmfold" in models


def test_fold_modelmeta(api_session):
    meta = fold_model_get(api_session, "esmfold")
    assert isinstance(meta, ModelMetadata)
    assert meta.model_id == "esmfold"


@pytest.fixture()
def test_fold_post(api_session):
    job = fold_models_esmfold_post(api_session, sequences=SEQUENCES)
    job = job.job

    assert isinstance(job, Job)
    assert isinstance(job.job_id, str)
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    if STATIC.esmfold_id is None:
        STATIC.esmfold_id = job.job_id


def test_fold_get(api_session, test_fold_post):
    job = job_get(api_session, STATIC.esmfold_id)

    assert isinstance(job, Job)
    assert isinstance(job.job_id, str)
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]

    time.sleep(4)
    # job.wait_until_done(api_session)

    f = api_session.load_job(STATIC.esmfold_id)
    assert f
    assert f.wait_until_done(timeout=TIMEOUT)
    pdb = f.get()
    print(pdb)
    pdb = pdb[0][1]
    assert "ATOM" in pdb.decode()
    assert len(pdb.decode().split("\n")) > 10

    sequences = fold_get_sequences(api_session, job_id=STATIC.esmfold_id)
    assert sorted(sequences) == sorted(SEQUENCES)

    pdbresult = fold_get_sequence_result(
        api_session, job_id=STATIC.esmfold_id, sequence=SEQUENCES[0]
    )
    assert "ATOM" in pdbresult.decode()
    assert len(pdbresult.decode().split("\n")) > 10


def test_fold_model(api_session):
    model = ESMFoldModel(api_session, "esmfold")
    assert isinstance(model.metadata, ModelMetadata)
    assert model.metadata.model_id == "esmfold"
    assert model.id == "esmfold"

    future = model.fold(SEQUENCES)
    assert future.wait(timeout=TIMEOUT)
    result = future.wait(verbose=True, timeout=TIMEOUT)
    assert len(result) == 1
    assert len(result[0]) == 2
    assert result[0][0] == SEQUENCES[0]
    assert "ATOM" in result[0][1].decode()


def test_fold_api(api_session):
    models = api_session.fold.list_models()
    assert all(
        [isinstance(m, ESMFoldModel) or isinstance(m, AlphaFold2Model) for m in models]
    )
    assert "esmfold" in [m.id for m in models]

    f = api_session.fold.esmfold.fold(SEQUENCES)

    assert f.wait_until_done()
    result = f.wait()
    assert len(result) == 1
    assert len(result[0]) == 2
    assert result[0][0] == SEQUENCES[0]
    assert "ATOM" in result[0][1].decode()


@pytest.fixture(autouse=False)
def test_msa_post(api_session):
    job = msa_post(api_session, seed=AF_SEQUENCE.encode())
    job = job.job

    assert isinstance(job, MSAJob)
    msaf = MSAFuture(api_session, job)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    if STATIC.msa_id is None:
        STATIC.msa_id = job.job_id
    yield msaf


def test_fold_api_colabfold(api_session, test_msa_post):
    models = api_session.fold.list_models()
    assert all(
        [isinstance(m, ESMFoldModel) or isinstance(m, AlphaFold2Model) for m in models]
    )
    assert "alphafold2" in [m.id for m in models]

    f = api_session.fold.alphafold2.fold(msa=test_msa_post, num_recycles=1)
    time.sleep(2)  # wait for job to reg
    assert f.wait_until_done(timeout=TIMEOUT)
    result = f.wait()
    assert len(result) == 1
    assert len(result[0]) == 2
    assert result[0][0] == AF_SEQUENCE.encode()
    assert "ATOM" in result[0][1].decode()
