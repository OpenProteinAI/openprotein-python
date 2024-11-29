import pytest
from openprotein.api.align import *
import json
from tests.conf import BACKEND, TIMEOUT
import time
import collections
import openprotein
from openprotein.schemas import JobType
from openprotein.jobs import *
from AWSTools.Batchtools.batch_utils import fakeseq

TEST_SEQUENCE = f"{fakeseq(5)}APPMYRMQLLSCIALSLALVTNSAPTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGMYRMQLLSCIALSLALVTNSAPTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGSEP"
print(f"USING BACKEND: {BACKEND} ")


@pytest.fixture
def api_session():
    with open("./secrets.config", "r") as f:
        secrets = json.load(f)
    session = openprotein.connect(
        username=secrets["username"], password=secrets["password"], backend=BACKEND
    )
    yield session


class Static:
    msa_id: str = None
    prompt_id: str = None
    score_job_id: str = None
    ssp_job_id: str = None
    generate_job_id: str = None
    uploaded_prompt_id: str = None


STATIC = Static()


@pytest.fixture(autouse=False)
def test_msa_post(api_session):
    job = msa_post(api_session, seed=TEST_SEQUENCE.encode())
    job = job.job
    print(job)
    assert job.job_type == JobType.align_align
    assert isinstance(job, MSAJob)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    if STATIC.msa_id is None:
        STATIC.msa_id = job.job_id
    yield job.job_id


@pytest.fixture(autouse=False)
def test_prompt_post(api_session, test_msa_post):
    job = prompt_post(
        api_session,
        STATIC.msa_id,
        num_sequences=10,
        min_similarity=0.1,
        num_ensemble_prompts=3,
    )
    job = job.job
    print(job)

    assert isinstance(job, PromptJob)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    if STATIC.prompt_id is None:
        STATIC.prompt_id = job.job_id
    yield job.job_id


def test_upload_invalid_prompt_post(api_session):
    sep = "<END_PROMPT>\n"
    prompt_file = "\n".join(["BB??BB"] * 3) + "\n"
    prompt_file = prompt_file + sep + "AAAAA"
    with pytest.raises(APIError):
        upload_prompt_post(api_session, prompt_file)


@pytest.fixture(autouse=False)
def test_upload_prompt_post(api_session):
    sep = "<END_PROMPT>\n"
    prompt_file = "\n".join(["LKLK"] * 3) + "\n"
    prompt_file = prompt_file + sep + "AAAAA"

    job = upload_prompt_post(api_session, prompt_file)
    job = job.job
    print(job)

    assert isinstance(job, PromptJob)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    if STATIC.uploaded_prompt_id is None:
        STATIC.uploaded_prompt_id = job.job_id
    yield job.job_id


def test_csv_stream():
    response = requests.Response()
    response.raw = BytesIO(b"col1,col2\nval1,val2")
    reader = csv_stream(response)
    assert list(reader) == [["col1", "col2"], ["val1", "val2"]]


def test_get_input(api_session, test_msa_post):
    job = job_get(api_session, job_id=STATIC.msa_id)
    print(job)

    reader = get_input(api_session, job, PoetInputType.INPUT)
    x = list(reader)
    assert len(x) == 1
    assert x[0][0] == "seed" or x[0][0] == "101"
    assert x[0][1] == TEST_SEQUENCE


def test_get_prompt(api_session, test_upload_prompt_post):
    prompt = api_session.load_job(STATIC.uploaded_prompt_id)
    assert isinstance(prompt, PromptFuture)

    prompt.wait_until_done(verbose=True, timeout=TIMEOUT)

    r = prompt.wait(verbose=True)
    x = list(r)
    assert any([i == ["<END_PROMPT>"] for i in x])
    assert len(x) == 9  # total seqs
    for i in x:
        if not i == ["<END_PROMPT>"]:
            assert len(i) == 2

    p1 = list(prompt.get_prompt(1))
    assert len(p1) > 0
    for i in p1:
        assert len(i) == 2

    p2 = list(prompt.get_prompt(2))
    assert len(p2) > 0
    for i in p2:
        assert len(i) == 2

    p3 = list(prompt.get_prompt(3))
    assert len(p3) == 0  # only 2 prompts


def test_get_msa(api_session, test_msa_post):
    msa = api_session.load_job(STATIC.msa_id)
    assert isinstance(msa, MSAFuture)
    msa.wait_until_done(verbose=True, timeout=TIMEOUT)
    assert msa.status == "SUCCESS"

    r = msa.wait(verbose=True)
    x = list(r)
    assert len(x) > 1
    assert x[0][0] == "seed" or x[0][0] == "101"
    assert x[0][1] == TEST_SEQUENCE

    assert len(list(msa.get_input("GENERATED"))) > 1
    assert len(list(msa.get_input("RAW"))) == 1


def test_msa_future(api_session, test_msa_post):
    job = job_get(api_session, job_id=STATIC.msa_id)
    print(job)

    future = MSAFuture(api_session, job)
    assert future.wait_until_done(verbose=True, timeout=TIMEOUT)

    assert future.id == STATIC.msa_id
    assert future.msa_id == STATIC.msa_id
    assert future.prompt_id is None

    reader = future.wait(verbose=True)
    assert isinstance(reader, collections.Iterator)

    reader = future.get()
    assert isinstance(reader, collections.Iterator)

    prompt_future = future.sample_prompt()
    assert isinstance(prompt_future, PromptFuture)


def test_prompt_future(api_session, test_prompt_post):
    job = job_get(api_session, job_id=STATIC.prompt_id)
    future = PromptFuture(api_session, job, msa_id=STATIC.msa_id)
    assert future.wait_until_done(verbose=True, timeout=TIMEOUT)

    assert future.id == STATIC.prompt_id
    assert future.msa_id == STATIC.msa_id
    assert future.prompt_id == STATIC.prompt_id

    reader = future.wait(verbose=True)
    assert isinstance(reader, collections.Iterator)

    reader = future.get()
    assert isinstance(reader, collections.Iterator)


def test_prompt_future_get(api_session, test_upload_prompt_post):
    # job = api_session.poet.load_prompt_job(STATIC.prompt_id).wait_until_done(timeout=120)
    job = job_get(api_session, job_id=STATIC.prompt_id)
    print(job)
    future = PromptFuture(session=api_session, job=job)

    reader = future.get_input(PoetInputType.MSA)
    assert isinstance(reader, collections.Iterator)

    reader = future.get_prompt()
    assert isinstance(reader, collections.Iterator)

    reader = future.get_seed()
    assert isinstance(reader, collections.Iterator)

    reader = future.get_msa()
    assert isinstance(reader, collections.Iterator)
