import pytest
from openprotein.api.align import *
import json
from tests.conf import BACKEND
import time
import collections
import openprotein
from openprotein.schemas import JobType
from openprotein.jobs import *
import numpy as np
from openprotein.api.poet import *
from openprotein.api.jobs import load_job

TEST_SEQUENCE = "MYRMQLLSCIALSLALVTNSAPTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGMYRMQLLSCIALSLALVTNSAPTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGSEP"
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
    prompt_single: str = None


STATIC = Static()


@pytest.fixture(autouse=False)
def test_msa_post(api_session):
    job = msa_post(api_session, seed=TEST_SEQUENCE.encode())
    job = job.job

    assert job.job_type == JobType.align_align
    assert isinstance(job, MSAJob)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    if STATIC.msa_id is None:
        STATIC.msa_id = job.job_id
    yield job.job_id


@pytest.fixture(autouse=False)
def test_upload_prompt_post(api_session):
    sep = "<END_PROMPT>\n"
    prompt_file = "\n".join(["LKLK"] * 3) + "\n"
    prompt_file = prompt_file + sep + "AAAAA"

    job = upload_prompt_post(api_session, prompt_file)
    job = job.job
    assert isinstance(job, PromptJob)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    if STATIC.prompt_id is None:
        STATIC.prompt_id = job.job_id
    yield job.job_id


@pytest.fixture(autouse=False)
def test_upload_single_prompt(api_session):
    prompt_file = "\n".join(["LKLK"] * 3) + "\n"

    job = upload_prompt_post(api_session, prompt_file)
    job = job.job

    assert isinstance(job, PromptJob)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    if STATIC.prompt_single is None:
        STATIC.prompt_single = job.job_id
    yield job.job_id


def test_csv_stream():
    response = requests.Response()
    response.raw = BytesIO(b"col1,col2\nval1,val2")
    reader = csv_stream(response)
    assert list(reader) == [["col1", "col2"], ["val1", "val2"]]


@pytest.fixture(autouse=False)
def test_poet_score_post(api_session, test_upload_prompt_post):
    queries = [b"AAAPPPLLL", b"AAAAPPPLK"]
    job = poet_score_post(api_session, prompt_id=STATIC.prompt_id, queries=queries)
    job = job.job
    assert isinstance(job, PoetScoreJob)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    assert "poet" in job.job_type
    if STATIC.score_job_id is None:
        STATIC.score_job_id = job.job_id
    yield job.job_id


@pytest.fixture(autouse=False)
def test_poet_single_site_post(api_session, test_upload_prompt_post):
    variant = "AAAPPPLLL"
    job = poet_single_site_post(
        api_session, variant=variant, prompt_id=STATIC.prompt_id
    )
    job = job.job
    assert isinstance(job, PoetSSPJob)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    assert "/poet/single_site" in job.job_type
    if STATIC.ssp_job_id is None:
        STATIC.ssp_job_id = job.job_id
    yield job.job_id


@pytest.fixture(autouse=False)
def test_poet_generate_post(api_session, test_upload_prompt_post):
    num_samples = 3
    temperature = 1.0
    job = poet_generate_post(
        api_session,
        prompt_id=STATIC.prompt_id,
        max_length=5,
        num_samples=num_samples,
        temperature=temperature,
    )
    job = job.job
    assert isinstance(job, Job)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    assert "/poet/generate" in job.job_type
    if STATIC.generate_job_id is None:
        STATIC.generate_job_id = job.job_id
    yield job.job_id


def test_post_poet_jobs(
    test_poet_score_post, test_poet_single_site_post, test_poet_generate_post
):
    assert True


def test_poet_future(api_session, test_upload_prompt_post):
    # job = api_session.poet.load_prompt_job(STATIC.prompt_id).wait_until_done(timeout=120)
    job = job_get(api_session, job_id=STATIC.prompt_id)
    future = PoetScoreFuture(session=api_session, job=job)
    # future = PromptFuture(session=api_session, job=job)

    reader = future.get_input(PoetInputType.MSA)
    assert isinstance(reader, collections.Iterator)

    reader = future.get_prompt()
    assert isinstance(reader, collections.Iterator)

    reader = future.get_seed()
    assert isinstance(reader, collections.Iterator)

    reader = future.get_msa()
    assert isinstance(reader, collections.Iterator)


def test_poet_score_get(api_session, test_poet_score_post):
    job = job_get(api_session, job_id=STATIC.score_job_id)
    assert isinstance(job, PoetScoreJob)
    timeout = 600  # dev scales to 0 so slow
    while job.status != "SUCCESS":
        time.sleep(10)
        job = job_get(api_session, job_id=STATIC.score_job_id)  # refresh results
        print(job.status)
        timeout += -10
        if timeout < 0:
            assert False, "timeout"

    job = poet_score_get(api_session, job_id=STATIC.score_job_id)
    assert len(job.result) == 2
    assert len(job.result[0].score) == 2  # 2 prompts

    # futures
    job = job_get(api_session, job_id=STATIC.score_job_id)
    future = PoetScoreFuture(api_session, job)
    assert future.wait_until_done(timeout=10)

    for results in [future.get(), future.wait()]:
        assert isinstance(results, list)
        for i in results:
            assert isinstance(i[0], str)
            assert isinstance(i[1], bytes)
            assert isinstance(i[2], np.ndarray)


def test_poet_single_site_get(api_session, test_poet_single_site_post):
    job = poet_single_site_get(api_session, STATIC.ssp_job_id)
    assert isinstance(job, PoetSSPJob)
    timeout = 600  # dev scales to 0 so slow
    while job.status != "SUCCESS":
        time.sleep(10)
        job = poet_single_site_get(
            api_session, job_id=STATIC.ssp_job_id
        )  # refresh results
        print(job.status)

        timeout += -10
        if timeout < 0:
            assert False, "timeout"

    assert len(job.result) > 2
    assert len(job.result[0].score) == 2  # 2 prompts
    # futures
    job = poet_single_site_get(api_session, STATIC.ssp_job_id)
    future = PoetSingleSiteFuture(api_session, job)
    assert future.wait_until_done(timeout=10)

    for results in [future.get(), future.wait()]:
        assert isinstance(results, dict)
        for k in list(results.keys()):
            assert len(results[k]) == 2  # 2 prompts= 2 scores


# @pytest.mark.skip("generate broken")
def test_poet_generate_get(api_session, test_poet_generate_post):
    f = load_job(api_session, STATIC.generate_job_id)
    job = f.job
    timeout = 600  # dev scales to 0 so slow
    while job.status != "SUCCESS":
        time.sleep(10)
        job = load_job(api_session, STATIC.generate_job_id)
        print(job.status)

        timeout += -10
        if timeout < 0:
            assert False, "timeout"

    response = poet_generate_get(api_session, STATIC.generate_job_id)
    assert response.status_code == 200
    assert response.raw is not None
    # futures
    f = load_job(api_session, STATIC.generate_job_id)
    job = f.job
    future = PoetGenerateFuture(api_session, job)
    future.refresh()
    assert future.wait_until_done(timeout=10)

    for results in [future.get(), future.wait()]:
        assert isinstance(results, list)
        for i in results:
            assert isinstance(i[0], str)
            assert isinstance(i[1], bytes)
            assert isinstance(i[2], np.ndarray)


def test_poet_model_interface(api_session, test_upload_single_prompt):
    seqs = [b"AAAPPLLLAAKAKAAA", b"IIGGGPPGGGGIIIILLAAA", b"ILKMEAPEAPEAPEA"]
    poet = api_session.embedding.get_model("poet")
    scorejob = poet.score(sequences=seqs, prompt=STATIC.prompt_single)
    embedjob = poet.embed(sequences=seqs, reduction="MEAN", prompt=STATIC.prompt_single)
    poetsvd = poet.fit_svd(prompt=STATIC.prompt_single, sequences=seqs)

    results = scorejob.wait()
    assert len(results) == len(seqs)
    assert isinstance(results, list)
    for i in results:
        assert all(isinstance(result[0], str) for result in results)
        assert all(isinstance(result[1], bytes) for result in results)
        assert all(isinstance(result[2], np.ndarray) for result in results)

    r = embedjob.wait()
    assert len(r) == len(seqs)
    for i in r:
        assert len(i) == 2
        assert isinstance(i[0], str)  # sequence
        assert isinstance(i[1], np.ndarray)
        assert i[1].shape == (1024,)

    r = poetsvd.wait()
    assert len(r) == len(seqs)
    for i in r:
        assert len(i) == 2
