import pytest
from openprotein.api.align import *
import json
from tests.conf import BACKEND, TIMEOUT
import time
import collections
import openprotein
from openprotein.schemas import JobType
from openprotein.jobs import *
import numpy as np
from openprotein.api.poet import *
from openprotein.api.jobs import load_job
from AWSTools.Batchtools.batch_utils import fakeseq

TEST_SEQUENCE = f"{fakeseq(5)}MYRMQLLSCIALSLALVTNSAPTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGMYRMQLLSCIALSLALVTNSAPTSSSTKKTQLQLEHLLLDLQMILNGINNYKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGSEP"
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
    prompt_from_msa: str = None
    score_from_msa: str = None


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
def test_prompt_from_msa(api_session, test_msa_post):
    random_seq = f"{fakeseq(5)}{TEST_SEQUENCE}"
    msa = msa_post(api_session, seed=random_seq.encode())
    msajob = msa.job
    print(f"MSA ID {msajob.job_id}")

    prompt = prompt_post(
        api_session,
        msa_id=msajob.job_id,
        num_sequences=10,
        min_similarity=0.1,
        num_ensemble_prompts=3,
    )
    job = prompt.job

    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    assert msa.wait_until_done(timeout=TIMEOUT)
    assert prompt.wait_until_done(timeout=TIMEOUT)
    print(f"prompt ID {job.job_id}")

    if STATIC.prompt_from_msa is None:
        STATIC.prompt_from_msa = job.job_id
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


@pytest.mark.skip("Optional test for msa-prompt-score")
def test_poet_score_from_msaprompt(api_session, test_prompt_from_msa):
    pid = test_prompt_from_msa

    queries = [b"AAAPPPLLL", b"AAAAPPPLK"]
    score = poet_score_post(api_session, prompt_id=pid, queries=queries)
    job = score.job
    print("SCORE job", job.job_id)
    assert isinstance(job, PoetScoreJob)
    assert job.job_id is not None
    assert job.status in ["PENDING", "RUNNING", "SUCCESS"]
    assert "poet" in job.job_type
    if STATIC.score_from_msa is None:
        STATIC.score_from_msa = job.job_id
    assert score.wait_until_done(timeout=TIMEOUT)
    assert score.done()
    result = score.get()
    assert len(result) == 2  # 2 seq
    assert len(result[0][-1]) == 3  # 3 prompts
    assert len(result[1][-1]) == 3  # 3 prompts


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
    timeout = TIMEOUT  # dev scales to 0 so slow
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
    timeout = TIMEOUT  # dev scales to 0 so slow
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
    timeout = TIMEOUT  # dev scales to 0 so slow
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
    seqs = [b"LKLK", b"LKLA", b"LKLI"]
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
        assert isinstance(i[0], bytes)  # sequence
        assert isinstance(i[1], np.ndarray)
        assert i[1].shape == (1024,)

    svd_embed = poetsvd.embed(seqs)
    r = svd_embed.wait()
    assert len(r) == len(seqs)
    for i in r:
        assert len(i) == 2
