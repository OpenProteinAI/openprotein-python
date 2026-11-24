import warnings
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from openprotein.align.api import (
    _msa_mmseqs_search_post,
    _sample_prompt,
    abnumber_post,
    antibody_schema_get,
    clustalo_post,
    get_align_job_inputs,
    get_input,
    get_msa,
    get_seed,
    mafft_post,
    msa_post,
    prompt_post,
)
from openprotein.align.schemas import (
    AlignType,
    Pairing,
    UnnumberedPolicy,
)
from openprotein.errors import HTTPError, InvalidParameterError, MissingParameterError
from openprotein.jobs import Job, JobStatus, JobType
from openprotein.prompt.schemas import PromptMetadata


@pytest.fixture
def sample_job_dict() -> dict:
    """Fixture for a sample job dictionary response."""
    return {
        "job_id": "job123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.align_align,
        "created_date": "2023-01-01T00:00:00",
        "last_update": "2023-01-01T00:00:00",
    }


def test_get_align_job_inputs(mock_session: MagicMock):
    """Test get_align_job_inputs calls the correct endpoint and params."""
    mock_session.get.return_value = MagicMock()

    with pytest.warns(FutureWarning):
        get_align_job_inputs(mock_session, "job1", AlignType.MSA)
    mock_session.get.assert_called_with(
        "v1/align/inputs",
        params={"job_id": "job1", "msa_type": "GENERATED"},
        stream=True,
    )

    with pytest.warns(FutureWarning):
        get_align_job_inputs(mock_session, "job1", AlignType.PROMPT, prompt_index=1)
    mock_session.get.assert_called_with(
        "v1/align/inputs",
        params={"job_id": "job1", "msa_type": "PROMPT", "replicate": 1},
        stream=True,
    )


@patch("openprotein.align.api._get_align_job_inputs")
@patch("openprotein.csv.parse_stream")
def test_get_input(mock_csv_stream, mock_get_align_job_inputs, mock_session: MagicMock):
    """Test get_input correctly calls dependencies."""
    with pytest.warns(FutureWarning):
        get_input(mock_session, "job1", AlignType.MSA)
    mock_get_align_job_inputs.assert_called_once_with(
        session=mock_session, job_id="job1", input_type=AlignType.MSA, prompt_index=None
    )
    mock_csv_stream.assert_called_once()


def test_get_msa_surfaces_incomplete_job_error(mock_session: MagicMock):
    """Reading a stored MSA whose job has not finished raises an HTTPError.

    The error carries the 409 status code and text naming the MSA id, so the
    user can tell the job is still pending.
    """
    response = MagicMock()
    response.status_code = 409
    response.url = "https://api.openprotein.ai/api/v1/data/msa/abc"
    response.text = '{"detail":"msa abc is PENDING"}'
    job = MagicMock(status_code=200)
    job.json.return_value = {
        "job_id": "abc",
        "job_type": "/align/align",
        "status": "SUCCESS",
        "created_date": datetime(2023, 1, 1).isoformat(),
    }

    def get(endpoint, **kwargs):
        if endpoint.startswith("v1/data/msa/"):
            raise HTTPError(response)
        return job

    mock_session.get.side_effect = get

    with pytest.raises(HTTPError) as excinfo:
        list(get_msa(mock_session, "abc"))

    assert excinfo.value.status_code == 409
    assert "abc" in excinfo.value.text


def test_get_seed(mock_session: MagicMock):
    """get_seed calls GET v1/align/seed?msa_id=... and returns the sequence."""
    mock_session.get.return_value.json.return_value = {
        "sequence": "ACDEFGHIK",
        "length": 9,
    }

    assert get_seed(mock_session, "job123") == "ACDEFGHIK"

    args, kwargs = mock_session.get.call_args
    assert args[0] == "v1/align/seed"
    assert kwargs["params"] == {"msa_id": "job123"}


def test_msa_mmseqs_search_post(mock_session: MagicMock, sample_job_dict: dict):
    """create_msa's homology search posts the seed to the mmseqs2 search endpoint."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    job = _msa_mmseqs_search_post(mock_session, seed="ACGT")

    args, kwargs = mock_session.post.call_args
    assert args[0] == "v1/align/search/mmseqs2/msa"
    assert kwargs["data"] == {"query": "ACGT", "pairing": "both"}
    assert "files" not in kwargs
    assert isinstance(job, Job)


def test_msa_mmseqs_search_post_pairing(mock_session: MagicMock, sample_job_dict: dict):
    """A non-default pairing mode is sent as the pairing form field."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    _msa_mmseqs_search_post(mock_session, seed="ACGT", pairing=Pairing.PAIRED)

    _, kwargs = mock_session.post.call_args
    assert kwargs["data"] == {"query": "ACGT", "pairing": "paired"}


@pytest.mark.parametrize(
    ("post", "option", "value"),
    [
        (mafft_post, "auto", False),
        (clustalo_post, "clustersize", 50),
        (abnumber_post, "scheme", "kabat"),
    ],
)
@pytest.mark.parametrize("positional", [False, True])
def test_align_post_accepts_sequence_files(
    mock_session, sample_job_dict, post, option, value, positional
):
    mock_session.post.return_value.json.return_value = sample_job_dict
    file = BytesIO(b">first\nACDE\n>second\nACDF\n")

    with pytest.warns(FutureWarning, match=r"0\.18\.0.*0\.19\.0"):
        job = (
            post(mock_session, file, value)
            if positional
            else post(mock_session, sequence_file=file, **{option: value})
        )

    assert job.job_id == "job123"
    assert mock_session.post.call_args.kwargs["json"] == {
        "sequences": ["ACDE", "ACDF"],
        "names": ["first", "second"],
    }
    assert mock_session.post.call_args.kwargs["params"][option] == value


def test_mafft_post(mock_session: MagicMock, sample_job_dict: dict):
    """Test mafft_post."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    sequences = ["ACGT", "ACGA"]
    mafft_post(mock_session, sequences, auto=False, ep=0.1, op=1.5)
    mock_session.post.assert_called_once_with(
        "v1/align/mafft",
        json={"sequences": sequences},
        params={"auto": False, "ep": 0.1, "op": 1.5},
    )


def test_mafft_post_carries_names(mock_session: MagicMock, sample_job_dict: dict):
    """Row labels ride alongside the sequences when the caller supplies them."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    sequences = ["ACGT", "ACGA"]
    names = ["a", "b"]
    mafft_post(mock_session, sequences, names=names)

    _args, kwargs = mock_session.post.call_args
    assert kwargs["json"] == {"sequences": sequences, "names": names}


def test_mafft_post_omits_names_when_absent(
    mock_session: MagicMock, sample_job_dict: dict
):
    """No names means no key, so the server applies its own labelling."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    mafft_post(mock_session, ["ACGT", "ACGA"])

    _args, kwargs = mock_session.post.call_args
    assert "names" not in kwargs["json"]


def test_clustalo_post(mock_session: MagicMock, sample_job_dict: dict):
    """Test clustalo_post."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    sequences = ["ACGT", "ACGA"]
    clustalo_post(mock_session, sequences, clustersize=50, iterations=2)
    mock_session.post.assert_called_once_with(
        "v1/align/clustalo",
        json={"sequences": sequences},
        params={"clustersize": 50, "iterations": 2},
    )


def test_clustalo_post_carries_names(mock_session: MagicMock, sample_job_dict: dict):
    """Row labels ride alongside the sequences when the caller supplies them."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    sequences = ["ACGT", "ACGA"]
    clustalo_post(mock_session, sequences, names=["a", "b"])

    _args, kwargs = mock_session.post.call_args
    assert kwargs["json"] == {"sequences": sequences, "names": ["a", "b"]}


SEQUENCES = ["QVQLQESGPG", "DIVMTQSPAS"]


def test_abnumber_post_invalid_scheme(mock_session: MagicMock):
    """Test abnumber_post raises error for invalid scheme."""
    with pytest.raises(
        Exception, match="Antibody numbering invalid_scheme not recognized"
    ):
        abnumber_post(mock_session, SEQUENCES, scheme="invalid_scheme")


def test_abnumber_post_sends_sequences_as_the_body(
    mock_session: MagicMock, sample_job_dict: dict
):
    """The sequences are the request body, not a file upload."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    abnumber_post(mock_session, SEQUENCES, names=["h", "l"])

    _args, kwargs = mock_session.post.call_args
    assert kwargs["json"] == {"sequences": SEQUENCES, "names": ["h", "l"]}


def test_abnumber_post_default_drop_minority_chains(
    mock_session: MagicMock, sample_job_dict: dict
):
    """abnumber_post should default drop_minority_chains to False."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    abnumber_post(mock_session, SEQUENCES)

    args, kwargs = mock_session.post.call_args
    assert args[0] == "v1/align/abnumber"
    assert kwargs["params"]["drop_minority_chains"] is False


def test_abnumber_post_with_drop_minority_chains(
    mock_session: MagicMock, sample_job_dict: dict
):
    """abnumber_post should forward drop_minority_chains=True."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    abnumber_post(mock_session, SEQUENCES, drop_minority_chains=True)

    args, kwargs = mock_session.post.call_args
    assert kwargs["params"]["drop_minority_chains"] is True


def test_abnumber_post_default_unnumbered_policy(
    mock_session: MagicMock, sample_job_dict: dict
):
    """abnumber_post should default unnumbered_policy to error, so dropping
    residues is always a choice the caller made."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    abnumber_post(mock_session, SEQUENCES)

    _args, kwargs = mock_session.post.call_args
    assert kwargs["params"]["unnumbered_policy"] == "error"


@pytest.mark.parametrize("policy", ["drop", "retain"])
def test_abnumber_post_forwards_unnumbered_policy(
    mock_session: MagicMock, sample_job_dict: dict, policy: str
):
    """abnumber_post should forward the policy as its wire value."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    abnumber_post(mock_session, SEQUENCES, unnumbered_policy=policy)

    _args, kwargs = mock_session.post.call_args
    assert kwargs["params"]["unnumbered_policy"] == policy


def test_abnumber_post_accepts_policy_enum(
    mock_session: MagicMock, sample_job_dict: dict
):
    """The enum and its string value are interchangeable on the wire."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    abnumber_post(mock_session, SEQUENCES, unnumbered_policy=UnnumberedPolicy.RETAIN)

    _args, kwargs = mock_session.post.call_args
    assert kwargs["params"]["unnumbered_policy"] == "retain"


def test_abnumber_post_invalid_unnumbered_policy(mock_session: MagicMock):
    """abnumber_post rejects an unknown policy before submitting."""
    with pytest.raises(
        Exception, match="Unnumbered residue policy maybe not recognized"
    ):
        abnumber_post(mock_session, SEQUENCES, unnumbered_policy="maybe")


def test_antibody_schema_get(mock_session: MagicMock):
    """antibody_schema_get returns the job's extras as sent."""
    extras = {"chains": [{"chain_type": "H"}], "warnings": []}
    mock_session.get.return_value.json.return_value = extras

    assert antibody_schema_get(mock_session, "job123") == extras

    args, kwargs = mock_session.get.call_args
    assert args[0] == "v1/align/antibody_schema"
    assert kwargs["params"] == {"job_id": "job123"}


@pytest.fixture
def sample_prompt_metadata() -> dict:
    """A PENDING prompt-metadata response from sampling an MSA."""
    return {
        "id": "prompt123",
        "name": "prompt123",
        "description": None,
        "created_date": "2023-01-01T00:00:00",
        "num_replicates": 1,
        "job_id": "prompt123",
        "status": JobStatus.PENDING,
    }


def test_prompt_post_warns_and_samples_through_the_align_route(
    mock_session: MagicMock, sample_job_dict: dict
):
    mock_session.post.return_value.json.return_value = sample_job_dict

    with pytest.warns(FutureWarning) as record:
        job = prompt_post(mock_session, "msa123", num_sequences=10)

    assert str(record[0].message) == (
        "`openprotein.align.api.prompt_post` is deprecated since 0.18.0 and will be "
        "removed in 0.19.0. Use `MSAFuture.sample_prompt` or `MSA.sample_prompt`."
    )
    assert record[0].filename == __file__
    args, kwargs = mock_session.post.call_args
    assert args[0] == "v1/align/prompt"
    assert kwargs["params"]["msa_id"] == "msa123"
    assert kwargs["params"]["max_msa_sequences"] == 10
    assert isinstance(job, Job)


def test_sampling_a_prompt_returns_its_metadata(
    mock_session: MagicMock, sample_prompt_metadata: dict
):
    mock_session.post.return_value.json.return_value = sample_prompt_metadata

    metadata = _sample_prompt(mock_session, "msa123", num_sequences=10)

    args, kwargs = mock_session.post.call_args
    assert args[0] == "v1/prompt/sample_msa"
    assert kwargs["params"]["msa_id"] == "msa123"
    assert kwargs["params"]["max_msa_sequences"] == 10
    assert isinstance(metadata, PromptMetadata)


def test_sampling_without_a_size_takes_the_default_token_budget(
    mock_session: MagicMock, sample_prompt_metadata: dict
):
    mock_session.post.return_value.json.return_value = sample_prompt_metadata
    _sample_prompt(mock_session, "msa123")
    assert mock_session.post.call_args.kwargs["params"]["max_msa_tokens"] == 12288


def test_sampling_refuses_invalid_params_before_any_request(mock_session: MagicMock):
    with pytest.raises(MissingParameterError):
        _sample_prompt(mock_session, "msa123", num_sequences=10, num_residues=100)
    with pytest.raises(InvalidParameterError):
        _sample_prompt(mock_session, "msa123", homology_level=2.0)
    mock_session.post.assert_not_called()


def test_msa_post_warns_and_posts_the_file_as_a_ready_made_msa(
    mock_session: MagicMock, sample_job_dict: dict
):
    mock_session.post.return_value.json.return_value = sample_job_dict
    msa_file = MagicMock()

    with pytest.warns(FutureWarning) as record:
        job = msa_post(mock_session, msa_file)

    expected = (
        "`openprotein.align.api.msa_post` is deprecated since 0.18.0 and will be "
        "removed in 0.19.0. Use `session.data.MSA.create` for a ready-made MSA, "
        "or `session.align.create_msa` for a seed."
    )
    assert [str(w.message) for w in record] == [expected]
    mock_session.post.assert_called_once_with(
        "v1/align/msa", files={"msa_file": msa_file}, params={"is_seed": False}
    )
    assert job.job_id == "job123"


@pytest.mark.parametrize("seed", ["ACDE", b"ACDE"])
def test_msa_post_posts_a_seed_as_the_seed_file(
    mock_session: MagicMock, sample_job_dict: dict, seed
):
    mock_session.post.return_value.json.return_value = sample_job_dict

    with pytest.warns(FutureWarning) as record:
        job = msa_post(mock_session, seed=seed)

    assert len(record) == 1
    args, kwargs = mock_session.post.call_args
    assert args[0] == "v1/align/msa"
    assert kwargs["params"] == {"is_seed": True}
    assert kwargs["files"]["msa_file"].read() == b">seed\nACDE"
    assert job.job_id == "job123"


def test_msa_post_requires_exactly_one_of_msa_file_and_seed(mock_session: MagicMock):
    with (
        pytest.warns(FutureWarning) as record,
        pytest.raises(MissingParameterError, match="seed OR msa_file"),
    ):
        msa_post(mock_session)
    assert len(record) == 1

    with (
        pytest.warns(FutureWarning) as record,
        pytest.raises(MissingParameterError, match="seed OR msa_file"),
    ):
        msa_post(mock_session, MagicMock(), seed="ACDE")
    assert len(record) == 1
    mock_session.post.assert_not_called()


def test_a_tuple_of_sequences_is_not_taken_for_a_sequence_file(
    mock_session: MagicMock, sample_job_dict: dict
):
    """A tuple of strings is a batch of sequences and sends with no warning."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mafft_post(mock_session, ("ACDE", "ACDF"))
    assert mock_session.post.call_args.kwargs["json"] == {"sequences": ("ACDE", "ACDF")}


def test_a_named_file_tuple_is_a_sequence_file(
    mock_session: MagicMock, sample_job_dict: dict
):
    """A ``(name, file)`` tuple reads as a sequence file."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    file = ("seqs.fasta", BytesIO(b">first\nACDE\n>second\nACDF\n"))
    with pytest.warns(FutureWarning, match=r"0\.18\.0"):
        mafft_post(mock_session, file)
    assert mock_session.post.call_args.kwargs["json"]["sequences"] == ["ACDE", "ACDF"]
