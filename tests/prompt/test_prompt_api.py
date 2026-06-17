import io
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openprotein.errors import APIError, InvalidParameterError
from openprotein.molecules import Complex, Protein
from openprotein.molecules.chains import DNA
from openprotein.prompt.api import (
    _assert_protein_only,
    _coerce_sequence,
    create_prompt,
    create_query,
    get_prompt,
    get_prompt_metadata,
    get_query,
    get_query_metadata,
    list_prompts,
    unzip_prompt,
    zip_prompt,
)
from openprotein.prompt.schemas import PromptMetadata, QueryMetadata

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def single_chain_protein_with_structure() -> Protein:
    c = Complex.from_filepath(str(DATA_DIR / "test_protein.cif"), verbose=False)
    return next(iter(c.get_proteins().values()))


@pytest.fixture(scope="module")
def multichain_complex_with_structure(
    single_chain_protein_with_structure: Protein,
) -> Complex:
    c2 = Complex.from_filepath(str(DATA_DIR / "9bkq-assembly2.cif"), verbose=False)
    other = next(iter(c2.get_proteins().values()))
    return Complex(
        {"A": single_chain_protein_with_structure, "B": other}, name="duo"
    )


def test_create_prompt(mock_session: MagicMock):
    """Test the create_prompt function."""
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "id": "prompt-123",
        "name": "Test Prompt",
        "created_date": datetime.now().isoformat(),
        "num_replicates": 1,
        "status": "SUCCESS",
    }

    context = ["ACGT"]
    metadata = create_prompt(mock_session, context, name="Test Prompt")

    mock_session.post.assert_called_once()
    assert isinstance(metadata, PromptMetadata)
    assert metadata.id == "prompt-123"


def test_get_prompt_metadata(mock_session: MagicMock):
    """Test the get_prompt_metadata function."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = {
        "id": "prompt-123",
        "name": "Test Prompt",
        "created_date": datetime.now().isoformat(),
        "num_replicates": 1,
        "status": "SUCCESS",
    }

    metadata = get_prompt_metadata(mock_session, "prompt-123")

    mock_session.get.assert_called_with("v1/prompt/prompt-123")
    assert isinstance(metadata, PromptMetadata)
    assert metadata.id == "prompt-123"


def test_list_prompts(mock_session: MagicMock):
    """Test the list_prompts function."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = [
        {
            "id": "prompt-123",
            "name": "Test Prompt 1",
            "created_date": datetime.now().isoformat(),
            "num_replicates": 1,
            "status": "SUCCESS",
        },
        {
            "id": "prompt-456",
            "name": "Test Prompt 2",
            "created_date": datetime.now().isoformat(),
            "num_replicates": 1,
            "status": "PENDING",
        },
    ]

    prompts = list_prompts(mock_session)

    mock_session.get.assert_called_with("v1/prompt")
    assert len(prompts) == 2
    assert all(isinstance(p, PromptMetadata) for p in prompts)


def test_create_query(mock_session: MagicMock):
    """Test the create_query function."""
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "id": "query-123",
        "created_date": datetime.now().isoformat(),
    }

    query = "ACGT"
    metadata = create_query(mock_session, query)

    mock_session.post.assert_called_once()
    assert isinstance(metadata, QueryMetadata)
    assert metadata.id == "query-123"


def test_get_query_metadata(mock_session: MagicMock):
    """Test the get_query_metadata function."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = {
        "id": "query-123",
        "created_date": datetime.now().isoformat(),
    }

    metadata = get_query_metadata(mock_session, "query-123")

    mock_session.get.assert_called_with("v1/prompt/query/query-123")
    assert isinstance(metadata, QueryMetadata)
    assert metadata.id == "query-123"


def test_api_error_handling(mock_session: MagicMock):
    """Test that APIError is raised for non-200 status codes."""
    mock_session.get.return_value.status_code = 404
    mock_session.get.return_value.json.return_value = {"detail": "Not Found"}

    with pytest.raises(APIError, match="Not Found"):
        get_prompt_metadata(mock_session, "non-existent-id")


def test_invalid_parameter_error(mock_session: MagicMock):
    """Test that InvalidParameterError is raised for 400 status code."""
    mock_session.post.return_value.status_code = 400
    mock_session.post.return_value.json.return_value = {"detail": "Invalid parameters"}

    with pytest.raises(InvalidParameterError, match="Invalid parameters"):
        create_prompt(mock_session, context=["ACGT"], name="Invalid Prompt")


# ---------- multichain support ----------


class TestCoerceSequence:
    def test_single_chain_returns_protein(self):
        result = _coerce_sequence("foo", "ACDE")
        assert isinstance(result, Protein)
        assert result.name == "foo"
        assert result.sequence == b"ACDE"

    def test_multichain_returns_complex(self):
        result = _coerce_sequence("foo", "ACDE:GHIK")
        assert isinstance(result, Complex)
        assert result.name == "foo"
        proteins = result.get_proteins()
        assert len(proteins) == 2
        seqs = [p.sequence for p in proteins.values()]
        assert b"ACDE" in seqs and b"GHIK" in seqs

    def test_bytes_input(self):
        result = _coerce_sequence("foo", b"ACDE:GHIK")
        assert isinstance(result, Complex)

    @pytest.mark.parametrize("bad", ["ACDE::GHIK", ":ACDE", "ACDE:", ":"])
    def test_invalid_chain_break_raises(self, bad: str):
        with pytest.raises(InvalidParameterError, match="chain break"):
            _coerce_sequence("foo", bad)

    @pytest.mark.parametrize(
        "sequence,n_chains",
        [("A:B:C", 3), ("ACDE:GHIK:LMNP:QRST", 4), ("A:B:C:D:E", 5)],
    )
    def test_n_chain_split(self, sequence: str, n_chains: int):
        result = _coerce_sequence("foo", sequence)
        assert isinstance(result, Complex)
        assert len(result.get_proteins()) == n_chains


class TestAssertProteinOnly:
    def test_protein_only_complex_passes(self):
        c = Complex({"A": Protein(sequence="ACDE"), "B": Protein(sequence="GHIK")})
        _assert_protein_only(c)  # should not raise

    def test_complex_with_dna_raises(self):
        c = Complex({"A": Protein(sequence="ACDE"), "B": DNA(sequence="ACGT")})
        with pytest.raises(InvalidParameterError, match="protein chains"):
            _assert_protein_only(c)


class TestZipUnzipRoundtrip:
    def test_single_chain_protein_returns_protein(self):
        context = [Protein(name="p1", sequence=b"ACDEFGHIK")]
        zips = zip_prompt(context)
        outer = self._wrap(zips)
        result = unzip_prompt(outer)
        assert len(result) == 1 and len(result[0]) == 1
        entry = result[0][0]
        assert isinstance(entry, Protein)
        assert entry.sequence == b"ACDEFGHIK"

    def test_multichain_complex_sequence_only_returns_complex(self):
        c = Complex(
            {"A": Protein(sequence=b"ACDE"), "B": Protein(sequence=b"GHIK")},
            name="duo",
        )
        zips = zip_prompt([c])
        # confirm wire format took the FASTA path (no .cif file inside)
        import zipfile

        zf = zipfile.ZipFile(zips[0])
        names = zf.namelist()
        assert all(n.endswith(".fasta") for n in names)
        body = zf.read(names[0])
        assert b":" in body  # multichain delimiter is preserved on the wire

        outer = self._wrap(zips)
        result = unzip_prompt(outer)
        entry = result[0][0]
        assert isinstance(entry, Complex)
        proteins = entry.get_proteins()
        assert len(proteins) == 2
        assert sorted(p.sequence for p in proteins.values()) == [b"ACDE", b"GHIK"]

    def test_raw_string_with_colon_returns_complex(self):
        zips = zip_prompt(["ACDE:GHIK"])
        outer = self._wrap(zips)
        result = unzip_prompt(outer)
        entry = result[0][0]
        assert isinstance(entry, Complex)
        assert len(entry.get_proteins()) == 2

    def test_complex_with_dna_rejected_clientside(self, mock_session: MagicMock):
        c = Complex({"A": Protein(sequence="ACDE"), "B": DNA(sequence="ACGT")})
        with pytest.raises(InvalidParameterError, match="protein chains"):
            zip_prompt([c])
        with pytest.raises(InvalidParameterError, match="protein chains"):
            create_prompt(mock_session, context=[c])
        mock_session.post.assert_not_called()

    def test_single_chain_cif_collapses_to_protein(
        self, single_chain_protein_with_structure: Protein
    ):
        """A single-chain Protein with structure writes CIF and collapses back to Protein."""
        import zipfile

        protein = single_chain_protein_with_structure
        zips = zip_prompt([protein])
        zf = zipfile.ZipFile(zips[0])
        names = zf.namelist()
        assert all(n.endswith(".cif") for n in names)

        outer = self._wrap(zips)
        result = unzip_prompt(outer)
        entry = result[0][0]
        assert isinstance(entry, Protein)
        assert entry.sequence == protein.sequence
        assert entry.has_structure

    def test_multichain_cif_preserves_complex(
        self, multichain_complex_with_structure: Complex
    ):
        """A multichain Complex with structure round-trips as a Complex."""
        import zipfile

        multi = multichain_complex_with_structure
        zips = zip_prompt([multi])
        zf = zipfile.ZipFile(zips[0])
        names = zf.namelist()
        assert all(n.endswith(".cif") for n in names)

        outer = self._wrap(zips)
        result = unzip_prompt(outer)
        entry = result[0][0]
        assert isinstance(entry, Complex)
        proteins = entry.get_proteins()
        assert len(proteins) == 2
        round_tripped_seqs = sorted(p.sequence for p in proteins.values())
        original_seqs = sorted(p.sequence for p in multi.get_proteins().values())
        assert round_tripped_seqs == original_seqs

    def test_unrecognized_extension_raises(self):
        """A zip with an unrecognized inner file extension surfaces an error
        instead of silently dropping the entry."""
        import zipfile

        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("000000.foo.txt", b"not a real entry")
        outer = self._wrap([inner])
        with pytest.raises(APIError, match="Unrecognized prompt context file"):
            unzip_prompt(outer)

    @staticmethod
    def _wrap(context_zips: list[io.BytesIO]) -> io.BytesIO:
        """Wrap per-context zips into the outer prompt zip format unzip_prompt expects."""
        import zipfile

        outer = io.BytesIO()
        with zipfile.ZipFile(outer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, inner in enumerate(context_zips):
                inner.seek(0)
                zf.writestr(f"context-{i}.zip", inner.read())
        outer.seek(0)
        return outer


class TestCreateQueryMultichain:
    def test_multichain_string_creates_query_via_fasta(self, mock_session: MagicMock):
        mock_session.post.return_value.status_code = 200
        mock_session.post.return_value.json.return_value = {
            "id": "query-456",
            "created_date": datetime.now().isoformat(),
        }
        create_query(mock_session, "ACDE:GHIK")
        call = mock_session.post.call_args
        files = call.kwargs["files"]
        filename, body, mimetype = files["query"]
        assert filename == "query.fasta"
        assert mimetype == "text/x-fasta"
        body.seek(0)
        content = body.read()
        assert b"ACDE:GHIK" in content

    def test_complex_with_dna_rejected_clientside(self, mock_session: MagicMock):
        c = Complex({"A": Protein(sequence="ACDE"), "B": DNA(sequence="ACGT")})
        with pytest.raises(InvalidParameterError, match="protein chains"):
            create_query(mock_session, c)
        mock_session.post.assert_not_called()

    def test_protein_no_structure_sends_fasta(self, mock_session: MagicMock):
        mock_session.post.return_value.status_code = 200
        mock_session.post.return_value.json.return_value = {
            "id": "query-789",
            "created_date": datetime.now().isoformat(),
        }
        create_query(mock_session, Protein(name="p1", sequence=b"ACDEFGHIK"))
        files = mock_session.post.call_args.kwargs["files"]
        filename, body, mimetype = files["query"]
        assert filename == "query.fasta"
        assert mimetype == "text/x-fasta"
        body.seek(0)
        content = body.read()
        assert b"ACDEFGHIK" in content

    def test_unnamed_protein_no_structure_sends_fasta(self, mock_session: MagicMock):
        """A Protein created without a name (e.g. ``Protein("ARN")``) must be
        accepted and serialized with a default fasta header."""
        mock_session.post.return_value.status_code = 200
        mock_session.post.return_value.json.return_value = {
            "id": "query-789",
            "created_date": datetime.now().isoformat(),
        }
        create_query(mock_session, Protein("ARN"))
        files = mock_session.post.call_args.kwargs["files"]
        filename, body, mimetype = files["query"]
        assert filename == "query.fasta"
        assert mimetype == "text/x-fasta"
        body.seek(0)
        content = body.read()
        assert content == b">query\nARN\n"

    def test_protein_force_structure_sends_cif(
        self,
        mock_session: MagicMock,
        single_chain_protein_with_structure: Protein,
    ):
        mock_session.post.return_value.status_code = 200
        mock_session.post.return_value.json.return_value = {
            "id": "query-789",
            "created_date": datetime.now().isoformat(),
        }
        create_query(mock_session, single_chain_protein_with_structure)
        files = mock_session.post.call_args.kwargs["files"]
        filename, _body, mimetype = files["query"]
        assert filename == "query.cif"
        assert mimetype == "chemical/x-mmcif"

    def test_protein_force_structure_true_promotes_to_cif(
        self, mock_session: MagicMock
    ):
        """force_structure=True writes CIF even when the Protein has no
        coordinates (used for structure-prediction queries)."""
        mock_session.post.return_value.status_code = 200
        mock_session.post.return_value.json.return_value = {
            "id": "query-789",
            "created_date": datetime.now().isoformat(),
        }
        p = Protein(name="p1", sequence=b"ACDEFGHIK")
        create_query(mock_session, p, force_structure=True)
        files = mock_session.post.call_args.kwargs["files"]
        filename, _body, mimetype = files["query"]
        assert filename == "query.cif"
        assert mimetype == "chemical/x-mmcif"


class TestGetQueryMultichain:
    def test_multichain_fasta_response_returns_complex(self, mock_session: MagicMock):
        """get_query must parse ':' chain breaks in FASTA responses into a Complex.

        Regression: previously this branch called ``Protein(name, sequence)``
        directly, which raised AssertionError because ':' isn't a valid amino
        acid character.
        """
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.content = b">my-query\nACDE:GHIK\n"
        mock_session.get.return_value.headers = {
            "Content-Disposition": "attachment; filename=query.fasta",
            "Content-Type": "text/x-fasta",
        }
        result = get_query(mock_session, "query-789")
        assert isinstance(result, Complex)
        seqs = sorted(p.sequence for p in result.get_proteins().values())
        assert seqs == [b"ACDE", b"GHIK"]

    def test_single_chain_fasta_response_returns_protein(
        self, mock_session: MagicMock
    ):
        """Single-chain FASTA responses still collapse to Protein."""
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.content = b">my-query\nACDEFGHIK\n"
        mock_session.get.return_value.headers = {
            "Content-Disposition": "attachment; filename=query.fasta",
            "Content-Type": "text/x-fasta",
        }
        result = get_query(mock_session, "query-789")
        assert isinstance(result, Protein)
        assert result.sequence == b"ACDEFGHIK"

    @pytest.mark.parametrize("status_code,detail", [(401, "Unauthorized"), (404, "Not Found")])
    def test_error_status_surfaces_detail_not_file_format_error(
        self, mock_session: MagicMock, status_code: int, detail: str
    ):
        """Regression: error responses must dispatch on status code before parsing
        the body as a query file. Previously this raised an opaque
        ``APIError("Unexpected file returned…")`` for 401/404 because file-format
        parsing ran first."""
        mock_session.get.return_value.status_code = status_code
        mock_session.get.return_value.json.return_value = {"detail": detail}
        mock_session.get.return_value.headers = {
            "Content-Disposition": "",
            "Content-Type": "text/plain",
        }
        mock_session.get.return_value.content = b"some error body"
        with pytest.raises(APIError, match=detail):
            get_query(mock_session, "query-789")

    def test_multichain_cif_response_returns_complex(
        self,
        mock_session: MagicMock,
        multichain_complex_with_structure: Complex,
    ):
        """A multichain CIF response is parsed as a Complex."""
        cif_bytes = multichain_complex_with_structure.to_string("cif").encode()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.content = cif_bytes
        mock_session.get.return_value.headers = {
            "Content-Disposition": "attachment; filename=query.cif",
            "Content-Type": "chemical/x-mmcif",
        }
        result = get_query(mock_session, "query-789")
        assert isinstance(result, Complex)
        assert len(result.get_proteins()) == 2

    def test_single_chain_cif_response_returns_protein(
        self,
        mock_session: MagicMock,
        single_chain_protein_with_structure: Protein,
    ):
        """A single-chain CIF response collapses to Protein."""
        cif_bytes = single_chain_protein_with_structure.to_string().encode()
        mock_session.get.return_value.status_code = 200
        mock_session.get.return_value.content = cif_bytes
        mock_session.get.return_value.headers = {
            "Content-Disposition": "attachment; filename=query.cif",
            "Content-Type": "chemical/x-mmcif",
        }
        result = get_query(mock_session, "query-789")
        assert isinstance(result, Protein)


class TestParseFastaAsProteins:
    def test_emits_future_warning_on_single_chain(self, tmp_path: Path):
        path = tmp_path / "single.fasta"
        path.write_bytes(b">p1\nACDEFGHIK\n")
        from openprotein.molecules.protein import parse_fasta_as_proteins

        with pytest.warns(FutureWarning, match="deprecated"):
            proteins = parse_fasta_as_proteins(path)
        assert len(proteins) == 1
        assert proteins[0].sequence == b"ACDEFGHIK"

    def test_raises_on_multichain(self, tmp_path: Path):
        path = tmp_path / "multi.fasta"
        path.write_bytes(b">p1\nACDE:GHIK\n")
        from openprotein.molecules.protein import parse_fasta_as_proteins

        with pytest.warns(FutureWarning):
            with pytest.raises(ValueError, match="chain break"):
                parse_fasta_as_proteins(path)
