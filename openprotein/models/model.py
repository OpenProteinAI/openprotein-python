"""Model handles for the models API."""

from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, ClassVar, cast

from openprotein.base import APISession
from openprotein.data.assaydataset import AssayDataset
from openprotein.jobs import _dispatch, futures
from openprotein.jobs.futures import (
    FitFuture,
    Future,
    _Submission,
)

from . import _api, _inputs
from .schemas import MethodDescriptor, ModelMetadata

if TYPE_CHECKING:
    pass

__all__ = ["Model"]


def _text(sequence: str | bytes, name: str = "sequence") -> str:
    if isinstance(sequence, bytes):
        try:
            return sequence.decode()
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} is not valid UTF-8 text") from exc
    return sequence


def _check_batch(sequences: Any) -> None:
    """Refuse a bare ``str`` or ``bytes`` where a list of sequences is expected."""
    if isinstance(sequences, (str, bytes)):
        raise TypeError(
            f"sequences must be a list of sequences, not a {type(sequences).__name__}; "
            'pass a list of sequences, e.g. ["ACD"]'
        )


def _texts(sequences: list[str] | list[bytes]) -> list[str]:
    """The batch as text, refusing a bare string or an entry that is not text."""
    _check_batch(sequences)
    texts: list[str] = []
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, (str, bytes)):
            raise TypeError(
                f"sequences[{index}] must be a str or bytes, "
                f"not {type(sequence).__name__}"
            )
        texts.append(_text(sequence, f"sequences[{index}]"))
    return texts


def _prepared(sequences: Any) -> Any:
    """``sequences`` with a batch of sequences as text; an assay or ``None`` is kept."""
    if sequences is None or isinstance(sequences, AssayDataset):
        return sequences
    return _texts(sequences)


#: The method a fitted SVD or UMAP serves, known before its fit has a card.
_PROJECTION = MethodDescriptor(path="batch/embeddings", outputs=["embeddings"])


def _project(
    fit: FitFuture, data: list[str] | list[bytes] | AssayDataset
) -> futures.EmbeddingFuture:
    """Submit a projection of sequences, or an assay's, through ``fit``'s model.

    The projection is submitted now and runs once the fit has finished.
    """
    sequences = (
        _inputs.assay_sequences(fit.session, data)
        if isinstance(data, AssayDataset)
        else _texts(data)
    )
    params: dict[str, Any] = {"sequences": sequences}
    model = f"{fit._family}/{fit.id}"
    future = _dispatch.create(
        fit.session,
        _api.run_method(fit.session, model, _PROJECTION.path, params),
        descriptor=_PROJECTION,
        submission=_Submission(model, _PROJECTION.path, params),
    )
    future.dataset = data if isinstance(data, AssayDataset) else None
    return cast(futures.EmbeddingFuture, future)


class Model:
    """A handle for submitting the methods listed on a model's card.

    Attributes
    ----------
    id : str
        The model identifier used to submit its methods.
    """

    #: whether ``batch/embeddings`` names an assay by ``assay_id``
    _embeds_assay_by_id: ClassVar[bool] = False
    #: the checkpoint a handle built without ``model_id`` uses
    _default_id: ClassVar[str | None] = None

    def __init__(
        self,
        session: APISession,
        model_id: str | None = None,
        metadata: ModelMetadata | None = None,
    ):
        if model_id is None:
            if self._default_id is None:
                raise TypeError(f"{type(self).__name__} needs a model_id")
            model_id = self._default_id
        self.session = session
        self.id = model_id
        self._metadata = metadata
        self._schemas: dict[str, dict[str, Any]] = {}

    @property
    def metadata(self) -> ModelMetadata:
        """The model card, loaded on first access."""
        if self._metadata is None:
            self._metadata = _api.get_model(self.session, self.id)
        return self._metadata

    @property
    def methods(self) -> list[MethodDescriptor]:
        """The method descriptors listed on the model card."""
        return self.metadata.methods

    def _params_schema(self, path: str) -> dict[str, Any]:
        """The JSON Schema of a method's params, read once per handle."""
        schema = self._schemas.get(path)
        if schema is None:
            schema = self._schemas[path] = _api.get_method_params(
                self.session, self.id, path
            )
        return schema

    def _method(self, verb: str, path: str) -> MethodDescriptor:
        """The card's first method matching ``path``, which may be a glob.

        Raises when the card lists none, so a verb can check before converting
        its inputs.
        """
        method = next((m for m in self.methods if fnmatchcase(m.path, path)), None)
        if method is None:
            paths = ", ".join(m.path for m in self.methods) or "none"
            raise ValueError(f"model {self.id!r} has no {verb}; its methods: {paths}")
        return method

    def _run(self, verb: str, path: str, **params: Any) -> Future:
        """Submit the card's first method matching ``path`` for ``verb``."""
        method = self._method(verb, path)
        job = _api.run_method(self.session, self.id, method.path, params)
        return _dispatch.create(
            self.session,
            job,
            descriptor=method,
            submission=_Submission(self.id, method.path, params),
        )

    def embed(
        self, sequences: list[str] | list[bytes] | AssayDataset, **params: Any
    ) -> futures.EmbeddingFuture:
        """Embed sequences, or an assay's, which stays on the future as ``dataset``.

        Submits ``batch/embeddings``.
        """
        if isinstance(sequences, AssayDataset) and self._embeds_assay_by_id:
            future = self._run(
                "embed", "batch/embeddings", assay_id=sequences.id, **params
            )
        elif isinstance(sequences, AssayDataset):
            texts = _inputs.assay_sequences(self.session, sequences)
            future = self._run("embed", "batch/embeddings", sequences=texts, **params)
        else:
            future = self._run(
                "embed", "batch/embeddings", sequences=_texts(sequences), **params
            )
        future.dataset = sequences if isinstance(sequences, AssayDataset) else None
        return cast(futures.EmbeddingFuture, future)

    def logits(
        self, sequences: list[str] | list[bytes], **params: Any
    ) -> futures.LogitsFuture:
        """Per-residue logits for sequences (``batch/logits``)."""
        return cast(
            futures.LogitsFuture,
            self._run("logits", "batch/logits", sequences=_texts(sequences), **params),
        )

    def attn(
        self, sequences: list[str] | list[bytes], **params: Any
    ) -> futures.AttentionFuture:
        """Attention maps for sequences (``batch/attn``)."""
        return cast(
            futures.AttentionFuture,
            self._run("attn", "batch/attn", sequences=_texts(sequences), **params),
        )

    def score(
        self, sequences: list[str] | list[bytes], **params: Any
    ) -> futures.LogLikelihoodFuture:
        """Log-likelihoods of sequences (``batch/loglikelihood``)."""
        return cast(
            futures.LogLikelihoodFuture,
            self._run(
                "score", "batch/loglikelihood", sequences=_texts(sequences), **params
            ),
        )

    def single_site(
        self, base_sequence: str | bytes, **params: Any
    ) -> futures.SingleSiteScoreFuture | futures.SingleSitePredictionFuture | Future:
        """Every single-site mutant of ``base_sequence`` (``single-site/*``)."""
        return self._run(
            "single_site", "single-site/*", base_sequence=_text(base_sequence), **params
        )

    def indel(
        self, base_sequence: str | bytes, **params: Any
    ) -> futures.IndelScoreFuture:
        """Every insertion or deletion variant of ``base_sequence`` (``indel/*``)."""
        return cast(
            futures.IndelScoreFuture,
            self._run("indel", "indel/*", base_sequence=_text(base_sequence), **params),
        )

    def generate(
        self, **params: Any
    ) -> futures.SequenceGenerationFuture | futures.StructureDesignFuture | Future:
        """Generate sequences or structures (``generate/*``)."""
        return self._run("generate", "generate/*", **params)

    def fold(
        self, complexes: list[Any], **params: Any
    ) -> futures.StructurePredictionFuture:
        """Predict structures of complexes (``batch/structures``)."""
        return cast(
            futures.StructurePredictionFuture,
            self._run("fold", "batch/structures", complexes=complexes, **params),
        )

    def predict(
        self, sequences: list[str] | list[bytes], **params: Any
    ) -> futures.PredictionFuture:
        """Predict properties of sequences (``batch/predictions``)."""
        return cast(
            futures.PredictionFuture,
            self._run(
                "predict", "batch/predictions", sequences=_texts(sequences), **params
            ),
        )

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return f"Model({self.id!r})"
