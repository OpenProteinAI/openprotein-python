"""The models API: discovery, method calls, and the job handle."""

import inspect
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
import requests

from openprotein.base import APISession
from openprotein.errors import APIError
from openprotein.jobs import Future, Job, JobStatus, _dispatch
from openprotein.jobs import futures as results
from openprotein.jobs.futures import SVDFitFuture, UMAPFitFuture
from openprotein.models import (
    MethodDescriptor,
    Model,
    ModelMetadata,
    ModelsAPI,
)
from openprotein.models import _api as models_api

MODEL_ID = "esm2_t33_650M_UR50D"

METADATA: dict[str, Any] = {
    "model_id": MODEL_ID,
    "type": None,
    "family_id": "esm",
    "model": "ESM2",
    "name": "ESM2 (650M)",
    "variant": "650M",
    "is_system": True,
    "base": True,
    "tags": None,
    "input_types": ["protein"],
    "description": {
        "summary": "ESM-2 650M protein language model.",
        "citation_title": None,
        "doi": "10.1101/2022.07.20.500902",
    },
    "dimension": 1280,
    "query": None,
    "multichain": None,
    "methods": [
        {
            "path": "batch/embeddings",
            "outputs": ["embeddings"],
            "default_output": "embeddings",
            "description": "Per-residue embeddings.",
            "name": None,
        },
        {
            "path": "batch/logits",
            "outputs": ["logits"],
            "default_output": "logits",
            "description": None,
            "name": None,
        },
    ],
    "max_sequence_length": 4094,
}

PARAMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"sequences": {"type": "array", "items": {"type": "string"}}},
    "required": ["sequences"],
}

JOB: dict[str, Any] = {
    "job_id": "e0b1c3d4-0000-4000-8000-000000000001",
    "job_type": f"/models/{MODEL_ID}/batch/embeddings",
    "status": "PENDING",
    "created_date": "2026-09-05T08:00:00Z",
    "args": {"sequences": ["MKTAYIAK"]},
}


NON_MODELS_JOB: dict[str, Any] = {
    "job_id": "e0b1c3d4-0000-4000-8000-000000000009",
    "job_type": "/embeddings/embed",
    "status": "SUCCESS",
    "created_date": "2026-09-05T08:00:00Z",
}


def _respond(mock: MagicMock, payload: object) -> None:
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = payload


@pytest.fixture
def fail_first_backend():
    """A server that drops the first POST's connection, then answers any later one.

    Yields a function taking the JSON body to answer the retried POST with,
    returning the backend url and the list of ``(body, idempotency key)``
    pairs it received.
    """
    received: list[tuple[bytes, str | None]] = []
    servers: list[ThreadingHTTPServer] = []
    threads: list[threading.Thread] = []

    def start(response_payload: dict) -> tuple[str, list[tuple[bytes, str | None]]]:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                content_length = int(self.headers["Content-Length"])
                body = self.rfile.read(content_length)
                received.append((body, self.headers.get("Idempotency-Key")))
                if len(received) == 1:
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    self.connection.close()
                    return

                response_body = json.dumps(response_payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
        return f"http://127.0.0.1:{server.server_port}/api/", received

    try:
        yield start
    finally:
        for server, thread in zip(servers, threads):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class TestDiscovery:
    @pytest.fixture
    def models_api(self, mock_session) -> ModelsAPI:
        """The legacy per-model attributes fetch their metadata on construction."""
        _respond(
            mock_session.get,
            {"model_id": "legacy", "description": {"summary": "legacy model"}},
        )
        api = ModelsAPI(mock_session)
        mock_session.get.reset_mock()
        return api

    def test_list_models_returns_ids_by_default(self, models_api, mock_session):
        _respond(mock_session.get, [MODEL_ID, "esm2_t6_8M_UR50D"])

        ids = models_api.list_models()

        mock_session.get.assert_called_once_with(
            "v1/models", params={"scope": "system", "limit": 1000, "offset": 0}
        )
        assert ids == [MODEL_ID, "esm2_t6_8M_UR50D"]

    def test_list_models_reads_every_page(self, models_api, mock_session):
        pages = [[f"m{i}" for i in range(1000)], ["last"]]
        mock_session.get.return_value.json.side_effect = pages

        ids = models_api.list_models()

        assert ids == [*pages[0], "last"]
        assert [
            c.kwargs["params"]["offset"] for c in mock_session.get.call_args_list
        ] == [
            0,
            1000,
        ]

    def test_list_models_verbose_parses_metadata_and_forwards_path_filters(
        self, models_api, mock_session
    ):
        _respond(mock_session.get, [METADATA])

        models = models_api.list_models(verbose=True, path=["embeddings"])

        mock_session.get.assert_called_once_with(
            "v1/models",
            params={
                "scope": "system",
                "limit": 1000,
                "verbose": "true",
                "path": ["embeddings"],
                "offset": 0,
            },
        )
        assert isinstance(models[0], ModelMetadata)
        assert models[0].model_id == MODEL_ID

    def test_list_models_keeps_only_method_path_filters(self):
        assert list(inspect.signature(ModelsAPI.list_models).parameters) == [
            "self",
            "verbose",
            "path",
        ]
        assert list(inspect.signature(models_api.list_models).parameters) == [
            "session",
            "verbose",
            "path",
        ]

    def test_get_model_reads_the_card_once(self, models_api, mock_session):
        _respond(mock_session.get, METADATA)

        model = models_api.get_model(MODEL_ID)

        assert [m.path for m in model.methods] == ["batch/embeddings", "batch/logits"]
        mock_session.get.assert_called_once_with(
            f"v1/models/{MODEL_ID}", timeout=mock_session.timeout
        )

    def test_method_descriptor_default_output_is_the_first_name(self):
        method = METADATA["methods"][0] | {"outputs": ["embeddings", "attention"]}
        metadata = ModelMetadata.model_validate(METADATA | {"methods": [method]})

        assert metadata.methods[0].outputs == ["embeddings", "attention"]
        assert metadata.methods[0].default_output == "embeddings"

    def test_method_descriptor_accepts_a_typed_outputs_map(self):
        method = METADATA["methods"][0] | {
            "outputs": {"attention": "vector", "embeddings": "vector"},
            "default_output": "embeddings",
        }
        metadata = ModelMetadata.model_validate(METADATA | {"methods": [method]})

        assert metadata.methods[0].outputs == ["embeddings", "attention"]
        assert metadata.methods[0].default_output == "embeddings"

    def test_method_descriptor_takes_an_input_output_pair_as_its_path(self):
        method = MethodDescriptor.model_validate(
            {"input": "batch", "output": "embeddings", "outputs": ["embeddings"]}
        )

        assert method.path == "batch/embeddings"

    def test_model_metadata_reads_the_display_taxonomy(self):
        metadata = ModelMetadata.model_validate(METADATA)

        assert (
            metadata.family_id,
            metadata.model,
            metadata.name,
            metadata.variant,
        ) == ("esm", "ESM2", "ESM2 (650M)", "650M")
        assert metadata.type is None


class TestMethod:
    def test_esm_alias_embed_submits_to_models_route(self, mock_session):
        _respond(mock_session.get, METADATA)
        _respond(mock_session.post, JOB)
        esm = ModelsAPI(mock_session).esm2

        future = esm.embed(sequences=[b"MKTAYIAK"])

        assert isinstance(future, Future)
        mock_session.get.assert_called_once_with(
            f"v1/models/{MODEL_ID}", timeout=mock_session.timeout
        )
        assert mock_session.post.call_args.args == (
            f"v1/models/{MODEL_ID}/batch/embeddings",
        )
        assert mock_session.post.call_args.kwargs["json"] == {
            "sequences": ["MKTAYIAK"],
            "reduction": "MEAN",
        }

    def test_esm_embed_preserves_explicit_unpooled_reduction(self, mock_session):
        _respond(mock_session.get, METADATA)
        _respond(mock_session.post, JOB)

        ModelsAPI(mock_session).esm2.embed(sequences=["MKTAYIAK"], reduction=None)

        assert mock_session.post.call_args.kwargs["json"]["reduction"] is None

    def test_esm_logits_and_attn_submit_to_model_methods(self, mock_session):
        attn = {
            "path": "batch/attn",
            "outputs": {"attn": "vector"},
            "default_output": "attn",
            "description": None,
        }
        _respond(mock_session.get, METADATA | {"methods": [*METADATA["methods"], attn]})
        _respond(mock_session.post, JOB)
        esm = ModelsAPI(mock_session).esm2

        esm.logits(sequences=[b"MKTAYIAK"])
        esm.attn(sequences=["MKTAYIAK"])

        assert [call.args[0] for call in mock_session.post.call_args_list] == [
            f"v1/models/{MODEL_ID}/batch/logits",
            f"v1/models/{MODEL_ID}/batch/attn",
        ]
        assert [call.kwargs["json"] for call in mock_session.post.call_args_list] == [
            {"sequences": ["MKTAYIAK"]},
            {"sequences": ["MKTAYIAK"]},
        ]

    def test_poet2_embed_submits_to_models_route(self, mock_session):
        _respond(mock_session.get, METADATA | {"model_id": "poet-2"})
        _respond(
            mock_session.post, JOB | {"job_type": "/models/poet-2/batch/embeddings"}
        )

        future = ModelsAPI(mock_session).poet2.embed(
            sequences=["MKTAYIAK"], prompt_id="prompt-1"
        )

        assert isinstance(future, Future)
        assert mock_session.post.call_args.args == (
            "v1/models/poet-2/batch/embeddings",
        )
        assert mock_session.post.call_args.kwargs["json"] == {
            "sequences": ["MKTAYIAK"],
            "prompt_id": "prompt-1",
            "reduction": "MEAN",
        }

    def test_a_card_without_the_method_submits_nothing(self, mock_session):
        _respond(mock_session.get, METADATA | {"model_id": "ablang2", "methods": []})

        with pytest.raises(ValueError, match="has no embed; its methods: none"):
            ModelsAPI(mock_session).ablang2.embed(sequences=["MKTAYIAK"])
        with pytest.raises(ValueError, match="has no generate"):
            ModelsAPI(mock_session).rfdiffusion.generate(query_id="q")

        mock_session.post.assert_not_called()

    @pytest.fixture
    def model(self, mock_session) -> Model:
        return Model(mock_session, MODEL_ID, ModelMetadata.model_validate(METADATA))

    def test_a_missing_verb_names_the_cards_methods(self, model, mock_session):
        with pytest.raises(
            ValueError,
            match="has no single_site; its methods: batch/embeddings, batch/logits",
        ):
            model.single_site("MKTAYIAK")
        mock_session.post.assert_not_called()

    @pytest.mark.parametrize(
        "verb, args, route, body",
        [
            ("embed", (["MKTAYIAK"],), "batch/embeddings", {"sequences": ["MKTAYIAK"]}),
            ("logits", ([b"MKTAYIAK"],), "batch/logits", {"sequences": ["MKTAYIAK"]}),
            ("attn", (["MKTAYIAK"],), "batch/attn", {"sequences": ["MKTAYIAK"]}),
            (
                "score",
                (["MKTAYIAK"],),
                "batch/loglikelihood",
                {"sequences": ["MKTAYIAK"]},
            ),
            (
                "single_site",
                (b"MKTAYIAK",),
                "single-site/scores",
                {"base_sequence": "MKTAYIAK"},
            ),
            (
                "single_site",
                ("MKTAYIAK",),
                "single-site/predictions",
                {"base_sequence": "MKTAYIAK"},
            ),
            ("generate", (), "generate/structures", {"query_id": "q"}),
            ("generate", (), "generate/sequences", {"query_id": "q"}),
            (
                "fold",
                ([["MKTAYIAK"]],),
                "batch/structures",
                {"complexes": [["MKTAYIAK"]]},
            ),
            (
                "predict",
                (["MKTAYIAK"],),
                "batch/predictions",
                {"sequences": ["MKTAYIAK"]},
            ),
        ],
    )
    def test_each_verb_submits_its_card_method(
        self, mock_session, verb, args, route, body
    ):
        method = {"path": route, "outputs": ["out"]}
        model = Model(
            mock_session,
            "m",
            ModelMetadata.model_validate(METADATA | {"methods": [method]}),
        )
        _respond(mock_session.post, JOB)
        params = {"query_id": "q"} if verb == "generate" else {}

        future = getattr(model, verb)(*args, **params)

        assert mock_session.post.call_args.args == (f"v1/models/m/{route}",)
        assert mock_session.post.call_args.kwargs["json"] == body
        assert future._descriptor.path == route

    def test_run_is_not_public(self, model):
        import openprotein.models

        assert not hasattr(model, "run") and not hasattr(model, "method")
        assert not hasattr(openprotein.models, "Method")

    def test_a_verb_posts_params_as_the_body(self, model, mock_session):
        _respond(mock_session.post, JOB)

        future = model.embed(["MKTAYIAK"])

        call = mock_session.post.call_args
        assert call.args == (f"v1/models/{MODEL_ID}/batch/embeddings",)
        assert call.kwargs["json"] == {"sequences": ["MKTAYIAK"]}
        assert UUID(call.kwargs["headers"]["Idempotency-Key"])
        assert isinstance(future, Future)
        assert future.id == JOB["job_id"]
        assert future._model_id == MODEL_ID
        assert future._path == "batch/embeddings"
        assert future.args == {"sequences": ["MKTAYIAK"]}

    def test_a_verb_hands_the_future_its_method(self, model, mock_session):
        _respond(mock_session.post, JOB)

        future = model.logits(["MKTAYIAK"])

        assert future._descriptor == model.methods[1]
        mock_session.get.assert_not_called()

    def test_a_verb_forwards_prompt_id_from_a_stored_or_a_sampled_prompt(
        self, mock_session
    ):
        from openprotein.data.prompts.prompts import Prompt
        from openprotein.data.prompts.schemas import PromptSummary

        poet2 = Model(
            mock_session,
            "poet-2",
            ModelMetadata.model_validate(METADATA | {"model_id": "poet-2"}),
        )
        _respond(
            mock_session.post, JOB | {"job_type": "/models/poet-2/batch/embeddings"}
        )

        stored_prompt = Prompt(
            mock_session,
            PromptSummary(
                id="prompt-1",
                owner="user",
                created_at="2026-09-05T08:00:00Z",
            ),
        )
        sampled_prompt = Prompt(
            mock_session,
            PromptSummary(id="sample-job-1", status="PENDING", job_id="sample-job-1"),
        )

        poet2.embed(["MKTAYIAK"], prompt_id=stored_prompt.id)
        poet2.embed(["MKTAYIAK"], prompt_id=sampled_prompt.id)

        calls = mock_session.post.call_args_list
        assert calls[0].args == ("v1/models/poet-2/batch/embeddings",)
        assert calls[0].kwargs["json"]["prompt_id"] == "prompt-1"
        assert calls[1].kwargs["json"]["prompt_id"] == "sample-job-1"

    def test_a_verb_sends_its_params_as_given(self, model, mock_session):
        _respond(mock_session.post, JOB)

        model.embed(["MKTAYIAK"], seed=None)

        assert mock_session.post.call_args.kwargs["json"]["seed"] is None

    def test_a_verb_keeps_a_given_seed(self, model, mock_session):
        _respond(mock_session.post, JOB)

        model.embed(["MKTAYIAK"], seed=7)

        assert mock_session.post.call_args.kwargs["json"]["seed"] == 7

    def test_a_verb_retries_the_same_body_on_425(
        self, model, mock_session, monkeypatch
    ):
        from openprotein.errors import HTTPError

        monkeypatch.setattr("openprotein.jobs._submission.time.sleep", lambda _: None)
        pending = MagicMock()
        pending.status_code = 425
        pending.headers = {"Retry-After": "0"}
        mock_session.post.side_effect = [
            HTTPError(pending),
            MagicMock(**{"json.return_value": JOB}),
        ]

        future = model.embed(["MKTAYIAK"])

        assert future.id == JOB["job_id"]
        calls = mock_session.post.call_args_list
        assert calls[0].kwargs["json"] == calls[1].kwargs["json"]
        assert calls[0].kwargs["headers"] == calls[1].kwargs["headers"]

    def test_each_logical_submission_gets_a_new_admission_id(self, model, mock_session):
        _respond(mock_session.post, JOB)

        with patch(
            "openprotein.models._api.uuid4", side_effect=["admission-1", "admission-2"]
        ):
            model.embed(["MKTAYIAK"])
            model.embed(["MKTAYIAK"])

        calls = mock_session.post.call_args_list
        assert [call.kwargs["headers"]["Idempotency-Key"] for call in calls] == [
            "admission-1",
            "admission-2",
        ]
        assert [call.kwargs["json"] for call in calls] == [
            {"sequences": ["MKTAYIAK"]},
            {"sequences": ["MKTAYIAK"]},
        ]

    def test_model_post_is_sent_once_when_the_connection_drops(
        self, fail_first_backend
    ):
        backend, received = fail_first_backend(JOB)
        with patch.object(APISession, "_init_auth"):
            session = APISession("user", "password", backend=backend, timeout=2)
        try:
            with pytest.raises(requests.exceptions.ConnectionError):
                models_api.run_method(
                    session, MODEL_ID, "batch/embeddings", {"sequences": ["MKTAYIAK"]}
                )
        finally:
            session.close()

        assert len(received) == 1


class TestJobsOutsideTheModelsAPI:
    """A job outside the models API is a handle with status, without outputs."""

    @pytest.fixture
    def future(self, mock_session):
        _respond(mock_session.post, NON_MODELS_JOB)
        model = Model(mock_session, MODEL_ID, ModelMetadata.model_validate(METADATA))
        return model.embed(["MKTAYIAK"])

    def test_the_submit_parses_and_tracks_the_job(self, future, mock_session):
        assert isinstance(future, Future)
        assert future.id == NON_MODELS_JOB["job_id"]
        assert future.status is JobStatus.SUCCESS
        assert future._path == "batch/embeddings"
        assert future._default_output == "embeddings"
        mock_session.get.assert_not_called()

    @pytest.mark.parametrize(
        "read",
        [
            lambda f: f.get(),
            lambda f: list(f.stream()),
            lambda f: f.get_item(0),
            lambda f: f.unit(0),
            lambda f: f.units,
            lambda f: f._model_id,
        ],
    )
    def test_reading_outputs_without_stored_outputs_raises(
        self, future, mock_session, read
    ):
        with pytest.raises(APIError, match="has no stored outputs"):
            read(future)
        mock_session.get.assert_not_called()


class TestFuture:
    def test_default_display_shows_id_status_and_compact_args(self, mock_session):
        sequences = ["MKTAYIAK" * 10, "MEEPQSDPSV"]
        job = Job.model_validate(
            JOB | {"args": {"sequences": sequences, "reduction": "MEAN"}}
        )
        future = Future(mock_session, job)

        assert repr(future) == (
            f"Future(job_id={JOB['job_id']!r}, status='PENDING', "
            "args={'sequences': <2 items>, 'reduction': 'MEAN'})"
        )
        assert str(future) == repr(future)
        assert all(sequence not in repr(future) for sequence in sequences)
        mock_session.get.assert_not_called()

    def test_display_elides_a_long_query(self, mock_session):
        query = "MKTAYIAK" * 10
        future = Future(
            mock_session, Job.model_validate(JOB | {"args": {"query": query}})
        )

        assert query not in repr(future)
        assert "<str of 80 chars>" in repr(future)

    def test_display_does_not_fetch_missing_args(self, mock_session):
        future = Future(mock_session, Job.model_validate(JOB | {"args": None}))

        assert "args=<not loaded>" in repr(future)
        mock_session.get.assert_not_called()

    def test_a_models_job_loads_as_its_typed_result_future(self, mock_session):
        future = Future.create(session=mock_session, response=JOB)

        assert type(future) is results.EmbeddingFuture
        assert _dispatch.is_models_job(future.job)
        assert future.job.job_type == JOB["job_type"]

    @pytest.mark.parametrize(
        "family, future_class, outputs",
        [
            ("svd", SVDFitFuture, ["sequences"]),
            ("umap", UMAPFitFuture, ["sequences", "embeddings"]),
        ],
        ids=["svd", "umap"],
    )
    def test_a_fit_job_loads_as_its_fit_future(
        self, mock_session, family, future_class, outputs
    ):
        fit = JOB | {
            "job_type": f"/models/{family}/fit",
            "outputs": ["ignored"],
        }

        future = Future.create(session=mock_session, response=fit)

        assert type(future) is future_class
        assert (future.outputs, future._default_output) == (outputs, outputs[0])
        mock_session.get.assert_not_called()

    def test_wait_until_done_reaches_the_terminal_status(
        self, mock_session, monkeypatch
    ):
        monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda _: None)
        _respond(mock_session.get, JOB | {"status": "SUCCESS"})
        future = Future(mock_session, Job.model_validate(JOB))

        done = future.wait_until_done()

        mock_session.get.assert_called_with(f"v1/jobs/{JOB['job_id']}")
        assert done
        assert future.job.status is JobStatus.SUCCESS
        assert future.job.model_dump()["args"] == JOB["args"]
