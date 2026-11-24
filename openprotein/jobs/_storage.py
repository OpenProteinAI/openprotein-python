"""Direct artifact reads; platform authentication is never sent to object storage."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, BinaryIO, Protocol
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#: Artifact downloads `JobsAPI._iter_job_outputs` runs at once.
DOWNLOAD_WORKERS = 16


def _download_session() -> requests.Session:
    """A pooled session for signed URLs that retries object-storage 5xx."""
    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_maxsize=DOWNLOAD_WORKERS, max_retries=retry)
    session = requests.Session()
    for prefix in ("https://", "http://"):
        session.mount(prefix, adapter)
    return session


class ArtifactStorage(Protocol):
    """Injectable backend boundary for authorized stored-value references."""

    def iter_bytes(self, reference: Mapping[str, Any]) -> Iterator[bytes]: ...


class DirectArtifactStorage:
    """Download authorized signed URLs or S3 references with ambient credentials.

    HTTP downloads use a separate requests session, never the platform's bearer
    token or user headers.
    """

    def __init__(
        self,
        s3_client: Any = None,
        *,
        chunk_size: int = 1024 * 1024,
        http: requests.Session | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.s3_client = s3_client
        self.chunk_size = chunk_size
        self.http = http if http is not None else _download_session()

    def iter_bytes(self, reference: Mapping[str, Any]) -> Iterator[bytes]:
        download_url = reference.get("download_url")
        if download_url is not None:
            if not isinstance(download_url, str) or urlsplit(
                download_url
            ).scheme not in ("https", "http"):
                raise ValueError("invalid artifact download URL")
            with self.http.get(download_url, stream=True, timeout=180) as response:
                response.raise_for_status()
                yield from response.iter_content(chunk_size=self.chunk_size)
            return
        uri = reference.get("uri")
        if not isinstance(uri, str):
            raise TypeError("artifact reference requires a uri")
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not parsed.path.lstrip("/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid S3 artifact URI")
        if self.s3_client is None:
            try:
                boto3 = importlib.import_module("boto3")
            except ImportError as exc:
                raise RuntimeError("S3 reads require boto3") from exc
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3")
                or os.environ.get("S3_ENDPOINT_URL"),
            )
        body = self.s3_client.get_object(
            Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
        )["Body"]
        try:
            while chunk := body.read(self.chunk_size):
                yield chunk
        finally:
            body.close()


@contextmanager
def verified_artifact(
    storage: ArtifactStorage, reference: Mapping[str, Any]
) -> Iterator[BinaryIO]:
    """Spool without a total-size ceiling and verify the entire content digest.

    Only a fixed-size transfer buffer is kept in memory. Storage checksums and
    multipart ETags deliberately do not substitute for full-content SHA-256.
    """
    size, digest = reference.get("byte_length"), reference.get("sha256")
    if type(size) is not int or size < 0:
        raise ValueError("job output ref has an invalid byte_length")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("job output ref has an invalid sha256")
    actual_size = 0
    hasher = hashlib.sha256()
    with tempfile.TemporaryFile(mode="w+b") as spool:
        for chunk in storage.iter_bytes(reference):
            actual_size += len(chunk)
            if actual_size > size:
                raise ValueError(
                    "downloaded job output size does not match its reference"
                )
            hasher.update(chunk)
            spool.write(chunk)
        if actual_size != size:
            raise ValueError("downloaded job output size does not match its reference")
        if hasher.hexdigest() != digest:
            raise ValueError(
                "downloaded job output digest does not match its reference"
            )
        spool.seek(0)
        yield spool


def download_ref(value: Mapping[str, Any]) -> bytes:
    """Fetch and verify raw bytes for a stored ``{ref: Ref}`` value."""
    reference = value.get("ref")
    if not isinstance(reference, Mapping):
        raise ValueError("expected a ref-backed value")
    storage = DirectArtifactStorage()
    with verified_artifact(storage, reference) as source:
        return source.read()
