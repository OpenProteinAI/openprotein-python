import json
import os
import tempfile
from pathlib import Path

from filelock import FileLock
from pydantic import BaseModel, ValidationError


class TokenRecord(BaseModel):
    """A persisted auth record for one identity."""

    access_token: str
    refresh_token: str | None = None
    expires_at: float = 0.0


class TokenStore:
    """Persist the rotating refresh token across processes.

    The on-disk file is a JSON object keyed by identity
    ``"{backend}|{username}"`` -> ``TokenRecord``. The read-modify-write
    critical section is guarded by a cross-process ``filelock`` so two
    processes never refresh the same rotating token concurrently.
    """

    def __init__(self, dir: Path | None = None):
        self.dir = Path(dir) if dir is not None else Path.home() / ".openprotein"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "token.json"
        # One FileLock instance per store -> reentrant within this process.
        self._lock = FileLock(str(self.dir / "token.lock"))

    @staticmethod
    def identity(backend: str, username: str) -> str:
        return f"{backend}|{username}"

    def lock(self):
        """Context manager for the cross-process critical section."""
        return self._lock

    def _read_all(self) -> dict:
        # Caller must hold self._lock: this is the read half of the
        # read-modify-write critical section in load()/save().
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            # Missing or corrupt file -> behave as if there were no records.
            return {}

    def load(self, identity: str) -> TokenRecord | None:
        with self._lock:
            data = self._read_all()
            entry = data.get(identity)
            if entry is None:
                return None
            try:
                return TokenRecord.model_validate(entry)
            except ValidationError:
                return None

    def save(self, identity: str, record: TokenRecord) -> None:
        with self._lock:
            data = self._read_all()
            data[identity] = record.model_dump()
            self._atomic_write(data)

    def _atomic_write(self, data: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix="token.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            # Don't leave a temp file behind on failure.
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
