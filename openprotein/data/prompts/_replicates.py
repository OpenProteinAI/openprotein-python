"""Each replicate's complex sequences, from the prompt download."""

import re
import zipfile
from io import BytesIO

from openprotein import fasta

_REPLICATE_NAME = re.compile(r"replicate-(\d+)\.fasta$")


def parse_replicates(data: bytes) -> list[list[str]]:
    """One entry per replicate, in position order; each a list of complex sequences."""
    with zipfile.ZipFile(BytesIO(data)) as archive:
        positions = sorted(
            (int(match.group(1)), name)
            for name in archive.namelist()
            if (match := _REPLICATE_NAME.search(name))
        )
        return [
            [
                sequence
                for _, sequence in fasta.parse_stream(
                    iter(archive.read(name).decode("utf-8").splitlines())
                )
            ]
            for _, name in positions
        ]
