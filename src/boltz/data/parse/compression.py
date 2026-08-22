import gzip
import io
import lzma
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import zstandard as zstd


@contextmanager
def open_maybe_compressed_text(path: Path) -> Iterator[TextIO]:
    """Open plain, gzip, xz, or zstd text based on content magic bytes."""
    with path.open("rb") as raw_file:
        header = raw_file.read(6)
        raw_file.seek(0)

        if header[:2] == b"\x1f\x8b":
            with gzip.open(raw_file, "rt") as text_file:
                yield text_file
        elif header == b"\xfd\x37\x7a\x58\x5a\x00":
            with lzma.open(raw_file, "rt") as text_file:
                yield text_file
        elif header[:4] == b"\x28\xb5\x2f\xfd":
            with zstd.open(raw_file, "rt") as text_file:
                yield text_file
        else:
            with io.TextIOWrapper(raw_file, encoding="utf-8") as text_file:
                yield text_file
