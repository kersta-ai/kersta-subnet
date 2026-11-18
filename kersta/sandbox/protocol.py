"""Framing between the validator and the model process, over stdin and stdout.

A frame is::

    4 bytes   big endian length of the header
    header    UTF-8 JSON; "blobs" lists the arrays that follow
    blobs     raw little endian array bytes, in the listed order

No pickle anywhere: arrays travel as dtype + shape + bytes.
"""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO, Final

import numpy as np

MAX_HEADER: Final[int] = 16 * 1024 * 1024
MAX_BLOB: Final[int] = 1024**3
ALLOWED_DTYPES: Final[frozenset[str]] = frozenset(
    {"uint8", "int16", "int32", "int64", "float16", "float32", "float64", "bool"}
)


class ProtocolError(RuntimeError):
    pass


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks, left = [], size
    while left:
        chunk = stream.read(left)
        if not chunk:
            raise EOFError("stream closed mid frame")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def write_frame(
    stream: BinaryIO, header: dict[str, Any], arrays: dict[str, np.ndarray] | None = None
) -> None:
    blobs, payloads = [], []
    for name, array in (arrays or {}).items():
        array = np.ascontiguousarray(array)
        if array.dtype.name not in ALLOWED_DTYPES:
            raise ProtocolError(f"array {name} has unsupported dtype {array.dtype}")
        data = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        blobs.append({"name": name, "dtype": array.dtype.name, "shape": list(array.shape),
                      "nbytes": len(data)})
        payloads.append(data)
    body = json.dumps({**header, "blobs": blobs}, allow_nan=False).encode("utf-8")
    stream.write(struct.pack(">I", len(body)) + body)
    for data in payloads:
        stream.write(data)
    stream.flush()


def read_frame(stream: BinaryIO) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    (length,) = struct.unpack(">I", _read_exact(stream, 4))
    if length > MAX_HEADER:
        raise ProtocolError(f"header of {length} bytes is too large")
    header = json.loads(_read_exact(stream, length))
    if not isinstance(header, dict):
        raise ProtocolError("header must be a JSON object")
    arrays: dict[str, np.ndarray] = {}
    for blob in header.pop("blobs", []) or []:
        dtype, shape, nbytes = str(blob["dtype"]), [int(s) for s in blob["shape"]], int(blob["nbytes"])
        if dtype not in ALLOWED_DTYPES or nbytes > MAX_BLOB or any(s < 0 for s in shape):
            raise ProtocolError(f"refusing blob {blob.get('name')!r}")
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if expected != nbytes:
            raise ProtocolError(f"blob {blob.get('name')!r} size does not match its shape")
        data = _read_exact(stream, nbytes)
        little = np.dtype(dtype).newbyteorder("<")
        arrays[str(blob["name"])] = np.frombuffer(data, dtype=little).reshape(shape)
    return header, arrays
