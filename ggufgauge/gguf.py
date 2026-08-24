"""Small, bounded GGUF metadata reader.

The parser intentionally stops before tensor descriptors and tensor data. It reads
enough scalar metadata to estimate a CPU launch without importing gguf-py/numpy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import struct
from typing import Any, BinaryIO


class GGUFError(ValueError):
    """Raised when a file is not a supported, safely parseable GGUF."""


_SCALARS: dict[int, tuple[str, int]] = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}
_STRING = 8
_ARRAY = 9
_MAX_METADATA = 1_000_000
_MAX_STRING_BYTES = 64 * 1024 * 1024
_MAX_ARRAY_ITEMS = 100_000_000
_CAPTURE_ARRAY_ITEMS = 64


@dataclass(frozen=True, slots=True)
class ModelInfo:
    path: str
    file_bytes: int
    version: int
    tensor_count: int
    metadata_count: int
    architecture: str
    name: str | None
    block_count: int | None
    embedding_length: int | None
    head_count: int | None
    head_count_kv: int | None
    context_length: int | None
    key_length: int | None
    value_length: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _Reader:
    def __init__(self, stream: BinaryIO, size: int) -> None:
        self.stream = stream
        self.size = size

    def read_exact(self, length: int) -> bytes:
        if length < 0 or self.stream.tell() + length > self.size:
            raise GGUFError("GGUF metadata points outside the file")
        data = self.stream.read(length)
        if len(data) != length:
            raise GGUFError("unexpected end of GGUF metadata")
        return data

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.read_exact(size))[0]

    def skip(self, length: int) -> None:
        if length < 0 or self.stream.tell() + length > self.size:
            raise GGUFError("GGUF metadata points outside the file")
        self.stream.seek(length, 1)

    def string(self, *, capture: bool = True) -> str | None:
        length = int(self.unpack("Q"))
        if length > _MAX_STRING_BYTES:
            raise GGUFError(f"GGUF string length is unsafe: {length}")
        raw = self.read_exact(length)
        if not capture:
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("GGUF metadata contains invalid UTF-8") from exc

    def value(self, value_type: int, *, capture: bool) -> object | None:
        if value_type in _SCALARS:
            fmt, _ = _SCALARS[value_type]
            value = self.unpack(fmt)
            return value if capture else None
        if value_type == _STRING:
            return self.string(capture=capture)
        if value_type != _ARRAY:
            raise GGUFError(f"unknown GGUF metadata value type: {value_type}")

        item_type = int(self.unpack("I"))
        count = int(self.unpack("Q"))
        if count > _MAX_ARRAY_ITEMS:
            raise GGUFError(f"GGUF array length is unsafe: {count}")
        if item_type == _ARRAY:
            raise GGUFError("nested GGUF metadata arrays are not supported")

        if not capture or count > _CAPTURE_ARRAY_ITEMS:
            if item_type in _SCALARS:
                self.skip(_SCALARS[item_type][1] * count)
            elif item_type == _STRING:
                for _ in range(count):
                    self.string(capture=False)
            else:
                raise GGUFError(f"unknown GGUF array value type: {item_type}")
            return None
        return [self.value(item_type, capture=True) for _ in range(count)]


def _positive_int(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def read_model_info(path: str | Path) -> ModelInfo:
    target = Path(path)
    if not target.is_file():
        raise GGUFError(f"model file does not exist: {target}")
    file_bytes = target.stat().st_size
    if file_bytes < 24:
        raise GGUFError("file is too small to contain a GGUF header")

    metadata: dict[str, object] = {}
    with target.open("rb") as stream:
        reader = _Reader(stream, file_bytes)
        if reader.read_exact(4) != b"GGUF":
            raise GGUFError("file does not start with GGUF magic")
        version = int(reader.unpack("I"))
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version: {version}")
        tensor_count = int(reader.unpack("Q"))
        metadata_count = int(reader.unpack("Q"))
        if metadata_count > _MAX_METADATA:
            raise GGUFError(f"GGUF metadata count is unsafe: {metadata_count}")

        for _ in range(metadata_count):
            key = reader.string(capture=True)
            assert key is not None
            value_type = int(reader.unpack("I"))
            value = reader.value(value_type, capture=True)
            if value is not None:
                metadata[key] = value

    architecture_value = metadata.get("general.architecture")
    architecture = (
        architecture_value.strip().lower()
        if isinstance(architecture_value, str) and architecture_value.strip()
        else "unknown"
    )
    prefix = architecture
    name_value = metadata.get("general.name")
    name = name_value.strip() if isinstance(name_value, str) and name_value.strip() else None

    return ModelInfo(
        path=str(target),
        file_bytes=file_bytes,
        version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        architecture=architecture,
        name=name,
        block_count=_positive_int(metadata, f"{prefix}.block_count"),
        embedding_length=_positive_int(metadata, f"{prefix}.embedding_length"),
        head_count=_positive_int(metadata, f"{prefix}.attention.head_count"),
        head_count_kv=_positive_int(metadata, f"{prefix}.attention.head_count_kv"),
        context_length=_positive_int(metadata, f"{prefix}.context_length"),
        key_length=_positive_int(metadata, f"{prefix}.attention.key_length"),
        value_length=_positive_int(metadata, f"{prefix}.attention.value_length"),
    )
