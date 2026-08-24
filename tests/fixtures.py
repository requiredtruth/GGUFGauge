from __future__ import annotations

from pathlib import Path
import struct


STRING = 8
UINT32 = 4


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_test_gguf(path: Path, *, payload_bytes: int = 2 * 1024**2) -> Path:
    metadata: list[tuple[str, int, object]] = [
        ("general.architecture", STRING, "llama"),
        ("general.name", STRING, "Synthetic Llama"),
        ("llama.block_count", UINT32, 32),
        ("llama.embedding_length", UINT32, 4096),
        ("llama.attention.head_count", UINT32, 32),
        ("llama.attention.head_count_kv", UINT32, 8),
        ("llama.context_length", UINT32, 8192),
    ]
    parts = [b"GGUF", struct.pack("<IQQ", 3, 0, len(metadata))]
    for key, value_type, value in metadata:
        parts.extend((_string(key), struct.pack("<I", value_type)))
        if value_type == STRING:
            parts.append(_string(str(value)))
        else:
            parts.append(struct.pack("<I", int(value)))
    header = b"".join(parts)
    with path.open("wb") as handle:
        handle.write(header)
        handle.truncate(len(header) + payload_bytes)
    return path
