"""Conservative CPU/local launch-envelope estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .gguf import ModelInfo
from .system import SystemCapacity


KV_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.125,
    "q5_1": 0.75,
    "q5_0": 0.6875,
    "q4_1": 0.625,
    "q4_0": 0.5625,
    "iq4_nl": 0.5625,
}


@dataclass(frozen=True, slots=True)
class LaunchEstimate:
    kv_type: str
    reserve_bytes: int
    estimated_model_bytes: int
    runtime_overhead_bytes: int
    kv_bytes_per_token: int | None
    recommended_context: int | None
    trained_context: int | None
    estimated_total_bytes: int | None
    fits_weights: bool
    confidence: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


def kv_bytes_per_token(model: ModelInfo, kv_type: str) -> int | None:
    bytes_each = KV_BYTES.get(kv_type)
    if bytes_each is None:
        raise ValueError(f"unsupported KV type: {kv_type}")
    if not model.block_count or not model.embedding_length or not model.head_count:
        return None
    kv_heads = model.head_count_kv or model.head_count
    if kv_heads <= 0 or model.head_count <= 0:
        return None
    default_head_dim = model.embedding_length / model.head_count
    key_dim = model.key_length or default_head_dim
    value_dim = model.value_length or default_head_dim
    value = model.block_count * kv_heads * (key_dim + value_dim) * bytes_each
    return max(1, math.ceil(value))


def memory_for_context(estimate: LaunchEstimate, context: int) -> int | None:
    if estimate.kv_bytes_per_token is None:
        return None
    return (
        estimate.reserve_bytes
        + estimate.estimated_model_bytes
        + estimate.runtime_overhead_bytes
        + estimate.kv_bytes_per_token * context
    )


def estimate_launch(
    model: ModelInfo,
    capacity: SystemCapacity,
    *,
    kv_type: str = "f16",
    reserve_percent: float = 15.0,
) -> LaunchEstimate:
    if kv_type not in KV_BYTES:
        raise ValueError(f"unsupported KV type: {kv_type}")
    if not 0.0 <= reserve_percent <= 80.0:
        raise ValueError("reserve_percent must be between 0 and 80")

    available = capacity.available_bytes
    proportional_reserve = math.ceil(available * reserve_percent / 100.0)
    reserve_floor = min(512 * 1024**2, available // 4)
    reserve = max(proportional_reserve, reserve_floor)
    estimated_model = math.ceil(model.file_bytes * 1.05)
    runtime_overhead = max(256 * 1024**2, math.ceil(model.file_bytes * 0.08))
    fixed = reserve + estimated_model + runtime_overhead
    fits_weights = fixed < available
    per_token = kv_bytes_per_token(model, kv_type)
    warnings: list[str] = []

    if not fits_weights:
        warnings.append("model plus reserve and runtime overhead exceed available memory")
    if per_token is None:
        warnings.append("architecture metadata is insufficient to derive KV-cache size")
    if model.context_length is None:
        warnings.append("GGUF does not expose a trained context limit")

    recommended: int | None = None
    estimated_total: int | None = None
    if fits_weights and per_token is not None:
        raw_context = (available - fixed) // per_token
        if model.context_length is not None:
            raw_context = min(raw_context, model.context_length)
        recommended = int(raw_context // 256 * 256)
        if recommended < 256:
            recommended = None
            warnings.append("less than 256 tokens fit inside the conservative envelope")
        else:
            estimated_total = fixed + recommended * per_token

    confidence = "medium"
    if per_token is None or not fits_weights:
        confidence = "low"
    elif model.key_length and model.value_length and model.context_length:
        confidence = "medium-high"
    warnings.append(
        "estimate excludes GPU offload and backend-specific buffers; verify with a short runtime load"
    )
    return LaunchEstimate(
        kv_type=kv_type,
        reserve_bytes=reserve,
        estimated_model_bytes=estimated_model,
        runtime_overhead_bytes=runtime_overhead,
        kv_bytes_per_token=per_token,
        recommended_context=recommended,
        trained_context=model.context_length,
        estimated_total_bytes=estimated_total,
        fits_weights=fits_weights,
        confidence=confidence,
        warnings=tuple(warnings),
    )
