"""Process-visible RAM and CPU capacity detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SystemCapacity:
    available_bytes: int
    host_available_bytes: int | None
    cgroup_remaining_bytes: int | None
    cgroup_path: str | None
    logical_cpus: int
    recommended_threads: int
    memory_source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_integer(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _host_available(path: Path) -> int | None:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def _sysconf_available() -> int | None:
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return pages * page_size if pages > 0 and page_size > 0 else None


def _unified_cgroup_path(path: Path) -> tuple[str, ...] | None:
    """Return a safe cgroup v2 path from /proc/self/cgroup."""
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        return None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3 or fields[0] != "0" or fields[1]:
            continue
        parts = tuple(part for part in fields[2].split("/") if part and part != ".")
        if any(part == ".." for part in parts):
            return None
        return parts
    return None


def _cgroup_remaining(
    root: Path, unified_path: tuple[str, ...] | None
) -> tuple[int | None, str | None]:
    # A process can be constrained by its own cgroup or any ancestor. Inspect
    # the full v2 chain and retain the tightest remaining-memory value.
    v2_candidates: list[tuple[int, str]] = []
    paths = [root]
    if unified_path:
        paths.extend(
            root.joinpath(*unified_path[:index])
            for index in range(1, len(unified_path) + 1)
        )
    for index, directory in enumerate(paths):
        limit = _read_integer(directory / "memory.max")
        current = _read_integer(directory / "memory.current")
        if limit is not None and current is not None and 0 < limit < (1 << 60):
            label = "/" if index == 0 else "/" + "/".join(unified_path[:index])
            v2_candidates.append((max(0, limit - current), label))
    if v2_candidates:
        return min(v2_candidates, key=lambda candidate: candidate[0])

    # Common cgroup v1 mount layout
    limit = _read_integer(root / "memory" / "memory.limit_in_bytes")
    current = _read_integer(root / "memory" / "memory.usage_in_bytes")
    if limit is not None and current is not None and 0 < limit < (1 << 60):
        return max(0, limit - current), "/memory"
    return None, None


def _logical_cpu_count() -> int:
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    count = len(affinity) if affinity else (os.cpu_count() or 1)
    return max(1, int(count))


def detect_capacity(
    *,
    ram_bytes: int | None = None,
    proc_meminfo: str | Path = "/proc/meminfo",
    proc_self_cgroup: str | Path = "/proc/self/cgroup",
    cgroup_root: str | Path = "/sys/fs/cgroup",
    logical_cpus: int | None = None,
) -> SystemCapacity:
    cpus = max(1, logical_cpus if logical_cpus is not None else _logical_cpu_count())
    threads = cpus - 1 if cpus > 2 else cpus

    if ram_bytes is not None:
        if ram_bytes <= 0:
            raise ValueError("ram_bytes must be positive")
        return SystemCapacity(
            available_bytes=ram_bytes,
            host_available_bytes=None,
            cgroup_remaining_bytes=None,
            cgroup_path=None,
            logical_cpus=cpus,
            recommended_threads=threads,
            memory_source="operator override",
        )

    host = _host_available(Path(proc_meminfo)) or _sysconf_available()
    cgroup, cgroup_path = _cgroup_remaining(
        Path(cgroup_root), _unified_cgroup_path(Path(proc_self_cgroup))
    )
    candidates = [value for value in (host, cgroup) if value is not None]
    if not candidates:
        raise RuntimeError("could not determine available memory; use --ram-gib")
    available = min(candidates)
    source = "host MemAvailable"
    if cgroup is not None and (host is None or cgroup <= host):
        source = "remaining cgroup memory"
    return SystemCapacity(
        available_bytes=available,
        host_available_bytes=host,
        cgroup_remaining_bytes=cgroup,
        cgroup_path=cgroup_path,
        logical_cpus=cpus,
        recommended_threads=threads,
        memory_source=source,
    )
