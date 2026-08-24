"""Command-line interface for GGUFGauge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

from .estimate import KV_BYTES, estimate_launch, memory_for_context
from .gguf import GGUFError, read_model_info
from .system import detect_capacity


def _bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} TiB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ggufgauge",
        description="Measure a GGUF launch envelope against process-visible RAM and CPUs.",
    )
    parser.add_argument("model", help="path to a GGUF model file")
    parser.add_argument("--ram-gib", type=float, help="override detected available RAM")
    parser.add_argument("--reserve-percent", type=float, default=15.0)
    parser.add_argument("--kv-type", choices=tuple(KV_BYTES), default="f16")
    parser.add_argument(
        "--context", type=int, action="append", default=[], help="context to check"
    )
    parser.add_argument("--threads", type=int, help="override affinity-derived threads")
    parser.add_argument("--executable", default="llama-server")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--emit-command", action="store_true", help="print only the recommended command"
    )
    return parser


def _command(
    executable: str, model_path: str, context: int | None, threads: int, kv_type: str
) -> str | None:
    if context is None:
        return None
    return shlex.join(
        [
            executable,
            "-m",
            model_path,
            "-c",
            str(context),
            "-t",
            str(threads),
            "-ctk",
            kv_type,
            "-ctv",
            kv_type,
        ]
    )


def run(arguments: argparse.Namespace) -> int:
    if arguments.ram_gib is not None and arguments.ram_gib <= 0:
        raise ValueError("--ram-gib must be positive")
    if arguments.threads is not None and arguments.threads <= 0:
        raise ValueError("--threads must be positive")
    if any(context <= 0 for context in arguments.context):
        raise ValueError("--context values must be positive")

    model = read_model_info(arguments.model)
    override = (
        int(arguments.ram_gib * 1024**3) if arguments.ram_gib is not None else None
    )
    capacity = detect_capacity(ram_bytes=override)
    threads = arguments.threads or capacity.recommended_threads
    estimate = estimate_launch(
        model,
        capacity,
        kv_type=arguments.kv_type,
        reserve_percent=arguments.reserve_percent,
    )
    command = _command(
        arguments.executable,
        str(Path(arguments.model)),
        estimate.recommended_context,
        threads,
        arguments.kv_type,
    )

    checks: list[dict[str, object]] = []
    for context in sorted(set(arguments.context)):
        total = memory_for_context(estimate, context)
        checks.append(
            {
                "context": context,
                "estimated_total_bytes": total,
                "fits": total is not None and total <= capacity.available_bytes,
                "within_trained_context": (
                    model.context_length is None or context <= model.context_length
                ),
            }
        )

    payload = {
        "model": model.to_dict(),
        "capacity": capacity.to_dict(),
        "estimate": estimate.to_dict(),
        "threads": threads,
        "command": command,
        "context_checks": checks,
    }
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif arguments.emit_command:
        if command:
            print(command)
        else:
            print("no safe command could be derived", file=sys.stderr)
    else:
        print("GGUFGauge preflight")
        print(f"model: {model.name or Path(model.path).name} ({model.architecture})")
        print(f"file: {_bytes(model.file_bytes)}; GGUF v{model.version}")
        print(
            f"usable RAM: {_bytes(capacity.available_bytes)} "
            f"from {capacity.memory_source}"
        )
        print(
            f"reserve: {_bytes(estimate.reserve_bytes)}; "
            f"model envelope: {_bytes(estimate.estimated_model_bytes)}; "
            f"runtime overhead: {_bytes(estimate.runtime_overhead_bytes)}"
        )
        print(f"KV per token ({estimate.kv_type}): {_bytes(estimate.kv_bytes_per_token)}")
        print(f"safe context: {estimate.recommended_context or 'not derivable'}")
        print(
            f"threads: {threads} of {capacity.logical_cpus} process-visible logical CPUs"
        )
        if command:
            print(f"command: {command}")
        for check in checks:
            print(
                f"context {check['context']}: {_bytes(check['estimated_total_bytes'])}; "
                f"fits={str(check['fits']).lower()}; "
                f"trained-limit={str(check['within_trained_context']).lower()}"
            )
        for warning in estimate.warnings:
            print(f"warning: {warning}")
    return 0 if command else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except (GGUFError, RuntimeError, ValueError) as exc:
        print(f"ggufgauge: error: {exc}", file=sys.stderr)
        return 2
