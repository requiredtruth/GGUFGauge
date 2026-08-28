# Changelog

## Unreleased

- Fixed process-visible RAM detection for nested cgroup v2 scopes, including tighter ancestor limits.
- Added the limiting cgroup path to deterministic JSON output.

## 0.1.0 - 2026-08-24

- Added bounded GGUF v2/v3 metadata parsing without external dependencies.
- Added host and Linux cgroup-aware memory detection.
- Added conservative model, KV-cache, and context estimates.
- Added ready-to-paste `llama-server` command and JSON output.
- Added synthetic GGUF tests that require no model download.
