# GGUFGauge

GGUFGauge answers a practical local-AI question: **will this GGUF survive on this machine or inside this container, and what conservative command should I run?**

It reads bounded GGUF metadata, detects Linux cgroup memory limits and CPU affinity, reserves operating-system headroom, estimates KV-cache growth, selects a safe context, and emits a ready-to-paste `llama-server` command. It is an offline preflight—not a benchmark result—and uses only the Python standard library.

## Why this is different

Static calculators make you enter a RAM or VRAM number. GGUFGauge measures the memory actually available to the current process, including container limits and current cgroup usage. Its output is an actionable launch envelope for CPU/local use:

- actual process-visible RAM budget;
- GGUF architecture, layer, head, and trained-context metadata;
- explicit reserve and estimation assumptions;
- recommended context and thread count;
- shell-escaped `llama-server` command;
- stable JSON output for scripts and deployment checks.

It does not promise that an estimate equals runtime allocation. Backend buffers, model architecture, mmap behavior, batch size, GPU offload, and cache implementation can change real usage.

## Run

```bash
python -m ggufgauge /models/model.gguf
python -m ggufgauge /models/model.gguf --json
python -m ggufgauge /models/model.gguf --kv-type q8_0 --emit-command
```

Useful overrides:

```bash
python -m ggufgauge model.gguf --ram-gib 12 --reserve-percent 20
python -m ggufgauge model.gguf --context 2048 --context 4096 --context 8192
python -m ggufgauge model.gguf --executable /opt/llama.cpp/llama-server
```

`--ram-gib` is for planning another machine. Without it, the detector uses the smaller of host `MemAvailable` and remaining cgroup memory. The recommended context never exceeds the GGUF training context when that metadata is present.

## Supported metadata

GGUFGauge reads GGUF versions 2 and 3 and derives common architecture keys such as `*.block_count`, `*.embedding_length`, `*.attention.head_count`, `*.attention.head_count_kv`, and `*.context_length`. Unknown or recurrent architectures are reported honestly when the KV-cache shape cannot be derived.

## Test

```bash
python -m unittest discover -s tests -v
```

The tests build tiny synthetic GGUF headers; no model download is required.

## Fund more development

Donations directly fund additional RequiredTruth development time and increase how much work can be produced. Bitcoin, Ethereum/EVM, and Dogecoin receiving addresses are listed in [`SUPPORT.md`](SUPPORT.md).

After a confirmed donation, you may open a GitHub issue with the asset, network, transaction hash, and the specific area you want expanded. The first issue claiming an unclaimed confirmed inbound transaction is accepted as its request attribution. A transaction hash is public and this rule is operational attribution, not cryptographic proof of wallet ownership. Never post a private key or seed phrase.

Apache-2.0 licensed.
