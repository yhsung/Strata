# Arena-size A/B and CodeAct regression: Qwen3.6-35B-A3B on Apple Silicon (Metal)

Measured on 2026-11-26. Question: which `--arena-gb` (4/8/12/16) and `--mtp` (2/3)
setting to keep for the Qwen3.6-35B-A3B server in `/Volumes/Samsung970EVOPlus/dev-projects/codeact`.
Result: **arena 16 GB + MTP 2** (locked into `strata_qwen36_config.json`).
CodeAct end-to-end: 144.65 s -> 53.37 s (2.7x), answer still correct (16566, PASS).

Single runs throughout (not medians); treat deltas under ~10% as noise. The
adopted deltas are 2-3x, well above that. Prefill/decode rates are the engine's
own `prompt_per_second` / `predicted_per_second` from `/v1/chat/completions`
`timings`.

## Hardware and software

- Machine: Mac14,13 identifier, 12 CPUs, 64 GiB unified memory; macOS 27.2; storage: external SSD (`/Volumes/Samsung970EVOPlus`).
- GPU: Apple Silicon integrated GPU via the Metal engine (no discrete GPU, no VRAM figure to report; unified memory is the constraint).
- strata-q36 at `6a2fcef` (2026-11-25): engine binary `engine/build-metal/strata-q36-metal`, server `terminal/serve/server.py`, Metal backend.
- Background load: only the benchmarked server; nothing else heavy. Engine RSS ~20.5 GiB at arena 16 (idle snapshot).

## Model and configuration

- Model: Unsloth `Qwen3.6-35B-A3B-MTP-GGUF`, `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`
  (21,388,319,008 bytes) + `.experts` (18,327,027,712 bytes); `mmproj-F32.gguf` unused (images off).
- Pack: `pack/qwen36-ud-q4ks/` (tokenizer + `expert-profile.bin`, used in every run).
- Fixed flags: `--native --max-context 8192 --stream`, `reasoning_effort none`,
  `temperature 0.0`, `parallel 2`. Only `--arena-gb` and `--mtp` varied:

```bash
STRATA_ROOT="/Volumes/Samsung970EVOPlus/dev-projects/strata-q36"  # all paths absolute; run from anywhere
```

`ab_run.sh` (this folder, executable) runs one A/B cell: it copies the live
canonical config, rewrites only `--arena-gb` / `--mtp`, restarts the server on
port 8095, waits for `/health`, runs `ab_probe.py`, then stops the server it
started. Usage: `./ab_run.sh <arena_gb> <mtp_depth> <label> [canonical_config] [port]`.
The full sweep that produced `runs.tsv`:

```bash
cd /Volumes/Samsung970EVOPlus/dev-projects/strata-q36/docs/bench-results/2026-11-26-q36-arena-ab
./ab_run.sh 4 2 baseline-arena4-mtp2
./ab_run.sh 8 2 arena8-mtp2
./ab_run.sh 12 2 arena12-mtp2
./ab_run.sh 16 2 arena16-mtp2
./ab_run.sh 16 3 arena16-mtp3
```

One cold server per cell (the script always restarts it); two resident engines
would contend for unified memory and invalidate the comparison.

## Method

`ab_probe.py` (copy in this folder) sends three fixed requests per configuration
and prints the server's `timings`:

1. `cold`: fixed 285-token prompt, fresh server (prefix cache empty), `max_tokens 8`.
2. `warm`: identical prompt again (256 tokens served from the prefix cache).
3. `decode`: fixed 90-token prompt, `max_tokens 256` (measures sustained decode + MTP draft acceptance).

CodeAct regression (`codeact_demo.py`, smolagents `CodeAgent` + `LiteLLMModel`
against `http://127.0.0.1:8095/v1`, model id `openai/qwen3.6-35b-a3b`): one
exchange-rate tool-use task, expected total 16566. The script times `agent.run()`,
sums `token_usage` over `agent.memory.steps`, verifies the answer with
`codeact_check.verify_answer`, prints `format_report`, and exits 0/1.
`test_codeact_check.py`: 5 unit tests for the verifier/reporter.

CodeAct regression on the winner (server running with the adopted config):

```bash
cd /Volumes/Samsung970EVOPlus/dev-projects/codeact && uv run --script codeact_demo.py
# expect: verdict=PASS expected=16566 ... and exit code 0
```

## Results

Raw per-run lines: `runs.tsv` (this folder).

| Configuration | Cold prefill (285 tok) | Warm prefill (29 new + 256 cached) | Decode 256 tok | MTP accepted |
| --- | --- | --- | --- | --- |
| arena 4 / mtp 2 (old) | 11,024 ms, 25.9 tok/s | 2,096 ms | 19,818 ms, 12.9 tok/s | 141/228 (62%) |
| arena 8 / mtp 2 | 6,996 ms, 40.7 tok/s | 1,413 ms | 10,176 ms, 25.2 tok/s | 141/228 |
| arena 12 / mtp 2 | 5,405 ms, 52.7 tok/s | 1,125 ms | 8,553 ms, 30.0 tok/s | 141/228 |
| **arena 16 / mtp 2 (adopted)** | **4,125 ms, 69.1 tok/s** | **838 ms** | **7,250 ms, 35.3 tok/s** | 141/228 |
| arena 16 / mtp 3 | 4,141 ms, 68.8 tok/s | 793 ms | 8,337 ms, 30.7 tok/s | 148/321 (46%) |

MTP draft counts are bit-stable across arena sizes at temperature 0 (141/228),
so the decode gains are cache residency, not luck. MTP-3 accepts a lower fraction
and decodes slower than MTP-2: rejected.

CodeAct regression on the adopted config:

| Run | verdict | elapsed s | input / output tok | blended tok/s |
| --- | --- | ---: | ---: | ---: |
| arena 4 (before) | PASS (16566) | 144.65 | 2,231 / 374 | 18.0 |
| arena 16 (after) | PASS (16566) | 53.37 | 2,231 / 374 | 48.8 |

Token counts are identical across runs (same tool calls), so the 2.7x is pure speed.

## Memory observations

`vm_stat` / `vm.swapusage` snapshots (unified memory, GiB):

- arena 4 server running: ~3.4 GiB free.
- arena 16 under probe load: ~0.1 GiB free, swap 795/2048 MiB used; swap-out
  counter frozen after warm-up and swap-ins negligible (+76) — cold pages evicted,
  no thrash; throughput rose monotonically, consistent with that.
- arena 16 idle afterwards: engine RSS 20.5 GiB, ~2.9 GiB free, swap unchanged.

If this machine later runs other large-memory workloads beside the server and
`vm.swapusage used` keeps climbing, fall back to arena 12 (still ~2.3x the old
decode rate).

## Correctness and limitations

- Answer check: the demo's arithmetic (100x32.5 + 50000x0.21 + 80x35.2 = 16566)
  verified programmatically; both runs PASS. No needle/recall or HumanEval run:
  this report covers serving throughput + one agentic smoke task only.
- Single runs, one machine, one quantization (UD-Q4_K_S), context 8192. Not a
  claim about other hardware, quants, or long contexts.
- Prefix-cache note (unplanned finding): a 323-token shared prefix served only
  256 cached tokens (block granularity); multi-turn agents re-read up to ~255
  shared-prefix tokens per turn. Not changed; candidate follow-up.
- The hand-made variant configs were deleted; `ab_run.sh` regenerates them
  from the live config on every run, so there is nothing stale to keep. The
  adopted `strata_qwen36_config.json` (arena 16) is the live config, not stored
  here (absolute local paths). `ab_probe.py` is copied here; the demo/check
  scripts live in the codeact dir and are referenced, not forked.
- Revalidation 2026-11-26 (`./ab_run.sh 16 2 verify-arena16`): cold 4,295 ms /
  66.4 tok/s, decode 7,330 ms / 34.9 tok/s, MTP 141/228 — matches the adopted row.
