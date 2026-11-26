# Arena-size A/B at 128K context: Qwen3.6-35B-A3B on Apple Silicon (Metal)

Measured on 2026-11-26. Follow-up to `2026-11-26-q36-arena-ab` (same method,
`--max-context 8192`): the server moved to `--max-context 131072` so omp's own
system prompt (~16K tokens) fits, and the arena sweep was rerun because the
larger KV cache changes unified-memory pressure. Result: **arena 16 GB + MTP 2
still wins** (decode 35.3 tok/s, identical to the 8K-context figure).

Single runs throughout (not medians); treat deltas under ~10% as noise. The
adopted deltas are 2-3x. Prefill/decode rates are the engine's own
`prompt_per_second` / `predicted_per_second` from `/v1/chat/completions`
`timings`.

## Hardware and software

- Machine: Mac14,13 identifier, 12 CPUs, 64 GiB unified memory; macOS 27.2; storage: external SSD (`/Volumes/Samsung970EVOPlus`).
- strata-q36 past `4c7d2a3` (this report's rerun tooling); engine binary
  `engine/build-metal/strata-q36-metal`, server `terminal/serve/server.py`, Metal backend.
- Background load: only the benchmarked server. No engine rebuild between the 8K and 128K sweeps.

## Model and configuration

- Unsloth `Qwen3.6-35B-A3B-MTP-GGUF`, `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` + `.experts`, `pack/qwen36-ud-q4ks/` (tokenizer + `expert-profile.bin`), `--stream`, `reasoning_effort none`, `temperature 0.0`, `parallel 2`.
- Fixed for this sweep: `--native --max-context 131072`. Only `--arena-gb` and `--mtp` varied, via `ab_run.sh` (copy in this folder; the `<max_context>` argument pins 131072 per cell, so a later live-config change cannot mislabel a rerun):

```bash
cd <this folder>
./ab_run.sh 4 2 ctx128k-arena4 131072
./ab_run.sh 8 2 ctx128k-arena8 131072
./ab_run.sh 12 2 ctx128k-arena12 131072
./ab_run.sh 16 2 ctx128k-arena16 131072
./ab_run.sh 16 3 ctx128k-arena16-mtp3 131072
```

One cold server per cell; `ab_run.sh` builds each variant config from the live
canonical JSON, waits for `/health`, runs `ab_probe.py`, then stops its server.

## Results

Raw per-run lines: `runs.tsv` (this folder).

| Configuration | Cold prefill (285 tok) | Warm prefill (29 new + 256 cached) | Decode 256 tok | MTP accepted |
| --- | --- | --- | --- | --- |
| arena 4 / mtp 2 | 14.5s, 19.7 tok/s | 2.7s | 24.1s, 10.6 tok/s | 141/228 (62%) |
| arena 8 / mtp 2 | 9.8s, 29.1 tok/s | 1.9s | 11.4s, 22.5 tok/s | 141/228 |
| arena 12 / mtp 2 | 6.9s, 41.0 tok/s | 1.4s | 9.0s, 28.4 tok/s | 141/228 |
| **arena 16 / mtp 2 (kept)** | **4.3s, 65.5 tok/s** | **0.9s** | **7.2s, 35.3 tok/s** | 141/228 |
| arena 16 / mtp 3 | 4.3s, 66.3 tok/s | 0.8s | 8.3s, 30.8 tok/s | 148/321 (46%) |

MTP drafts are bit-stable across arenas at temperature 0 (141/228), so the
decode gains are cache residency. MTP-3 accepts a lower fraction and decodes
slower: rejected again. The 128K winner matches the 8K winner exactly
(35.3 tok/s decode), so the larger KV cache did not move the optimum.

## omp registration

`~/.omp/agent/models.yml` provider `strata` now declares `contextWindow:
131072`, `maxTokens: 131064` (from `/v1/models` `limit` at this context).
`omp -p --model strata/qwen3.6-35b-a3b "Reply with only OK"` passes where the
8192-context setup failed with `request exceeds the context window`.

## Memory observations

128K context raises the KV working set, but the ranking and winner are
unchanged from 8K; swap behavior to be re-sampled on the adopted config under
idle + load (see note below). If `vm.swapusage used` keeps climbing with other
workloads beside the server, fall back to arena 12 (~2.7x the old decode rate).

## Correctness and limitations

- Single runs, one machine, UD-Q4_K_S, 131072 context. Not a claim about other
  hardware, quants, or longer prompts.
- CodeAct regression at 128K (`examples/codeact_demo.py`, adopted config):
  `verdict=PASS expected=16566 elapsed_s=49.11 input_tokens=2231
  output_tokens=374 blended_tok/s=53.0`, exit 0 — same answer and tokens as the
  8K runs (53.37 s), context bump cost nothing on this task.
- Long-prompt prefill at 128K context (2026-11-27, calibrated via
  `/v1/messages/count_tokens`): 9,776 tokens pass (168.6 s, 58.0 tok/s) and
  16,355 tokens pass (331.3 s, 49.4 tok/s, answer correct) — omp-scale (~16K)
  system prompts are inside the working range.
- A 19,528-token prompt dies twice with HTTP 503 (`mtl: command buffer failed:
  Impacting Interactivity` in the engine log — macOS kills one long-running
  Metal command buffer), identically with and without the vision encoder
  loaded, so this is not vision memory. Progress lines show steady 64-token-step
  prefill before the kill. The ceiling is between 16,355 and 19,528 tokens, not
  at 16K. Engine-side prefill yield/smaller dispatch would be needed to raise
  it; `STRATA_Q36_CKPT_*` only moves checkpoint cadence, not buffer length.
- At failure time: ~60 MB free, swap 2.5/4 GB used; the engine's 17.2 GB wired
  arena is released on crash (35 GB free after). Short prompts keep working —
  the server restarts the engine on the next request.
- The discarded 32K sweep (interrupted mid-run, port contention with the config
  change) is not evidence and is not published here.
