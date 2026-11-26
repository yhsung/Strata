"""Fixed-probe A/B for the Strata Qwen3.6 server (stdlib only).

Sends a fixed prefill-heavy prompt twice (cold + warm prefix cache) and one
decode-heavy request, prints one TSV line per phase with the server timings.

Run: python3 ab_probe.py --port 8095 --label baseline
"""

import argparse
import json
import urllib.request

BODY = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod " * 24).strip()
PROMPT = f"Prefill probe {BODY} Please reply with exactly: DONE"
DECODE_PROMPT = "Write a 200-word story about a modem engineer. " * 6


def post(port, content, max_tokens, tag):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "qwen3.6-35b-a3b",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    t = d.get("timings", {})
    u = d.get("usage", {})
    print(
        f"{tag}\tprompt_n={u.get('prompt_tokens')}\tcache_n={t.get('cache_n')}\t"
        f"prompt_ms={t.get('prompt_ms')}\tprompt_tps={t.get('prompt_per_second')}\t"
        f"pred_n={t.get('predicted_n')}\tpred_ms={t.get('predicted_ms')}\t"
        f"pred_tps={t.get('predicted_per_second')}\t"
        f"draft={t.get('draft_n_accepted')}/{t.get('draft_n')}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()
    post(args.port, PROMPT, 8, f"{args.label}\tcold")
    post(args.port, PROMPT, 8, f"{args.label}\twarm")
    post(args.port, DECODE_PROMPT, 256, f"{args.label}\tdecode")


if __name__ == "__main__":
    main()
