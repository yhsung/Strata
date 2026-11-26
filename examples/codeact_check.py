"""Shared check helpers for the CodeAct demo (stdlib only, safe to import in tests).

EXPECTED_TOTAL: 100*32.5 + 50000*0.21 + 80*35.2 = 16566.
"""

import re

EXPECTED_TOTAL = 16566


def verify_answer(result) -> bool:
    """True when the agent's answer carries the expected total (int, float, or str)."""
    if result is None:
        return False
    if isinstance(result, bool):
        return False
    if isinstance(result, (int, float)):
        return result == EXPECTED_TOTAL
    found = re.findall(r"\d+", str(result))
    return any(int(x) == EXPECTED_TOTAL for x in found)


def format_report(ok: bool, elapsed_s: float, input_tokens: int, output_tokens: int) -> str:
    """One comparable block: verdict, wall time, tokens, blended throughput."""
    total = input_tokens + output_tokens
    rate = total / elapsed_s if elapsed_s > 0 else 0.0
    return (
        f"[codeact] verdict={'PASS' if ok else 'FAIL'} expected={EXPECTED_TOTAL} "
        f"elapsed_s={elapsed_s:.2f} input_tokens={input_tokens} output_tokens={output_tokens} "
        f"blended_tok/s={rate:.1f}"
    )
