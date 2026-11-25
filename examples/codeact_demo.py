# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "smolagents[litellm]",
# ]
# ///

import argparse
import os

from smolagents import CodeAgent, LiteLLMModel, tool


@tool
def get_exchange_rate(currency: str) -> float:
    """查詢指定幣別兌換台幣 (TWD) 的匯率。

    Args:
        currency: 幣別代碼，例如 'USD' 或 'JPY'。
    """
    rates = {"USD": 32.5, "JPY": 0.21, "EUR": 35.2}
    return rates.get(currency.upper(), 1.0)


def main():
    ap = argparse.ArgumentParser(description="CodeAct demo backed by Strata (Qwen3.6-35B-A3B) or LM Studio")
    ap.add_argument("--lmstudio", action="store_true", help="使用本地 LM Studio (http://localhost:1234/v1)")
    ap.add_argument("--api-base", default=os.environ.get("CODEACT_API_BASE", "http://127.0.0.1:8095/v1"))
    ap.add_argument("--model", default=os.environ.get("CODEACT_MODEL_ID", "openai/qwen3.6-35b-a3b"))
    ap.add_argument("--api-key", default=os.environ.get("CODEACT_API_KEY", "strata"))
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    api_base = "http://localhost:1234/v1" if args.lmstudio else args.api_base
    model_id = "openai/local-model" if args.lmstudio else args.model
    api_key = "lm-studio" if args.lmstudio else args.api_key

    model = LiteLLMModel(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        temperature=args.temperature,
    )

    agent = CodeAgent(
        tools=[get_exchange_rate],
        model=model,
        additional_authorized_imports=["math"],
    )

    result = agent.run(
        "我有 100 USD、50000 JPY 和 80 EUR，請幫我換算成台幣並加總，最後無條件進位到整數。"
    )
    print("最終結果：", result)


if __name__ == "__main__":
    main()
