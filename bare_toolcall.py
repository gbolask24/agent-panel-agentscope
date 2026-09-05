"""Bare OpenAI-SDK tool call against the local Ollama endpoint.

Proves the model returns a structured tool call, not tool-call text in content.
Run: python bare_toolcall.py
"""
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the weather in Manchester right now, in celsius?"}],
        tools=TOOLS,
        tool_choice="auto",
    )
    elapsed = time.perf_counter() - t0
    msg = resp.choices[0].message
    print(f"model={MODEL} base_url={BASE_URL} latency={elapsed:.1f}s")
    print("finish_reason:", resp.choices[0].finish_reason)
    print("content:", repr(msg.content)[:300])
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print("TOOL CALL:", tc.function.name, tc.function.arguments)
            json.loads(tc.function.arguments)
        print("RESULT: structured tool call received")
    else:
        print("RESULT: no structured tool call (see content above)")


if __name__ == "__main__":
    main()
