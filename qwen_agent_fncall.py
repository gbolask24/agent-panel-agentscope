"""Qwen-Agent function calling against the local Qwen endpoint.

Minimal: one registered tool, one Assistant, one prompt. Prints the messages
Qwen-Agent produced, including the function_call it parsed out of the model.
Run: python qwen_agent_fncall.py
"""
import json
import os
import time

import json5
from dotenv import load_dotenv
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

load_dotenv()
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "none") or None


@register_tool("get_weather")
class GetWeather(BaseTool):
    description = "Get the current weather for a city."
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city", "unit"],
    }

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params) if isinstance(params, str) else params
        return json.dumps({"city": args.get("city"), "temp": 17, "unit": args.get("unit"), "sky": "overcast"})


def main() -> None:
    generate_cfg = {"temperature": 0.3}
    if REASONING_EFFORT:
        generate_cfg["reasoning_effort"] = REASONING_EFFORT  # passed straight through to the API
    if os.getenv("QWEN_AGENT_USE_RAW_API"):
        generate_cfg["use_raw_api"] = True  # let the server parse tool calls instead of Qwen-Agent
    bot = Assistant(
        llm={"model": MODEL, "model_server": BASE_URL, "api_key": API_KEY, "generate_cfg": generate_cfg},
        system_message="You are a helpful assistant. Use tools when they fit.",
        function_list=["get_weather"],
    )
    t0 = time.perf_counter()
    final = []
    for final in bot.run([{"role": "user", "content": "What's the weather in Manchester right now, in celsius?"}]):
        pass
    print(f"model={MODEL} base_url={BASE_URL} use_raw_api={bool(generate_cfg.get('use_raw_api'))} "
          f"latency={time.perf_counter() - t0:.1f}s")
    for m in final:
        print(json.dumps(m, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
