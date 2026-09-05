"""Qwen-Agent with an MCP server (stdio) against the local Qwen endpoint.

Uses the reference time server (uvx mcp-server-time). Qwen-Agent starts the
server, lists its tools, and exposes them to the model as functions.
Run: python qwen_agent_mcp.py
"""
import json
import os
import time

from dotenv import load_dotenv
from qwen_agent.agents import Assistant

load_dotenv()
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "none") or None

MCP_TOOLS = [{
    "mcpServers": {
        "time": {"command": "uvx", "args": ["mcp-server-time", "--local-timezone=Europe/London"]},
    }
}]


def main() -> None:
    generate_cfg = {"temperature": 0.3}
    if REASONING_EFFORT:
        generate_cfg["reasoning_effort"] = REASONING_EFFORT
    bot = Assistant(
        llm={"model": MODEL, "model_server": BASE_URL, "api_key": API_KEY, "generate_cfg": generate_cfg},
        system_message="You are a helpful assistant. Use tools when they fit.",
        function_list=MCP_TOOLS,
    )
    t0 = time.perf_counter()
    final = []
    for final in bot.run([{"role": "user", "content": "What time is it in Tokyo right now, and how far ahead of London is that?"}]):
        pass
    print(f"model={MODEL} latency={time.perf_counter() - t0:.1f}s")
    for m in final:
        print(json.dumps(m, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
