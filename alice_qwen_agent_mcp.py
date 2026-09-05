"""Qwen-Agent driving Alice through Alice's own MCP server, on local Qwen.

Alice (github.com/gbolask24/alice) exposes its 13 copilot tools over stdio MCP
(scripts/mcp-server.ts in that repo). Qwen-Agent starts that server, loads the
tools, and the local model calls them. Writes follow Alice's policy: with
ALICE_MCP_WRITE_POLICY=preview (default) destructive tools return the
confirmation card and change nothing.

Run: python alice_qwen_agent_mcp.py
Env: ALICE_DIR (default ~/Downloads/Web Project/Content Generator), ALICE_DB_PATH
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
ALICE_DIR = os.getenv("ALICE_DIR", os.path.expanduser("~/Downloads/Web Project/Content Generator"))
ALICE_DB = os.getenv("ALICE_DB_PATH", "./data/alice.db")
POLICY = os.getenv("ALICE_MCP_WRITE_POLICY", "preview")

MCP_TOOLS = [{
    "mcpServers": {
        "alice": {
            "command": "zsh",
            "args": ["-c", f"cd '{ALICE_DIR}' && pnpm --silent tsx scripts/mcp-server.ts"],
            "env": {"DB_PATH": ALICE_DB, "ALICE_MCP_WRITE_POLICY": POLICY, "PATH": os.environ["PATH"]},
        }
    }
}]

PROMPTS = [
    "Which content generators does Alice have for LinkedIn? Names only.",
    "Create a task in Alice titled 'Judgement panel post' for the linkedin channel, with the content "
    "'Draft a LinkedIn post on why a judgement panel records verdicts before members confer.'",
    "Show me my open tasks in Alice.",
]


def main() -> None:
    generate_cfg = {"temperature": 0.3, "use_raw_api": True}
    if REASONING_EFFORT:
        generate_cfg["reasoning_effort"] = REASONING_EFFORT
    bot = Assistant(
        llm={"model": MODEL, "model_server": BASE_URL, "api_key": API_KEY, "generate_cfg": generate_cfg},
        system_message="You are working inside Alice, a content studio. Use the Alice tools to answer; "
                       "do not invent data. Report what the tool returned.",
        function_list=MCP_TOOLS,
        name="Alice via Qwen-Agent",
    )
    messages = []
    for prompt in PROMPTS:
        messages.append({"role": "user", "content": prompt})
        t0 = time.perf_counter()
        final = []
        for final in bot.run(messages):
            pass
        messages.extend(final)
        print(f"\n### {prompt}\n({time.perf_counter() - t0:.1f}s)")
        for m in final:
            if m.get("function_call"):
                print("  tool call:", m["function_call"]["name"], m["function_call"]["arguments"][:200])
            elif m.get("role") == "function":
                print("  tool result:", str(m.get("content"))[:200].replace("\n", " "))
            elif m.get("content"):
                print("  answer:", str(m["content"])[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
