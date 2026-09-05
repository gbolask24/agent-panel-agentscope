"""AgentScope 2.0 driving Alice through Alice's MCP server, on local Qwen.

Same server and prompts as alice_qwen_agent_mcp.py, through AgentScope 2.0's
unified MCPClient and Agent. Run: python alice_agentscope_mcp.py

AgentScope 2.0 runs a permission engine on every tool call. In the default
mode any tool not marked read-only is held with a RequireUserConfirmEvent
until something answers with a UserConfirmResultEvent. Here `boundary()`
plays the human: it approves Alice's undoable creates and refuses the
destructive tools, and every decision is printed. That is the framework's
own version of the approval boundary, sitting in front of Alice's.
"""
import asyncio
import os
import time

from dotenv import load_dotenv
from agentscope.agent import Agent, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.event import ConfirmResult, RequireUserConfirmEvent, UserConfirmResultEvent
from agentscope.mcp import MCPClient, StdioMCPConfig
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

load_dotenv()
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "none") or None
ALICE_DIR = os.getenv("ALICE_DIR", os.path.expanduser("~/Downloads/Web Project/Content Generator"))
ALICE_DB = os.getenv("ALICE_DB_PATH", "./data/alice.db")
POLICY = os.getenv("ALICE_MCP_WRITE_POLICY", "preview")

APPROVE_PREFIXES = ("create_", "save_", "add_")  # AgentScope names MCP tools mcp__<server>__<tool>


def boundary(evt: RequireUserConfirmEvent) -> UserConfirmResultEvent:
    """Decide each held tool call. Approve undoable creates, refuse the rest."""
    results = []
    for tc in evt.tool_calls:
        ok = tc.name.split("__")[-1].startswith(APPROVE_PREFIXES)
        print(f"  [boundary] {tc.name} {str(tc.input)[:120]} -> {'APPROVED' if ok else 'REFUSED'}")
        results.append(ConfirmResult(confirmed=ok, tool_call=tc))
    return UserConfirmResultEvent(reply_id=evt.reply_id, confirm_results=results)


async def turn(agent: Agent, prompt: str):
    """One user turn, answering permission requests until the reply finishes."""
    pending = UserMsg("user", prompt)
    while True:
        held = None
        final = None
        async for item in agent.reply_stream(pending, yield_final_msg=True):
            if isinstance(item, RequireUserConfirmEvent):
                held = item
            elif hasattr(item, "get_text_content"):
                final = item
        if held is None:
            return final
        pending = boundary(held)


PROMPTS = [
    "Which content generators does Alice have for LinkedIn? Names only.",
    "Create a task in Alice titled 'Judgement panel post' for the linkedin channel, with the content "
    "'Draft a LinkedIn post on why a judgement panel records verdicts before members confer.'",
    "Show me my open tasks in Alice.",
]


async def main() -> None:
    alice = MCPClient(
        name="alice",
        is_stateful=True,
        mcp_config=StdioMCPConfig(
            command="pnpm", args=["--silent", "tsx", "scripts/mcp-server.ts"], cwd=ALICE_DIR,
            env={"DB_PATH": ALICE_DB, "ALICE_MCP_WRITE_POLICY": POLICY, "PATH": os.environ["PATH"]},
        ),
    )
    await alice.connect()
    try:
        toolkit = Toolkit(mcps=[alice])
        model = OpenAIChatModel(
            credential=OpenAICredential(api_key=API_KEY, base_url=BASE_URL),
            model=MODEL, stream=False,
            extra_body={"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else None,
        )
        agent = Agent(
            name="Alice via AgentScope",
            system_prompt="You are working inside Alice, a content studio. Use the Alice tools to answer; "
                          "do not invent data. Report what the tool returned.",
            model=model, toolkit=toolkit, react_config=ReActConfig(max_iters=4, stop_on_reject=True),  # a refusal ends the attempt, no retry loop
        )
        for prompt in PROMPTS:
            t0 = time.perf_counter()
            reply = await turn(agent, prompt)
            print(f"\n### {prompt}\n({time.perf_counter() - t0:.1f}s)")
            print("  answer:", ((reply.get_text_content() if reply else "") or "")[:400].replace("\n", " "))
    finally:
        await alice.close()


if __name__ == "__main__":
    asyncio.run(main())
