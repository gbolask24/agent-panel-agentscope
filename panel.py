"""Three-agent judgement panel on AgentScope 1.x MsgHub.

Researcher and Critic assess a piece of marketing copy independently. Their
verdicts are written to the log before either sees the other's view. Then one
exchange round runs through the hub, and the Chair records a final verdict.

Run: python panel.py [path/to/input.md]
Model config comes from .env (see .env.example).
"""
import asyncio
import datetime as dt
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import MsgHub
from agentscope.tool import Toolkit

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
# "none" switches Qwen3.5 thinking off through Ollama's OpenAI endpoint.
# Set LLM_REASONING_EFFORT= (empty) for models that reject the parameter.
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "none") or None

RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
LOG_PATH = Path("runs") / f"panel-{RUN_ID}.jsonl"
PHASE = {"name": "setup"}  # mutable so the logging hook can read the current phase


# ---------- log: every message, with agent, phase, timestamp, content ----------

def log(agent: str, role: str, content, metadata=None, event: str = "message") -> None:
    record = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": RUN_ID,
        "model": MODEL,
        "phase": PHASE["name"],
        "event": event,
        "agent": agent,
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_reply_hook(agent: ReActAgent, kwargs: dict, output: Msg):
    """AgentScope post_reply hook: log whatever an agent says, in any phase."""
    log(agent.name, output.role, output.get_text_content(), output.metadata)
    return None


# ---------- structured verdicts ----------

class Verdict(BaseModel):
    verdict: Literal["approve", "revise", "reject"] = Field(
        description="approve = publish as is; revise = fixable problems; reject = start again"
    )
    confidence: int = Field(ge=1, le=5, description="1 = guess, 5 = certain")
    top_issue: str = Field(description="The single most important problem, one sentence")
    reasoning: str = Field(description="Three to five sentences of reasoning")


class ChairVerdict(BaseModel):
    verdict: Literal["approve", "revise", "reject"]
    confidence: int = Field(ge=1, le=5)
    agreed: str = Field(description="What Researcher and Critic agreed on")
    disputed: str = Field(description="Where they disagreed and how the chair resolved it")
    reasoning: str = Field(description="Three to five sentences of reasoning")
    dissent: str = Field(description="Any dissent the chair overruled, or 'none'")


# ---------- agents ----------

def make_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=MODEL,
        api_key=API_KEY,
        stream=False,
        reasoning_effort=REASONING_EFFORT,
        client_kwargs={"base_url": BASE_URL, "timeout": 300},
        generate_kwargs={"temperature": 0.3},
    )


def make_agent(name: str, sys_prompt: str) -> ReActAgent:
    agent = ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=make_model(),
        formatter=OpenAIMultiAgentFormatter(),
        toolkit=Toolkit(),
        memory=InMemoryMemory(),
        max_iters=3,
    )
    agent.register_instance_hook("post_reply", "jsonl_log", log_reply_hook)
    return agent


RESEARCHER_PROMPT = (
    "You are Researcher on a judgement panel reviewing marketing copy. "
    "You check claims: are they specific, verifiable and honest? Flag numbers with "
    "no source, superlatives, and promises the product cannot guarantee. "
    "Judge the copy in front of you, not the product. Be concrete and brief."
)
CRITIC_PROMPT = (
    "You are Critic on a judgement panel reviewing marketing copy. "
    "You judge whether the copy would work on a sceptical buyer: clarity, tone, "
    "credibility, and whether the call to action is earned by what came before. "
    "Disagree with the other panellist when you have a reason. Be concrete and brief."
)
CHAIR_PROMPT = (
    "You are Chair of a judgement panel. You do not assess the copy yourself first. "
    "You read the independent verdicts, listen to one exchange round, then record "
    "a chaired verdict that states what was agreed, what was disputed, how you "
    "resolved it, and any dissent you overruled. Do not average; decide."
)


def moderator(text: str) -> Msg:
    return Msg(name="Moderator", content=text, role="user")


def verdict_line(name: str, msg: Msg) -> str:
    v = msg.metadata or {}
    if v:
        return (f"{name}: {v.get('verdict', '?').upper()} (confidence {v.get('confidence', '?')}/5). "
                f"Top issue: {v.get('top_issue', '')} Reasoning: {v.get('reasoning', '')}")
    return f"{name}: {msg.get_text_content()}"


async def run_panel(copy_text: str) -> None:
    researcher = make_agent("Researcher", RESEARCHER_PROMPT)
    critic = make_agent("Critic", CRITIC_PROMPT)
    chair = make_agent("Chair", CHAIR_PROMPT)
    t_start = time.perf_counter()

    log("Moderator", "system", f"run start model={MODEL} base_url={BASE_URL} "
        f"reasoning_effort={REASONING_EFFORT}", event="run_start")

    brief = moderator(
        "The panel is reviewing the following marketing copy.\n\n---\n"
        f"{copy_text}\n---\n"
    )

    # One hub for the whole panel. Auto-broadcast starts OFF so that phase 1
    # replies stay private to the agent that made them.
    async with MsgHub(
        participants=[researcher, critic, chair],
        announcement=brief,
        enable_auto_broadcast=False,
        name="judgement-panel",
    ) as hub:
        log("Moderator", "user", brief.get_text_content(), event="announcement")

        # ---- Phase 1: independent verdicts, logged before anyone sees them ----
        PHASE["name"] = "1-independent"
        ask = moderator(
            "Record your independent verdict on the copy now. Nobody else has "
            "seen it yet and you have not seen theirs."
        )
        r_verdict = await researcher(ask, structured_model=Verdict)
        c_verdict = await critic(ask, structured_model=Verdict)
        # Both verdicts are on disk (post_reply hook) before phase 2 begins.
        log("Moderator", "system", "phase 1 complete: both verdicts written before exchange",
            event="phase_gate")

        # ---- Phase 2: one exchange round through the hub ----
        PHASE["name"] = "2-exchange"
        hub.set_auto_broadcast(True)  # from here every reply reaches every participant
        await hub.broadcast(moderator(
            "Independent verdicts are now published to the panel.\n\n"
            + verdict_line("Researcher", r_verdict) + "\n\n"
            + verdict_line("Critic", c_verdict) + "\n\n"
            "One exchange round. Reply in under 120 words: what you accept from "
            "the other panellist, what you dispute and why, and whether your "
            "verdict changes. The Chair is listening."
        ))
        log("Moderator", "user", "published both verdicts; opened exchange round", event="broadcast")
        await researcher()
        await critic()

        # ---- Phase 3: chaired verdict ----
        PHASE["name"] = "3-chair"
        hub.set_auto_broadcast(False)
        final = await chair(moderator(
            "Exchange round closed. Record the chaired verdict now."
        ), structured_model=ChairVerdict)

    elapsed = time.perf_counter() - t_start
    log("Moderator", "system", f"run complete in {elapsed:.1f}s", event="run_end",
        metadata={"elapsed_s": round(elapsed, 1)})

    print("\n=== Chaired verdict ===")
    print(json.dumps(final.metadata, indent=2, ensure_ascii=False))
    print(f"\nLog: {LOG_PATH}  ({elapsed:.0f}s, model {MODEL})")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("inputs/marketing-copy.md")
    LOG_PATH.parent.mkdir(exist_ok=True)
    asyncio.run(run_panel(path.read_text()))


if __name__ == "__main__":
    main()
