"""Judgement panel as a content critic, in the score shape Alice's pipeline expects.

Three AgentScope 1.x ReActAgents on one MsgHub, same three phases as panel.py:
independent verdicts (logged before anyone sees them), one exchange round, a
chaired verdict. The chair's verdict carries the four scores Alice's Critic
step produces (vibe_match, hook, clarity, platform_fit, 0..10) plus a note for
each, so it can replace the single-model critic behind a flag.
"""
import asyncio
import datetime as dt
import json
import os
import time
import uuid
from pathlib import Path

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
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "none") or None
EXCHANGE = os.getenv("PANEL_EXCHANGE", "1") not in ("0", "false", "")
RUNS = Path(__file__).parent / "runs"


AXES = ("vibe_match", "hook", "clarity", "platform_fit")


# Flat schemas on purpose: nested objects (scores: {...}, notes: {...}) failed AgentScope's
# structured-output validation three times out of three on the 4B model. Flat fields pass.
class MemberVerdict(BaseModel):
    vibe_match: int = Field(ge=0, le=10, description="0 to 10, how well it matches the requested voice")
    hook: int = Field(ge=0, le=10, description="0 to 10, strength of the opening line")
    clarity: int = Field(ge=0, le=10, description="0 to 10, how easy it is to follow")
    platform_fit: int = Field(ge=0, le=10, description="0 to 10, fit for the channel and format")
    note_vibe_match: str = Field(description="One sentence on vibe_match")
    note_hook: str = Field(description="One sentence on hook")
    note_clarity: str = Field(description="One sentence on clarity")
    note_platform_fit: str = Field(description="One sentence on platform_fit")
    top_issue: str = Field(description="The single most important problem, one sentence")


class ChairVerdict(BaseModel):
    vibe_match: int = Field(ge=0, le=10)
    hook: int = Field(ge=0, le=10)
    clarity: int = Field(ge=0, le=10)
    platform_fit: int = Field(ge=0, le=10)
    note_vibe_match: str = Field(description="One sentence")
    note_hook: str = Field(description="One sentence")
    note_clarity: str = Field(description="One sentence")
    note_platform_fit: str = Field(description="One sentence")
    agreed: str = Field(description="What the two members agreed on")
    disputed: str = Field(description="Where they disagreed and how the chair resolved it")
    dissent: str = Field(description="Any member view the chair overruled, or 'none'")


def _scores(v: dict) -> dict | None:
    return {a: v[a] for a in AXES} if all(a in v for a in AXES) else None


def _notes(v: dict) -> dict | None:
    return {a: v.get(f"note_{a}", "") for a in AXES} if any(f"note_{a}" in v for a in AXES) else None


RESEARCHER = ("You are Researcher on a judgement panel scoring a social post. You check substance: "
              "is the claim specific, is the hook honest, would a practitioner learn something. Score "
              "0 to 10 on each axis. Be concrete and brief.")
CRITIC = ("You are Critic on a judgement panel scoring a social post. You judge it as a sceptical "
          "reader on the platform: does the first line earn the scroll-stop, is it clear, does it fit "
          "the format. Disagree with the other member when you have a reason. Score 0 to 10 on each axis.")
CHAIR = ("You are Chair. You do not score first. You read the independent verdicts, listen to one "
         "exchange, then record final scores with what was agreed, what was disputed and how you "
         "resolved it, and any dissent you overruled. Do not average; decide.")


def _model() -> OpenAIChatModel:
    return OpenAIChatModel(model_name=MODEL, api_key=API_KEY, stream=False, reasoning_effort=REASONING_EFFORT,
                           client_kwargs={"base_url": BASE_URL, "timeout": 300}, generate_kwargs={"temperature": 0.3})


def _agent(name: str, prompt: str, logger) -> ReActAgent:
    a = ReActAgent(name=name, sys_prompt=prompt, model=_model(), formatter=OpenAIMultiAgentFormatter(),
                   toolkit=Toolkit(), memory=InMemoryMemory(), max_iters=4)
    a.set_console_output_enabled(False)
    a.register_instance_hook("post_reply", "jsonl_log", lambda ag, kw, out: logger(ag.name, out))
    return a


def _mod(text: str) -> Msg:
    return Msg(name="Moderator", content=text, role="user")


def _verdict_line(name: str, m: Msg) -> str:
    v = m.metadata or {}
    return f"{name}: scores {json.dumps(_scores(v))}; top issue: {v.get('top_issue', '')}; notes: {json.dumps(_notes(v))}"


async def critique_one(content: str, channel: str, fmt: str, profile: dict | None, run_id: str, index: int) -> dict:
    """Run the three-phase panel on one variant. Returns Alice's critic shape plus panel detail."""
    log_path = RUNS / f"critic-{run_id}.jsonl"
    phase = {"name": "setup"}

    def log(agent: str, out: Msg):
        rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"), "run_id": run_id,
               "variant": index, "phase": phase["name"], "agent": agent,
               "content": out.get_text_content(), "metadata": out.metadata or {}}
        with log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    researcher, critic, chair = _agent("Researcher", RESEARCHER, log), _agent("Critic", CRITIC, log), _agent("Chair", CHAIR, log)
    brief = _mod(f"Channel: {channel}. Format: {fmt}.\n"
                 + (f"Style profile (JSON): {json.dumps(profile)[:1500]}\n" if profile else "")
                 + f"The post under review:\n---\n{content}\n---")
    t0 = time.perf_counter()
    async with MsgHub([researcher, critic, chair], announcement=brief, enable_auto_broadcast=False) as hub:
        phase["name"] = "1-independent"
        ask = _mod("Record your independent scores and notes now. Nobody else has seen them.")
        r = await researcher(ask, structured_model=MemberVerdict)
        c = await critic(ask, structured_model=MemberVerdict)
        if EXCHANGE:
            phase["name"] = "2-exchange"
            hub.set_auto_broadcast(True)
            await hub.broadcast(_mod("Independent verdicts are published.\n\n" + _verdict_line("Researcher", r)
                                     + "\n\n" + _verdict_line("Critic", c)
                                     + "\n\nOne exchange round, under 100 words each: what you accept, what you dispute, "
                                       "and whether your scores change. The Chair is listening."))
            await researcher()
            await critic()
            hub.set_auto_broadcast(False)
        else:
            await chair.observe([_mod(_verdict_line("Researcher", r)), _mod(_verdict_line("Critic", c))])
        phase["name"] = "3-chair"
        final = await chair(_mod("Record the chaired scores and notes now."), structured_model=ChairVerdict)
    v = final.metadata or {}
    return {
        "scores": _scores(v), "notes": _notes(v),
        "panel": {"agreed": v.get("agreed"), "disputed": v.get("disputed"), "dissent": v.get("dissent"),
                  "researcher": _scores(r.metadata or {}), "critic": _scores(c.metadata or {}),
                  "exchange": EXCHANGE, "seconds": round(time.perf_counter() - t0, 1), "log": str(log_path)},
    }


async def critique(variants: list[str], channel: str, fmt: str, profile: dict | None = None) -> dict:
    RUNS.mkdir(exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    results = []
    for i, content in enumerate(variants):  # sequential: one local model, one GPU
        results.append(await critique_one(content, channel, fmt, profile, run_id, i))
    return {"run_id": run_id, "model": MODEL, "critiques": results}


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "Most AI talks about your business. We build agents that act in it, safely. Here is what that took."
    print(json.dumps(asyncio.run(critique([text], "linkedin", "linkedin_post")), indent=2))
