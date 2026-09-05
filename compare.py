"""Same five tool-call prompts through three runners on the same local model.

  python compare.py --runner raw          (openai SDK, one chat.completions call)
  python compare.py --runner agentscope   (AgentScope 1.x ReActAgent, .venv1)
  python compare.py --runner qwen_agent   (Qwen-Agent Assistant, .venv)
  python compare.py --report              (merge runs/compare-*.jsonl into a table)

Each prompt has one correct tool and a set of expected argument values.
Outcomes: correct, bad_args (tool right, values wrong or missing), wrong_tool,
malformed (call present but arguments not parseable / not an object), no_call.
Latency is the wall time of the whole turn; for the two frameworks that is the
full ReAct loop (tool call, tool result, final answer), for raw it is one call.
"""
import argparse
import asyncio
import datetime as dt
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "none") or None
REPEATS = int(os.getenv("COMPARE_REPEATS", "3"))
SET = {"name": "basic"}
RUNS = Path("runs")

# ---------- the tool set (plain JSON schema, shared by all runners) ----------

TOOLS = {
    "get_weather": {
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
    "convert_currency": {
        "description": "Convert an amount from one currency to another using today's rate.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "description": "ISO 4217 code, e.g. GBP"},
                "to_currency": {"type": "string", "description": "ISO 4217 code, e.g. EUR"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    "create_calendar_event": {
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "start_time": {"type": "string", "description": "HH:MM, 24 hour"},
                "duration_minutes": {"type": "integer"},
            },
            "required": ["title", "date", "start_time", "duration_minutes"],
        },
    },
    "search_contacts": {
        "description": "Search the user's contacts by name or company.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    },
    "set_reminder": {
        "description": "Set a reminder for the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "minutes_from_now": {"type": "integer"},
            },
            "required": ["text", "minutes_from_now"],
        },
    },
}

# Canned results so the ReAct loops can finish a turn.
RESULTS = {
    "get_weather": lambda a: {"city": a.get("city"), "temp": 17, "unit": a.get("unit"), "sky": "overcast"},
    "convert_currency": lambda a: {"amount": a.get("amount"), "result": round(float(a.get("amount", 0) or 0) * 1.17, 2), "to": a.get("to_currency")},
    "create_calendar_event": lambda a: {"ok": True, "id": "evt_123", "title": a.get("title")},
    "search_contacts": lambda a: {"matches": [{"name": "Priya Nair", "company": "Ocado"}]},
    "set_reminder": lambda a: {"ok": True, "text": a.get("text")},
}

PROMPTS = [
    {
        "id": "weather",
        "prompt": "What's the weather like in Manchester right now? Use celsius.",
        "tool": "get_weather",
        "expect": {"city": "manchester", "unit": "celsius"},
    },
    {
        "id": "currency",
        "prompt": "How much is 250 pounds sterling in euros today?",
        "tool": "convert_currency",
        "expect": {"amount": 250, "from_currency": "gbp", "to_currency": "eur"},
    },
    {
        "id": "calendar",
        "prompt": "Put a 45 minute meeting called 'Panel review' in my calendar on 2026-09-08 at 14:30.",
        "tool": "create_calendar_event",
        "expect": {"title": "panel review", "date": "2026-09-08", "start_time": "14:30", "duration_minutes": 45},
    },
    {
        "id": "contacts",
        "prompt": "Find my contacts at Ocado.",
        "tool": "search_contacts",
        "expect": {"query": "ocado"},
    },
    {
        "id": "reminder",
        "prompt": "Remind me in 20 minutes to call the dentist.",
        "tool": "set_reminder",
        "expect": {"text": "dentist", "minutes_from_now": 20},
    },
]

# Harder set: indirect units, numbers as words, quotes in strings, optional params,
# arithmetic, and one prompt that must NOT call a tool.
PROMPTS_HARD = [
    {
        "id": "h-zurich",
        "prompt": "It's freezing here in Zürich, what does the thermometer actually say? I only understand fahrenheit.",
        "tool": "get_weather",
        "expect": {"city": "z", "unit": "fahrenheit"},
    },
    {
        "id": "h-quid",
        "prompt": "two hundred and fifty quid to euros please",
        "tool": "convert_currency",
        "expect": {"amount": 250, "from_currency": "gbp", "to_currency": "eur"},
    },
    {
        "id": "h-quotes",
        "prompt": "Book 'Q3 review: numbers & \"risks\"' for the 8th of September 2026, half two in the afternoon, an hour and a quarter.",
        "tool": "create_calendar_event",
        "expect": {"title": "q3 review", "date": "2026-09-08", "start_time": "14:30", "duration_minutes": 75},
    },
    {
        "id": "h-top3",
        "prompt": "Who do I know at Ocado? Just the top 3.",
        "tool": "search_contacts",
        "expect": {"query": "ocado", "limit": 3},
    },
    {
        "id": "h-arith",
        "prompt": "It's 16:25. Remind me at quarter to five to call the dentist.",
        "tool": "set_reminder",
        "expect": {"text": "dentist", "minutes_from_now": 20},
    },
    {
        "id": "h-nocall",
        "prompt": "Don't set anything up or look anything up yet. Just list, in one line, the tools you have.",
        "tool": None,
        "expect": {},
    },
]

def ACTIVE_PROMPTS():
    return PROMPTS_HARD if SET["name"] == "hard" else PROMPTS


SYSTEM = "You are a helpful assistant. Use the provided tools when they fit the request. Call at most one tool."


# ---------- scoring ----------

def _norm(v):
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v.strip().lower())
    return v


def score(case: dict, calls: list[dict]) -> tuple[str, str]:
    """calls: [{"name": str, "arguments": raw str or dict}] in order of appearance."""
    if case["tool"] is None:
        return ("correct", "") if not calls else ("wrong_tool", f"called {calls[0].get('name')} when no tool was wanted")
    if not calls:
        return "no_call", ""
    first = calls[0]
    args = first.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return "malformed", f"unparseable arguments: {args[:80]!r}"
    if not isinstance(args, dict):
        return "malformed", f"arguments not an object: {args!r}"
    if first["name"] != case["tool"]:
        return "wrong_tool", first["name"]
    required = TOOLS[case["tool"]]["parameters"]["required"]
    missing = [k for k in required if k not in args]
    if missing:
        return "malformed", f"missing required {missing}"
    problems = []
    for k, want in case["expect"].items():
        got = _norm(args.get(k))
        if isinstance(want, (int, float)):
            try:
                ok = float(got) == float(want)
            except (TypeError, ValueError):
                ok = False
        elif isinstance(got, str):
            ok = want in got  # substring match, e.g. "dentist" in "call the dentist"
        else:
            ok = False
        if not ok:
            problems.append(f"{k}={args.get(k)!r} (wanted {want!r})")
    if problems:
        return "bad_args", "; ".join(problems)
    return "correct", ""


def write(runner: str, case: dict, rep: int, outcome: str, detail: str, latency: float, calls: list, note: str = "") -> None:
    RUNS.mkdir(exist_ok=True)
    rec = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "runner": runner, "model": MODEL, "set": SET["name"], "prompt_id": case["id"], "rep": rep,
        "outcome": outcome, "detail": detail, "latency_s": round(latency, 2),
        "calls": calls, "note": note,
    }
    with (RUNS / f"compare-{runner}.jsonl").open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{runner}/{SET['name']}] {case['id']:9s} rep{rep} {outcome:10s} {latency:5.1f}s {detail}")


# ---------- runner: raw openai SDK ----------

def run_raw() -> None:
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=300)
    tools = [{"type": "function", "function": {"name": n, **spec}} for n, spec in TOOLS.items()]
    extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
    for rep in range(1, REPEATS + 1):
        for case in ACTIVE_PROMPTS():
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=MODEL, temperature=0.3,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]}],
                tools=tools, tool_choice="auto", **extra,
            )
            latency = time.perf_counter() - t0
            msg = resp.choices[0].message
            calls = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in (msg.tool_calls or [])]
            outcome, detail = score(case, calls)
            write("raw", case, rep, outcome, detail, latency, calls, note=(msg.content or "")[:200])


# ---------- runner: AgentScope 1.x ReActAgent ----------

def run_agentscope() -> None:
    from agentscope.agent import ReActAgent
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.message import Msg
    from agentscope.model import OpenAIChatModel
    from agentscope.tool import Toolkit, ToolResponse
    from agentscope.message import TextBlock

    def make_tool(name):
        def tool_fn(**kwargs) -> ToolResponse:
            return ToolResponse(content=[TextBlock(type="text", text=json.dumps(RESULTS[name](kwargs)))])
        tool_fn.__name__ = name
        return tool_fn

    async def one(case, rep):
        toolkit = Toolkit()
        for name, spec in TOOLS.items():
            schema = {"type": "function", "function": {"name": name, "description": spec["description"], "parameters": spec["parameters"]}}
            toolkit.register_tool_function(make_tool(name), json_schema=schema)
        agent = ReActAgent(
            name="Assistant", sys_prompt=SYSTEM,
            model=OpenAIChatModel(model_name=MODEL, api_key=API_KEY, stream=False,
                                  reasoning_effort=REASONING_EFFORT,
                                  client_kwargs={"base_url": BASE_URL, "timeout": 300},
                                  generate_kwargs={"temperature": 0.3}),
            formatter=OpenAIChatFormatter(), toolkit=toolkit, memory=InMemoryMemory(), max_iters=3,
        )
        agent.set_console_output_enabled(False)
        t0 = time.perf_counter()
        reply = await agent(Msg("user", case["prompt"], "user"))
        latency = time.perf_counter() - t0
        calls, errors = [], []
        for m in await agent.memory.get_memory():
            if isinstance(m.content, list):
                for b in m.content:
                    if b.get("type") == "tool_use":
                        calls.append({"name": b.get("name"), "arguments": b.get("input")})
                    if b.get("type") == "tool_result":
                        txt = json.dumps(b.get("output"))
                        if "Error" in txt or "error" in txt[:200]:
                            errors.append(txt[:160])
        outcome, detail = score(case, calls)
        note = reply.get_text_content() or ""
        if errors:
            detail = (detail + " | tool errors: " + " / ".join(errors)).strip(" |")
        write("agentscope", case, rep, outcome, detail, latency, calls, note=note[:200])

    async def all_():
        for rep in range(1, REPEATS + 1):
            for case in ACTIVE_PROMPTS():
                await one(case, rep)

    asyncio.run(all_())


# ---------- runner: Qwen-Agent Assistant ----------

def run_qwen_agent() -> None:
    import json5
    from qwen_agent.agents import Assistant
    from qwen_agent.tools.base import BaseTool, register_tool

    for name, spec in TOOLS.items():
        def _make(name=name, spec=spec):
            class T(BaseTool):
                description = spec["description"]
                parameters = spec["parameters"]

                def call(self, params, **kwargs) -> str:
                    args = json5.loads(params) if isinstance(params, str) else params
                    return json.dumps(RESULTS[name](args))
            T.__name__ = name
            return T
        register_tool(name)(_make())

    gen_cfg = {"temperature": 0.3}
    if REASONING_EFFORT:
        gen_cfg["reasoning_effort"] = REASONING_EFFORT
    llm_cfg = {"model": MODEL, "model_server": BASE_URL, "api_key": API_KEY, "generate_cfg": gen_cfg}
    if os.getenv("QWEN_AGENT_FNCALL_PROMPT_TYPE"):
        gen_cfg["fncall_prompt_type"] = os.environ["QWEN_AGENT_FNCALL_PROMPT_TYPE"]
    if os.getenv("QWEN_AGENT_USE_RAW_API"):
        gen_cfg["use_raw_api"] = True
    runner_name = "qwen_agent" + ("_raw_api" if gen_cfg.get("use_raw_api") else "")

    for rep in range(1, REPEATS + 1):
        for case in ACTIVE_PROMPTS():
            bot = Assistant(llm=llm_cfg, system_message=SYSTEM, function_list=list(TOOLS))
            t0 = time.perf_counter()
            final = []
            for final in bot.run([{"role": "user", "content": case["prompt"]}]):
                pass
            latency = time.perf_counter() - t0
            calls = [{"name": m["function_call"]["name"], "arguments": m["function_call"]["arguments"]}
                     for m in final if m.get("function_call")]
            outcome, detail = score(case, calls)
            note = next((m.get("content", "") for m in reversed(final) if m.get("role") == "assistant" and m.get("content")), "")
            write(runner_name, case, rep, outcome, detail, latency, calls, note=str(note)[:200])


# ---------- report ----------

def report() -> None:
    rows = []
    for p in sorted(RUNS.glob("compare-*.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r.get("set", "basic"), r["runner"])].append(r)
    print("| Set | Runner | Turns | Correct | Bad args | Wrong tool | Malformed | No call | Median latency | p90 latency |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (set_name, runner), rs in sorted(by.items()):
        n = len(rs)
        c = lambda o: sum(1 for r in rs if r["outcome"] == o)
        lat = sorted(r["latency_s"] for r in rs)
        med = lat[len(lat) // 2]
        p90 = lat[min(len(lat) - 1, int(len(lat) * 0.9))]
        print(f"| {set_name} | {runner} | {n} | {c('correct')} ({100*c('correct')//n}%) | {c('bad_args')} | {c('wrong_tool')} | {c('malformed')} | {c('no_call')} | {med:.1f}s | {p90:.1f}s |")
    print("\nFailures:")
    for r in rows:
        if r["outcome"] != "correct":
            print(f"- {r.get('set','basic')} / {r['runner']} / {r['prompt_id']} rep{r['rep']}: {r['outcome']} {r['detail']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", choices=["raw", "agentscope", "qwen_agent"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--set", choices=["basic", "hard"], default="basic")
    a = ap.parse_args()
    SET["name"] = a.set
    if a.report:
        report()
    elif a.runner:
        {"raw": run_raw, "agentscope": run_agentscope, "qwen_agent": run_qwen_agent}[a.runner]()
    else:
        ap.print_help()
