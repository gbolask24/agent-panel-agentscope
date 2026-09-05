# agent-panel-agentscope

A weekend of hands-on work with two Chinese open-source agent frameworks, AgentScope and
Qwen-Agent, on a local Qwen3.5 4B served by Ollama on an 8 GB M1 MacBook. Built 5 and 6
September 2026 as evidence for a conversation about a multi-agent platform with judgement
panels, a message bus and a human-approval boundary.

What is here:

- `panel.py`: a three-agent judgement panel on AgentScope's MsgHub. Verdicts are recorded before
  the agents exchange views, so the log is not an echo. Every message is written to JSONL with
  agent, phase, timestamp and content.
- `compare.py`: the same five tool-call prompts through a raw OpenAI-compatible call, AgentScope's
  ReAct agent and Qwen-Agent's Assistant, on the same local model. Numbers in
  [COMPARISON.md](COMPARISON.md): 100% on the basic set for every runner except Qwen-Agent's
  default mode (0%, server 500s), 66% for all of them on the hard set with identical failures.
- `qwen_agent_fncall.py` and `qwen_agent_mcp.py`: the smallest Qwen-Agent function-calling and
  MCP examples that run against Ollama.
- `bare_toolcall.py`: proves the local model returns a structured tool call through the OpenAI
  SDK before any framework is involved.
- [NOTES.md](NOTES.md): what each framework does well, what I would not use it for, how each
  compares to the OpenAI Agents SDK, and what I would take from the ecosystem for a 34-agent
  platform. Also what broke.

## The panel

Three `ReActAgent`s (Researcher, Critic, Chair) share one `MsgHub`. The run has three phases:

1. Independent verdicts. The hub is opened with `enable_auto_broadcast=False`, so a reply stays
   with the agent that made it. Researcher and Critic each return a structured `Verdict`
   (approve / revise / reject, confidence 1 to 5, top issue, reasoning). A `post_reply` hook writes
   each verdict to the JSONL log the moment it is produced. A `phase_gate` record marks the point
   at which both are on disk and nobody has seen the other's.
2. One exchange round. `hub.set_auto_broadcast(True)`, the moderator broadcasts both verdicts,
   and Researcher and Critic each reply once. Because of the hub, every reply reaches every
   participant, including the Chair, without any hand-written message passing.
3. Chaired verdict. Auto-broadcast goes back off and the Chair returns a structured
   `ChairVerdict`: what was agreed, what was disputed and how it was resolved, reasoning, and any
   dissent overruled.

### Why the verdict is recorded before the exchange

If panellists see each other's view before committing, the second speaker anchors on the first
and the panel converges. The log then contains one opinion written three times, which is useless
as training data for an editing intelligence and useless as an audit trail. Recording first and
exchanging second gives you three things: an honest measure of disagreement (the first run had
Researcher on revise at confidence 5 and Critic on approve at confidence 5), a record of who
changed their mind and why, and a chair verdict that can name the dissent it overruled.

### Running it

```bash
uv venv --python 3.13 .venv1
uv pip install --python .venv1/bin/python "agentscope<2" "mcp<1.10" tqdm openai python-dotenv
cp .env.example .env          # defaults to Ollama at localhost:11434, model qwen3.5:4b
.venv1/bin/python panel.py    # or: panel.py path/to/other-input.md
```

Output goes to `runs/panel-<timestamp>.jsonl`. One run on the M1 takes about 3.5 minutes for six
model calls with thinking switched off.

The panel is pinned to the AgentScope 1.x line (1.0.21) on purpose. The 2.0 line, released
25 May 2026, removed `MsgHub` and the pipeline module. See NOTES.md for what replaced them and why
that matters more than MsgHub does.

### Switching to OpenAI

Edit `.env`: set `LLM_BASE_URL=https://api.openai.com/v1`, `LLM_MODEL` to a model name and
`LLM_API_KEY` to a key. `LLM_REASONING_EFFORT` defaults to `none`, which is what turns off
Qwen3.5's thinking through Ollama; blank it for models that reject the parameter.

## The comparison

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python agentscope "qwen-agent[mcp]" soundfile tqdm python-dateutil openai python-dotenv
.venv1/bin/python compare.py --runner raw
.venv1/bin/python compare.py --runner agentscope
.venv/bin/python compare.py --runner qwen_agent
.venv/bin/python compare.py --report
```

Two virtual environments because AgentScope 1.x and 2.x cannot coexist and Qwen-Agent wants a
newer `mcp` package than AgentScope 1.x tolerates. That fact is itself in the notes.

Qwen-Agent's default mode does not send `tools` to the API. It writes the function signatures
into the system prompt in the Hermes `<tools>` format and parses `<tool_call>` tags back out of
the text. On Ollama with qwen3.5 that collides with Ollama's own qwen3.5 tool-call parser, which
sees the `<tool_call>` text, fails to parse it (`qwen3.5 tool call parsing failed error=EOF` in
the server log) and returns a 500. `QWEN_AGENT_USE_RAW_API=1` switches Qwen-Agent to native tool
calls and everything works. The two Qwen-Agent examples default to that; the comparison records
both modes.

## Qwen-Agent examples

```bash
.venv/bin/python qwen_agent_fncall.py     # one registered tool
.venv/bin/python qwen_agent_mcp.py        # reference time server over stdio MCP (uvx mcp-server-time)
QWEN_AGENT_USE_RAW_API=0 .venv/bin/python qwen_agent_fncall.py   # see the 500
```

## Driving Alice over MCP

Alice (my content studio, github.com/gbolask24/alice) now serves its 13 copilot tools over stdio
MCP from `scripts/mcp-server.ts` in that repo. The server uses the same tool registry the in-app
copilot uses, binds one user, and applies a write policy: `preview` (default) makes destructive
tools return the confirmation card Alice would show a person and change nothing; `allow` runs
them; `deny` refuses them. Two scripts here drive it on local Qwen with the same three prompts
(list LinkedIn generators, create a task, list tasks):

```bash
.venv/bin/python alice_qwen_agent_mcp.py    # Qwen-Agent: mcpServers entry, native tool calls
.venv/bin/python alice_agentscope_mcp.py    # AgentScope 2.0: MCPClient + Agent + permission engine
```

Both completed all three turns on the first working run (roughly 10 to 30 s a turn). Two things
worth knowing:

- AgentScope 2.0 held the `create_task` call on its own. Its permission engine treats any tool
  not annotated read-only as "ask" in the default mode and emits a `RequireUserConfirmEvent`; the
  reply does not continue until something answers with a `UserConfirmResultEvent`. The script's
  `boundary()` function plays the human. On the first attempt it refused by mistake (AgentScope
  names MCP tools `mcp__alice__create_task`) and the model retried the refused call four times
  with the same arguments. `ReActConfig(stop_on_reject=True)` ends the attempt on a refusal
  instead. That is the "unsafe retries" failure mode and its fix, in one line.
- On both frameworks the 4B model dropped the `channel` argument from the create call while
  saying "for the linkedin channel" in its answer. Arguments that validate are not arguments
  that are complete; the schema allowed the omission, so the task landed in the default channel.

## Model

Ollama 0.33.3, tag `qwen3.5:4b` (Qwen3.5, 4.7B parameters, Q4_K_M, 3.4 GB), Apple M1, 8 GB RAM,
5.3 GiB available to Metal. Thinking is on by default for this tag and only
`reasoning_effort: "none"` turns it off through the OpenAI-compatible endpoint; `think: false`
is rejected there and `/no_think` in the prompt is ignored. With thinking on, a one-sentence
answer took 47 to 68 seconds and never finished inside 600 tokens; with it off, 4 seconds.
