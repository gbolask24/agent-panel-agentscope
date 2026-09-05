# Position notes: AgentScope and Qwen-Agent, 5 and 6 September 2026

Written after building the panel in `panel.py`, running the comparison in `compare.py`, and
reading the parts of both frameworks I did not run. Everything below is from the installed
packages and this machine, not from memory of older versions.

## Machine and model

- MacBook Pro, Apple M1, 8 GB RAM, arm64, macOS 26.5. 132 GB free disk.
- Ollama 0.33.3 as a brew service. Tag `qwen3.5:4b`: Qwen3.5, 4.7B parameters, Q4_K_M, 3.4 GB.
  Ollama's `/api/show` lists capabilities completion, vision, tools, thinking. The model card
  says 262,144 context; Ollama picks a 4,096 default from the 5.3 GiB it can see on Metal.
- Thinking is on by default. Through the OpenAI-compatible endpoint the only thing that turns it
  off is `reasoning_effort: "none"`. `think: false` is rejected on `/v1`, `/no_think` in the
  prompt is ignored by Qwen3.5, and the native `/api/chat` honours `think: false`. With thinking
  on, a one-sentence answer ran 47 to 68 seconds and hit a 600-token cap without producing any
  content. With it off, 4 seconds. Every number in this repo is with thinking off.
- First call after five idle minutes pays a 25 to 30 second model load. Ollama unloads by default.
- Frameworks: AgentScope 1.0.21 (panel and ReAct runner), AgentScope 2.0.7.post1 (read, and
  used for the API facts below), Qwen-Agent 0.0.34, openai 2.x, Python 3.13 via uv.

## AgentScope

What it does well. The 1.x MsgHub is a clean broadcast primitive: participants subscribe to each
other, a reply from one is delivered to the rest through `observe`, and auto-broadcast can be
switched off and on inside one context, which is exactly what the pre-exchange verdict rule
needs. Structured output through `structured_model` worked first time on a 4B model, because it
is implemented as a tool call with the schema, not as "please answer in JSON". The hook system
(pre and post reply, observe, print, reasoning, acting) gave me the JSONL audit log in five lines
without touching the agent loop. The 2.0 line, released 25 May 2026, is a different framework
wearing the same name: one `Agent` class with a permission engine consulted on every tool call
(allow, deny, ask, passthrough; modes default, accept_edits, explore, bypass, dont_ask), a
middleware chain with named stages (on_system_prompt, on_model_call, on_reasoning, on_acting,
on_check_permission, on_compress_context, on_reply), context compression with a trigger ratio
and a structured continuation summary, tool-result compaction, a typed event stream
(RequireUserConfirmEvent, UserConfirmResultEvent, RequireExternalExecutionEvent, ToolCall and
ModelCall start and end), a token budget middleware that forces a wrap-up, and an agent service
whose `MessageBus` (in-memory or Redis) has queues, an append-only log, pub/sub, locks with a
TTL, a registry, per-session inboxes and wake-up signals. Agent Team, added 5 June 2026, is
built on that bus: a lead agent creates teammates with tools, and teammates talk through inbox
hint blocks drained before each reasoning step.

What I would not use it for. I would not build a new multi-agent system on the 1.x MsgHub,
because the line that has it is the line that is no longer moving. I would not treat AgentScope
2.0 as a light library either: the interesting parts (bus, team, permission approval loop) live
in the FastAPI agent service and assume you run that service, and the Team mode wants Redis. The
1.x memory story is broad but shallow for what a platform needs: `AsyncSQLAlchemyMemory` gives
you SQLite, Postgres or MySQL keyed by session and user, and `CompressionConfig` gives you an
in-context summary when a token threshold trips, but there is no graph memory and the vector
side is a RAG knowledge base, not agent memory. In 2.0 long-term memory is a middleware
(`AgenticMemoryMiddleware` writing files under a memory directory, plus Mem0 and ReMe adapters),
which is honest about what it is. The fine-tuning loop (`agentscope.tuner`: dataset, judge,
algorithm, workflow, plus a dspy prompt tuner) is a harness that expects an external trainer;
it is relevant to training an editing intelligence on logged judgement only as scaffolding.
Dependency hygiene is poor: 1.0.21 breaks on the current `mcp` package and 2.0.7 imports fail
without `apscheduler` unless you install the service extras.

Against the OpenAI Agents SDK. The Agents SDK gives you handoffs, guardrails, sessions and
tracing in one small package and leaves coordination topology to you. AgentScope 2.0 has an
opinion about topology (a bus, a service, a team) and about approval (a permission engine that
suspends the reply and waits for a confirm event). If the boundary layer is the product, that
opinion saves months. If you want to stay model-agnostic, AgentScope is the better bet on paper:
it ships formatters for OpenAI, Anthropic, Gemini, DashScope, Ollama, DeepSeek, Moonshot and xAI,
and separate multi-agent formatters that tag speaker names when several agents share one
transcript. The Agents SDK's tracing is better and its surface is smaller.

## Qwen-Agent

What it does well. One thing, deliberately: it makes tool calling work on Qwen models whatever
the server does. By default (`use_raw_api=False`, `fncall_prompt_type="nous"`) it does not send
`tools` to the API at all. It writes the function signatures into the system prompt inside
`<tools></tools>`, asks the model to answer with `<tool_call>{"name": ..., "arguments": ...}
</tool_call>`, and parses that back out of the content with json5 (so trailing commas and single
quotes survive), strips half-emitted special tokens, and refuses to parse a tool call that
appears inside a thinking block. That is the Hermes format Qwen was trained on, so the model is
doing what it saw in training rather than what the serving layer's parser expects. `use_raw_api`
flips it to native tool calls when the server's parser is trustworthy, and it flips itself on
for Qwen3-Max. MCP servers are a dict in `function_list`, started as subprocesses by
`MCPManager`, and their tools appear next to Python tools with no other code. `generate_cfg`
passes unknown keys through to the API, which is how `reasoning_effort` reached Ollama.

What I would not use it for. Anything beyond the tool layer. There is no multi-agent primitive
beyond `GroupChat`, no memory beyond the message list and a RAG helper, no permission model, no
event stream, no tracing. It is synchronous generators end to end. It pulls in DashScope and
prints a deprecation warning on import, and the base install is missing five runtime imports
(`soundfile`, `tqdm`, `python-dateutil`, `six`, `json5` arrived only with the `mcp` extra). I
would not run it on a non-Qwen model; the prompt template is the point.

Against the OpenAI Agents SDK. Not the same kind of thing. The Agents SDK assumes the provider
returns well-formed tool calls and builds an agent loop above that assumption. Qwen-Agent exists
because that assumption fails on small or oddly served open models. The right comparison is to
the tool-call parser inside vLLM or Ollama, and Qwen-Agent's answer is "do not trust it, own the
format end to end". On this machine the comparison table shows what that buys and what it costs.

## What I would take for a 34-agent platform

- The bus shape from AgentScope 2.0's `MessageBus`, not MsgHub. The abstract class is almost a
  spec for what LEC describes: append-only log with read and trim, queue push and drain, pub/sub,
  lock with TTL, registry, inbox, wake-up. A single-writer rule is one `try_lock` per agent file
  plus a rule that only the lock holder may `log_append` to that file's stream; the framework
  gives you the lock, not the rule.
- The permission engine's approval loop as the model for the boundary layer: a tool call
  evaluated against allow, deny and ask rules, an `ask` suspending the reply with a
  `RequireUserConfirmEvent`, and the reply resuming when a `UserConfirmResultEvent` arrives.
  That is the confirmation card I already run in Alice, done at the framework level.
- The pre-exchange verdict pattern from `panel.py`, wherever the panel is built. Two lines of
  MsgHub configuration (`enable_auto_broadcast=False`, then `set_auto_broadcast(True)`) are
  the whole mechanism; the log shape (agent, phase, timestamp, structured verdict) is what makes
  the transcript usable as training data.
- Qwen-Agent's Hermes template and json5 parsing as the tool layer whenever a Qwen model is
  served locally, switched by model, not globally.
- `ReplyBudgetControlMiddleware` as the per-reply cost cap, and the context compression
  summary schema (task overview, current state, discoveries, next steps, context to preserve)
  as the handoff document between agents; both are small and worth copying.

Working hypotheses, after the build:

- "MsgHub is the closest open-source primitive to the bus." Rejected as stated. MsgHub is a
  broadcast list inside one process; it was removed in 2.0. The closest primitive is the 2.0
  `MessageBus`, and it is closer than I expected.
- "The finetuning loop is relevant because their editing intelligence is trained on logged
  judgement." Weakly confirmed. `agentscope.tuner` is a harness around an external trainer; the
  relevant part is that the framework already logs reasoning in a shape a judge can score.
- "Qwen-Agent is the tool layer for locally served Qwen and not much beyond." Confirmed.
- "Neither gives you a single-writer rule, reconciliation or self-revising agent files."
  Confirmed. The 2.0 bus gives you locks and an append-only log to build the first two on.
  Versioned, self-revising agent files exist nowhere in either framework.

## What surprised me or broke

- The Python line went 2.0 on 25 May 2026 and removed MsgHub, pipelines and the a2a package.
  The Java line's 2.0 GA followed in July 2026 with the same event stream, permission system and
  middleware, plus distributed session storage (Redis, MySQL, PostgreSQL, OSS, COS) and, in
  August 2026, the AgentScope Service control plane. So the lines did not diverge; they converged
  on the same 2.0 architecture, Python first, with Java ahead on distributed session state.
- `pip install agentscope` gives you 2.0.7 and the tutorials, blog posts and most of what a
  search returns still describe 1.x. The 1.x pin needs `mcp<1.10`.
- Qwen3.5 through Ollama cannot be talked out of thinking by any of the documented Qwen3
  switches; only `reasoning_effort: "none"` works, and it is undocumented for this model.
- The very first structured tool call from the 4B model spelt Manchester "Manchater" in the
  arguments. Arguments that parse are not arguments that are right.
- In the first panel run Researcher and Critic disagreed at confidence 5 each (revise versus
  approve). The chair recorded the overruled dissent. That is the log I wanted and would not have
  had if the exchange had come first.

## Comparison in one paragraph

Full table in [COMPARISON.md](COMPARISON.md). On the five basic prompts, the raw call,
AgentScope's ReAct agent and Qwen-Agent in native mode all scored 15 of 15 with zero malformed
arguments; Qwen-Agent in its default prompt-templated mode scored 0 of 5 because Ollama's own
qwen3.5 parser returns a 500 when it meets Qwen-Agent's `<tool_call>` text. On six harder prompts
all three scored 12 of 18 with the identical two failures every time: "half two" became 14:15
and "quarter to five" from 16:25 became 35 minutes. Those are the model's, not the frameworks'.
Median turn latency was 5.7 to 7.2 seconds raw and 8.2 to 9.2 seconds through either framework,
the difference being the second model call that writes the answer.


## Alice

Alice is a Next.js app on the Vercel AI SDK (`ai` 6.0.191, `@ai-sdk/openai` 3.0.65). The
default `openai` provider reads `OPENAI_BASE_URL`, so the move to local Qwen is four
environment variables and no code: `OPENAI_BASE_URL=http://localhost:11434/v1`,
`OPENAI_API_KEY=ollama`, `OPENAI_MODEL=qwen3.5:4b`, `OPENAI_MODEL_FAST=qwen3.5:4b`. What does not
move: embeddings. The knowledge retriever indexes with `text-embedding-3-small` (1,536
dimensions); pointing it at an Ollama embedding model means a re-index. Alice had tools (13
copilot tools with confirmation cards on writes) but no MCP client or server in its code. It has
one now: `scripts/mcp-server.ts` serves the copilot tools over stdio MCP with a write policy, and
both Qwen-Agent and AgentScope drive it on local Qwen (see README, "Driving Alice over MCP").

What happened when I ran it. Login, dashboard and the copilot all came up. The first prompt
("How many tasks do I have at the moment, and what are they called?") produced the right tool
call, `list_tasks` with scope mine, status all, limit 10, and the tool ran. The turn then died:
`@ai-sdk/openai` 3.x's default `openai(model)` speaks the Responses API, Ollama 0.33 accepts
`/v1/responses` for the first step, and rejects the second step with
`input[2]: unknown input item type: "item_reference"`, which is how the SDK refers back to the
earlier tool call. One env-gated line in `lib/models/index.ts` (`openai.chat(id)` when
`OPENAI_USE_CHAT_COMPLETIONS=1`) routes through Chat Completions instead, and the same prompt
then completed: tool call, "Found 0 tasks" card, and the answer "You have 0 tasks at the
moment." The whole turn took 38 seconds: 22 seconds for the first model call (Alice's system
prompt plus 17 tool schemas is about 2,300 tokens, and prompt processing on the M1 is the cost,
not generation), 10 seconds for the answer, 5 seconds for a follow-up call. On gpt-5.4 the same
turn is two to three seconds. That 2,300-token prompt also sits close to Ollama's 4,096 default
context; a longer thread would be silently truncated from the top, tools first, unless
`num_ctx` is raised. Argument formatting was fine on both attempts; the model chose a `limit`
of 10 the first time and 20 the second, both valid. Nothing was dropped.

The Chat Completions switch, a local run script and the MCP server are committed to the alice
repo; the switch is off unless `OPENAI_USE_CHAT_COMPLETIONS=1`, so production behaviour is
unchanged. The runs used a copy of the local database and a throwaway user, not the real data.

What AgentScope 2.0 added on top of Alice's own gating: its permission engine held the create
call with a `RequireUserConfirmEvent` and would not continue without a `UserConfirmResultEvent`.
With the wrong approval rule the model retried the refused call four times with identical
arguments until `max_iters`; `ReActConfig(stop_on_reject=True)` is the one-line fix. That is
LEC's "unsafe retries" in miniature, and the framework has the switch for it.

