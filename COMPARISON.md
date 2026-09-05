# Tool-call comparison: raw call, AgentScope ReAct, Qwen-Agent, same local model

Model `qwen3.5:4b` on Ollama 0.33.3, Apple M1 8 GB, thinking off (`reasoning_effort: "none"`),
temperature 0.3. Each prompt run three times. Latency is the wall time of the whole turn: one
request for `raw`, the full loop (tool call, tool execution, final answer) for the two
frameworks, so the framework rows carry one extra model call by construction. Raw data is in
`runs/compare-*.jsonl`; regenerate the table with `python compare.py --report`.

Runners:

- `raw`: `openai` SDK, `chat.completions.create` with `tools=`, no loop.
- `agentscope`: AgentScope 1.0.21 `ReActAgent` with the same five tools in a `Toolkit`,
  `OpenAIChatModel` pointed at Ollama.
- `qwen_agent`: Qwen-Agent 0.0.34 `Assistant`, default configuration (prompt-templated tool
  calls, `fncall_prompt_type="nous"`).
- `qwen_agent_raw_api`: the same with `use_raw_api=True` (native tool calls).

## Basic set: five one-tool prompts

| Runner | Turns | Correct | Bad args | Wrong tool | Malformed | No call | Error | Median | p90 |
|---|---|---|---|---|---|---|---|---|---|
| raw | 15 | 15 (100%) | 0 | 0 | 0 | 0 | 0 | 7.2s | 28.9s |
| agentscope | 15 | 15 (100%) | 0 | 0 | 0 | 0 | 0 | 9.2s | 13.7s |
| qwen_agent (default) | 5 | 0 (0%) | 0 | 0 | 0 | 0 | 5 | 13.0s | 20.2s |
| qwen_agent_raw_api | 15 | 15 (100%) | 0 | 0 | 0 | 0 | 0 | 8.2s | 11.7s |

The raw p90 of 28.9s is the first two calls of the run paying a cold model load after Ollama
had unloaded the model; the same prompts ran in 6 to 8 seconds on repeats 2 and 3.

The Qwen-Agent default row is five straight HTTP 500s from Ollama. Qwen-Agent puts the tool
schemas in the system prompt in the Hermes format and asks the model for
`<tool_call>{json}</tool_call>` in its text. Ollama runs its own qwen3.5 tool-call parser over the
output, meets that text, fails (`qwen3.5 tool call parsing failed error=EOF` in the server log)
and aborts the response. Qwen-Agent retried each turn, which is the 13 to 20 seconds. With
`use_raw_api=True` Qwen-Agent sends `tools=` like everyone else and matched the others exactly.
I stopped the default run after one repeat because the failure was deterministic.

## Hard set: idiom, arithmetic, quotes, optional parameter, one must-not-call

| Runner | Turns | Correct | Bad args | Wrong tool | Malformed | No call | Error | Median | p90 |
|---|---|---|---|---|---|---|---|---|---|
| raw | 18 | 12 (66%) | 6 | 0 | 0 | 0 | 0 | 5.7s | 8.8s |
| agentscope | 18 | 12 (66%) | 6 | 0 | 0 | 0 | 0 | 8.3s | 12.6s |
| qwen_agent_raw_api | 18 | 12 (66%) | 6 | 0 | 0 | 0 | 0 | 8.2s | 13.2s |

The six failures are the same two prompts, three times each, identical across all three
runners:

- "half two in the afternoon" became `start_time="14:15"` every time (wanted 14:30).
- "It's 16:25. Remind me at quarter to five" became `minutes_from_now=35` every time (wanted 20;
  the model treated quarter to five as five o'clock).

Everything else on the hard set passed: `Zürich` with fahrenheit inferred from "I only
understand fahrenheit", "two hundred and fifty quid" to `amount=250, from_currency="GBP"`, a
title with nested quotes and an ampersand, `limit=3` from "just the top 3", and the
must-not-call prompt was correctly answered without a tool call on all nine turns.

## What the numbers say

1. On this model and server, native tool calling is already reliable: zero malformed arguments in
   96 framework and raw turns. The failures are comprehension failures (British time idioms and
   clock arithmetic) that no tool layer can fix, and they were bit-for-bit identical across
   runners because the same model saw the same prompt at low temperature.
2. Neither framework changed correctness. AgentScope's ReAct loop and Qwen-Agent in native mode
   produced the same calls as the raw request and cost one to three seconds per turn for the
   second model call that writes the answer.
3. Qwen-Agent's one distinctive feature, owning the tool-call format in the prompt, is exactly the
   thing that broke here, because Ollama also owns it for qwen3.5. The two parsers fight and the
   server wins. That feature earns its place on servers with no parser or a bad one, and has to be
   switched off on servers with a good one. It is a per-deployment decision, not a default.
4. The `limit` optional parameter was sometimes filled (10) and sometimes omitted on the basic
   contacts prompt across all runners. Both are valid; a scorer that demanded a specific value
   would have called that a failure. Argument correctness needs to be judged against the schema
   and the intent, not a golden string.
