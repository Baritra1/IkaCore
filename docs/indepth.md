## IkaCore Architecture Notes

This document explains how the current IkaCore runtime is structured and where the main execution boundaries live.

### Package Layout

- `src/IkaCore`: agent runtime, stages, workflow graph, logging, checkpoints
- `src/IkaModel`: provider payload builders, transport layer, tool execution loop, summarization helpers
- `src/IkaMem`: short-term and long-term memory abstractions
- `src/IkaTest`: regression tests for runtime behavior

### Core Runtime Model

`IkaBaseAgent` is the public entry point. It mixes together four responsibilities:

- `AgentMemoryMixin`: memory initialization and memory tool helpers
- `AgentToolsMixin`: tool conversion, stage tool assembly, subagent delegation
- `AgentExecutionMixin`: thin execution entry points
- `AgentHelpersMixin`: shared runtime helpers, validation, checkpoint and HITL support

The constructor in [`agents.py`](../src/IkaCore/agents.py) is the contract that the rest of the repo now follows.

### Execution Modes

There are three relevant execution modes.

1. Simple agent execution
   The agent uses its top-level prompt and tool set. This is the narrowest loop.

2. Staged execution
   The agent walks through `IkaStage` objects. Each stage rebuilds the active prompt and tool set while preserving prior transcript context.

3. Workflow execution
   `IkaWorkflow` runs multiple agents and propagates summaries across graph edges.

### Prompt And Transcript Flow

The runtime keeps a normalized `message_history` with:

- `system`
- `first_input`
- `summary`
- `messages`

The important current rule is:

- the stage-specific prompt lives in `first_input`
- the long-lived transcript lives in `messages`
- summarization can compress prior transcript into `summary`

This avoids the older bug where stage 0's first prompt kept being replayed as the active user instruction for later stages.

Each stage also records its opening prompt in `messages` as a `stage_input` entry (HITL answers are
`hitl_input` entries). Providers that replay persisted history (Anthropic, Gemini, DeepSeek) use
these to send earlier stages in order and end on the current stage's instructions; see
`IkaModel/history_replay.py`. Replay also skips entries already present in the live message buffer,
so a round is never sent twice.

### Control Tools

IkaCore models agent control through tool calls instead of hidden side channels.

Important control tools:

- `agent_end`
- `stage_end`
- `change_stage`
- `ask_user`

The control parser path is now unified. The dead duplicate control-call implementation was removed from the active execution path, so provider behavior is interpreted in one place.

### Subagents

Subagents are exposed as tools. The relevant code is in [`agent_tools.py`](../src/IkaCore/agent_tools.py).

Current boundary rules:

- parent agents may delegate work to subagents through a tool call
- delegated task input is treated as untrusted content
- the delegated task is not merged into the subagent's instruction prompt
- original subagent prompt state is restored after execution

That change closed the prompt-injection bug found during the security scan.

### Workflow Semantics

`IkaWorkflow` supports two edge types:

- `next`
- `child`

The semantics are intentionally strict:

- `next` means downstream dependency and summary propagation
- `child` means stage wiring only
- `child` requires `stage_index`
- `child` does not create an auto-executed child node

This behavior is enforced by tests in [`test_workflow_semantics.py`](../src/IkaTest/test_workflow_semantics.py).

Async runs are scheduled as dataflow (`IkaCore/workflow_dataflow.py`). A node starts as soon as
all of its own dependencies have completed, not when every concurrently running node has
finished. Its context (upstream results through `compress_hook`) is prepared once per node, off
the scheduler thread, and shared by all of its instances. Completion semantics are unchanged: a
node completes when all its instances finish, and a failed instance blocks the node's
dependents.

Summary calls along edges are kept to what the semantics need:
- `initial_context` reaches the start node verbatim; it's never paraphrased.
- In both sync and async runs, an edge's upstream result is summarized once, by the consumer.
- Within one run, nodes whose default-hook summary request would be byte-identical (same upstream
  texts, same model and API key) share a single summary call. Custom `compress_hook`s are never
  shared.
- The default hook only summarizes when the merged upstream context exceeds
  `IkaWorkflow(summarize_context_above_tokens=4000)` estimated tokens; shorter context is passed
  through verbatim. Pass `None` to always summarize. Measured on live DeepSeek, a summary call on the
  critical path cost ~6 s (its output tokens), while the input tokens it saved downstream had no
  measurable latency effect, so summaries are worth it only for large contexts.
- A node receives upstream context *in addition to* its own prompt: `inject_workflow_context`
  prepends it, labeled, to the agent's prompt, so simple and staged agents keep their instructions.
  `next_agent` hand-offs work the same way.

### Provider Layer

`IkaModel` separates three concerns:

1. Provider detection and URL selection in [`request_interface.py`](../src/IkaModel/request_interface.py)
2. Provider-specific payload builders under `openai/`, `anthropic/`, `deepseek/`, `gemini/`, `openrouter/`
3. The transport and tool-call loop in [`chat_interface.py`](../src/IkaModel/chat_interface/chat_interface.py) and [`chat_runtime.py`](../src/IkaModel/chat_interface/chat_runtime.py)

Important current behavior:

- OpenAI Responses is the default OpenAI backend
- OpenAI Chat Completions requires `use_responses_api=False`
- Codex is a separate provider that targets `CODEX_API_URL`
- explicit non-OpenAI URLs are respected

Provider requests share one retry policy (`IkaModel/retry_policy.py`). Only 408, 409, 429 and
5xx (including 529) are retried; other errors fail on the first attempt, so a context-length
error goes straight to the summarize-and-resend fallback. Waits follow the provider's hint
(`Retry-After` in seconds or as an HTTP-date, `retry-after-ms`, OpenAI's `x-ratelimit-reset-*`,
Google's `RetryInfo.retryDelay`); without a hint they use exponential backoff from 1 s with
jitter.

Connections are pooled process-wide (`IkaModel/http_pool.py`). Every agent still owns its
`httpx.Client`, but all clients share one keep-alive pool through a transport whose `close()`
does nothing, so owners close their clients as before. Async agents run chat rounds on a
persistent per-thread event loop (`IkaModel/async_runner.py`) so their per-loop pool survives
between rounds.

Codex auth is intentionally not automatic in the request path. The Codex provider treats `BareBoneModel.api_key` as the literal bearer token. Users can pass their own bearer token, or call `IkaModel.codex.codex_auth.get_bearer()` to read and refresh the Codex CLI credentials from `~/.codex/auth.json` or `$CODEX_HOME/auth.json`.

### HITL And Resume

Human-in-the-loop no longer depends on blocking terminal input.

Current behavior:

- `ask_user` raises a structured interrupt
- the interrupt can be checkpointed
- `resume_execution(..., resume_input=...)` continues the run

Regression coverage lives in [`test_hitl_interrupt_resume.py`](../src/IkaTest/test_hitl_interrupt_resume.py).

### Checkpointing

Checkpoint persistence is implemented in [`checkpoint.py`](../src/IkaCore/checkpoint.py).

The current implementation is intentionally simple:

- SQLite-backed storage
- JSON payloads
- resumable run metadata for agent/stage/HITL state

It is materially better than the older coarse resume path, but it is still lighter-weight than full durable orchestration systems such as LangGraph.

### Memory

`IkaMem` exposes:

- `STMemory`
- `LTMemory`
- `Mem0Store`

Agent-side memory tools are assembled by `AgentMemoryMixin`. Long-term save and search now use structured tool schemas instead of the old string-splitting format.

### Maintenance Gates

The repo now has explicit gates for the maintenance risks that used to drift:

- no production function or class may grow to 70 lines or more
- broad `except Exception` boundaries are limited to the current intentional runtime isolation points
- `pyright` checks import/name, call, argument, assignment, return, optional, iterable, and attribute access regressions
- coverage must stay above the configured project threshold
- optional live provider smoke tests are kept separate from deterministic CI

The remaining maturity gap is not local unit behavior. It is broader integration confidence: real provider smoke coverage is opt-in, and long-running degraded-network workflows still need periodic manual or scheduled runs with credentials.
