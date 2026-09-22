# IkaCore runtime benchmark

Measures IkaCore's own runtime overhead (framework + httpx client code) across its
whole execution surface, and pins its observable behavior so that optimizations can
be proven behavior-preserving.

```bash
python -m benchmarks                          # full suite + behavior check
python -m benchmarks --quick                  # shorter budget per scenario
python -m benchmarks -k workflow -k staged    # subset by name substring
python -m benchmarks --json base.json         # save a run...
python -m benchmarks --compare base.json      # ...and compare a later run against it
python -m benchmarks --profile staged_3x3 --profile-sort tottime
python -m benchmarks --dump DIR               # write every request/output for diffing
python -m benchmarks --update-golden          # re-pin behavior (intentional changes only)
```

The process exits non-zero if any scenario's behavior differs from `golden.json`.

## How it works

- **No network, full stack.** `fake_provider.py` patches `httpx.HTTPTransport.handle_request`
  and `httpx.AsyncHTTPTransport.handle_async_request`. Everything above the socket runs for
  real: agent loops, payload builders, JSON encoding, httpx client/response handling,
  provider parsers, Codex SSE collection, tool execution, checkpoints and console rendering.
  Console output goes to a null writer, so rendering is measured but terminal I/O is not.
- **Stateless scripted model.** Each agent gets the API key `bench-<name>`. The fake picks a
  per-agent *policy* and answers from the request alone: the tools on offer and the distinct
  tool calls already in the conversation. Like a real model, its answer is a function of
  the conversation, which keeps it deterministic under async fan-out.
- **Net timings.** Time spent inside the fake (parsing requests, building responses) is
  subtracted. `net ms` is the median per iteration; `us/call` divides it by LLM calls.
  `+-%` is the median absolute deviation, a noise indicator.
- **Traffic metrics.** These are exact and deterministic, taken from the recording iteration.
  `summ` counts LLM calls sent without tools (summarization and forced answers). `KB out`
  is request bytes, roughly 4 bytes per token. `conns` counts distinct httpx connection
  pools that carried a request, and each one is a TCP+TLS handshake (about 2 network
  round trips) on a real network. That count is exact while each client has at most one
  request in flight, which holds for IkaCore today. Revisit it if clients become shared.
  These capture the model-side latency and cost the framework controls, which usually
  outweighs its CPU overhead.
- **Behavior fingerprints.** One untimed iteration per scenario records every request body
  (normalized for uuids) and the final output. Their hashes are compared to `golden.json`.
  Mismatches name the agent and request index that changed. Concurrent scenarios
  (`order_insensitive=True`) compare as multisets, because IkaCore joins parallel results in
  completion order.

## Scenarios

| group | covers |
|---|---|
| overhead | bare httpx round trip through the fake (the floor), agent construction |
| agent | single turn on all 8 providers, 10-round tool loops on 5 providers, 16-wide parallel tools, async agent, final-answer retry, max-step exhaustion, 3-stage agent |
| scaling | 40-round conversation with 1 KB outputs, 20 KB prompts + 40 tool schemas |
| multi-agent | subagent delegation, next-agent hand-off with summarization, sync/async workflow chains, async diamond DAG with 3×4 instances |
| durability | SQLite checkpointing every step, HITL interrupt → checkpoint → resume |

Add a scenario in `scenarios.py`, then run `--update-golden -k <name>` to pin it.

## Measuring a change

1. On the base revision: `python -m benchmarks --json base.json`
2. Apply the change and run `python -m benchmarks --compare base.json`. Behavior must stay
   `ok`, and the summary prints the geometric-mean speedup.
3. For small effects, narrow with `-k` and raise `--budget`/`--cap` to cut noise.

## Ablation study

`python -m benchmarks.ablate` knocks out one component at a time with a temporary patch and
measures how much net runtime disappears. IkaCore's source is never edited. Variants run
interleaved with the baseline, and each one is fingerprint-checked (`neutral` / `changed` /
`broken`). A `control` variant patches nothing; its reading is the noise floor, so ignore
savings of about that size. `--list` shows the knockouts, and `-a NAME -k SCENARIO` narrows
the run.

## Live quality A/B

`python -m benchmarks.live_ab --keys PATH --yes` runs checkable multi-round tasks against real
Anthropic, Gemini and DeepSeek models. It compares the pre-fix history replay (arm A) with the
current code (arm B), interleaved at temperature 0. It reports success, errors, input tokens and
latency per arm. Without `--yes` it only prints the plan. Keep the keys file outside the repo.

## Error paths

`python -m benchmarks.faults` has the fake inject provider failures: 429s carrying each kind
of retry hint, 5xx, Anthropic 529, a context-length 400, permanent 4xx, timeouts and
connection errors. Sleeps are virtualised, so the suite runs in seconds but reports how long
IkaCore *would* have waited. It also checks that each scenario ends correctly, either
recovered or failed with the right error. Use `--json` / `--compare` for before/after runs.

## Other frameworks

`python -m benchmarks.compare_langgraph` runs identical workloads on IkaCore and LangGraph
against the same fake. Run it from an environment with `langgraph` and `langchain-openai`
installed. The fake intercepts both `httpx` and `httpx2`, which openai 3.x uses.

## Workflow graphs

`python -m benchmarks.graphs` runs async workflows (chain, diamond with instances, uneven branches,
wide fan-out, layered DAG) with simulated model latency. It reports wall time against the
**critical-path optimum** under IkaCore's own semantics. Each node pays its own calls plus one
context summary if it receives context, instances run in parallel, and a node can start once
its dependencies finish. `eff` = optimum ÷ actual. The benchmark also checks that every node
received all of its parents' results, and exits non-zero otherwise.

## Long-horizon comparison

`python -m benchmarks.long_horizon` runs a 14-agent, ~150-call audit workflow on IkaCore and
LangGraph with identical prompts, tools and models. It scores the final answer on 9 exact facts.

- `--fake`: free and deterministic, with jittered simulated latency (`--latency`, default 2 s).
  It also runs a hand-fused LangGraph graph, and prints the critical-path optimum and the best
  case for a superstep scheduler.
- Live (`--keys PATH --providers deepseek --yes`): spends money, so estimate the cost first.
  Each node is timed from its own requests, so efficiency is judged against the latencies the
  provider actually delivered.

Every run logs each call's input, cached and output tokens and its latency (`--calls-csv`), then
fits `seconds = fixed + per input token + per cached token + per output token` per provider. That
fit shows which lever matters. On DeepSeek it was ~0.85 s fixed + ~3.5 ms per output token, with
no measurable cost for input size.

Run it from an environment with `langgraph`, `langchain-openai` and `langchain-anthropic`. On
Windows, a venv on a long path may need a `subst` drive to install `anthropic`.
